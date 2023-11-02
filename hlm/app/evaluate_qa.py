import os.path as osp, re, numpy as np, time, datetime
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from itertools import chain
from mmengine import load, dump
from math import ceil

from hlm.utils.misc import get_current_time
from hlm.utils.openai_utils import login_openai, DecodingArguments, get_total_money
from hlm.utils.dataset_utils import build_raw_dataset, dict_to_list, list_to_dict, augment_dataset
from hlm.utils.text_utils import rstrip_string
from hlm.utils.model_utils import ChatBot, get_max_length
from hlm.utils.prompt_utils import apply_prompt_template
from hlm.utils.answer_utils import extract_answers, get_answer_type
from hlm.utils.metric_utils import compute_majority_metric, groups_filter_by_sample_n
from hlm.utils.retrieval_utils import build_dpr_index, dpr_query
from hlm import RESULTS, DATASETS


def main():
    global num_call, results
    correct = expectation = recall = total = num_call = 0

    st = time.time()
    for i in range(len(dataset)):
        example = dataset[i]
        final_answer = example.get("final_answer", None)
        example_idx = str(example["example_idx"])
        
        if args.reasoning_fn == "explore":
            if "retrieval" in reasoning_args["type"]:
                total_sample_n = sample_n * reasoning_args["n_retrievals"]
            elif "key-techniques" in reasoning_args["type"]:
                example.pop("answer_techniques_prompt", None)
                techniques = [re.sub(r"^[\d.:]+|[\d.:]+$", "", technique).strip() for technique in example.pop("answer_techniques").split("\n")]
                example.update({"techniques": techniques})
                total_sample_n = sample_n * (len(techniques) if (n_retrievals is None or n_retrievals < 0) else min(len(techniques), n_retrievals))
        else:
            total_sample_n = sample_n

        reasoning_fn(example)
        result_i = results[example_idx]
        result_i.update(example)
        model_outputs = result_i[f"{args.output_key}_outputs"]

        if isinstance(model_outputs, str):
            model_outputs = [model_outputs]
        else:
            model_outputs_new = []
            for output in model_outputs:
                if isinstance(output, list):
                    assert len(output) == 1
                    model_outputs_new += output
                else:
                    model_outputs_new.append(output)
            model_outputs = model_outputs_new
            result_i[f"{args.output_key}_outputs"] = model_outputs

        total += 1
        money = get_total_money()
        ETA = ceil((time.time() - st) / (i + 1) * len(dataset))
        ETA = str(datetime.timedelta(seconds=ETA))
        extra_print_str = ""
        print_str = ""
        assert len(model_outputs) >= total_sample_n, f"Not enough samples: {len(model_outputs)} < {total_sample_n}."

        pred_answers = extract_answers(
            model_outputs,
            final_answer=final_answer,
            answer_type=answer_type,
        )
        result_i["pred_answers"] = pred_answers
        groups = result_i.get("groups", {})
        groups = {int(key): item for key, item in groups.items()}
        
        groups_filter_fn = lambda _: groups_filter_by_sample_n(_, total_sample_n)
        infos = compute_majority_metric(pred_answers, final_answer, answer_type, groups=groups, groups_filter_fn=groups_filter_fn)

        num_full_indices = len(list(chain(*infos["groups"].values())))
        assert num_full_indices == len(pred_answers), f"Wrong number of full indices: {num_full_indices} != {len(pred_answers)}"

        final_results, normalized_final_answer = infos["majority_results"], infos["normalized_final_answer"]

        result_i.update(infos)
        correct += np.mean(infos["majority_corrects"])
        extra_print_str = f" | #Majority={infos['majority_count']}"

        results[example_idx] = result_i
        per_sample_correct = result_i["per_sample_correct"][:total_sample_n]

        recall += np.max(per_sample_correct) > 0
        gt_count = np.sum(per_sample_correct)
        expectation += gt_count / total_sample_n
        print_str = f"idx={example_idx} | pred={final_results} | gt={normalized_final_answer} "
        acc_i = correct / total * 100
        sample_i = expectation / total * 100
        recall_i = recall / total * 100
        print_str += f"| acc={acc_i:.2f}% | sample={sample_i:.2f}% | recall={recall_i:.2f}% |"
        print_str += f" #Samples={total_sample_n}/{len(result_i['pred_answers'])} | #NA={result_i['na_count']} | #GT={gt_count}"
        print_str += extra_print_str
        prefix = f"{get_current_time()} | Iter {i + 1} / {len(dataset)} | ETA: {ETA} | "
        suffix = f" | money={money:.4f} | ETM={money * (len(dataset) - i - 1) / max(num_call, 1):.4f}"
        print(prefix + print_str + suffix)


def one_pass_reasoning(example):
    global num_call, results, decoding_args

    example_idx = example["example_idx"]
    input_key = f"{args.output_key}_input"
    output_key = f"{args.output_key}_outputs"

    existing = []

    decoding_args.n = sample_n
    if example_idx in results and output_key in results[example_idx]:
        if len(results[example_idx][output_key]) >= sample_n and isinstance(results[example_idx][output_key][0], str):
            return
        if isinstance(results[example_idx][output_key][0], str) and sample_n > 1:
            assert reasoning_args["temperature"] > 0, "We does not support greedy decoding!"
            existing = results[example_idx][output_key]

            decoding_args.n = sample_n - len(existing)
    num_call += 1
    model_input = apply_prompt_template(reasoning_args["prompt"], dataset_name, example)
    model_outputs = ChatBot.call_model(model_input, decoding_args=decoding_args)
    model_outputs = existing + model_outputs
    example[input_key] = model_input
    example[output_key] = model_outputs
    results[example_idx] = example
    dump(results, str(result_filename), indent=2)


def prompt_based_exploration(example):
    global num_call, results, decoding_args, args, reasoning_args, prompt_fn, n_retrievals, retrieval_ratio, demo_embeddings

    example_idx = example["example_idx"]
    input_key = f"{args.output_key}_inputs"
    raw_output_key = f"{args.output_key}_raw_outputs"
    output_key = f"{args.output_key}_outputs"

    result_i = results.get(example_idx, {})
    raw_outputs = result_i .get(raw_output_key, {})
    existing_inputs = result_i.get(input_key, {})
    prompts, prompt_infos = {}, {}
    query_key, query_prompts, query_decodings = [], [], []
    potential_keys = []
    if "retrieval" in reasoning_args["type"]:
        global demo_embeddings, demo
        indices, scores = dpr_query(example, demo_embeddings, retrieval_ratio, n_retrievals)

        for i in range(n_retrievals):
            demo = demos[indices[i]]
            demo_idx = demo["example_idx"]
            prompt_infos[demo_idx] = {key: scores[key][i] for key in scores}

            example.update({"demo_question": demo["question"], "demo_answer": demo["answer"]})
            prompt = apply_prompt_template(reasoning_args["prompt"], dataset_name, example)
            example.pop("demo_question")
            example.pop("demo_answer")
            prompts[demo_idx] = prompt
            potential_keys.append(demo_idx)
    elif "key-techniques" in reasoning_args["type"]:
        techniques = example["techniques"]
        if n_retrievals > 0:
            techniques = techniques[:n_retrievals]
        for technique_i in techniques:
            example.update({"key_technique": technique_i})
            prompt = apply_prompt_template(reasoning_args["prompt"], dataset_name, example)
            example.pop("key_technique")
            prompts[technique_i] = prompt
            potential_keys.append(technique_i)

    for key in potential_keys:
        prompt = prompts[key]
        if existing_inputs.get(key, "") != prompt:
            raw_outputs[key] = []
        decoding_args_i = deepcopy(decoding_args)
        n_existing = len(raw_outputs[key])
        if n_existing < decoding_args_i.n:
            decoding_args_i.n -= n_existing
            query_key.append(key)
            query_prompts.append(prompt)
            query_decodings.append(decoding_args_i)

    if len(query_prompts) > 0:
        num_call += 1
        extra_raw_outputs = ChatBot.call_model(query_prompts, decoding_args=query_decodings, return_list=True)
        for key, result in zip(query_key, extra_raw_outputs):
            raw_outputs[key] += result
        result_i["prompt_infos"] = prompt_infos
        result_i[input_key] = prompts
        result_i[raw_output_key] = raw_outputs
    outputs = []
    for key in potential_keys:
        assert len(raw_outputs[key]) >= decoding_args.n, f"Not enough samples: {len(raw_outputs[key])} < {decoding_args.n}."
        outputs += raw_outputs[key][: decoding_args.n]
    result_i["subgroup_keys"] = potential_keys
    result_i[output_key] = outputs
    results[example_idx] = result_i
    dump(results, str(result_filename), indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate QA performance")
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)

    parser.add_argument("--dataset", default="MATH", type=str)
    parser.add_argument("--dataset-args", default="dict(levels=5, subsample=20, include_asy=False)", type=str)
    parser.add_argument("--dataset-name", default="math", type=str)
    parser.add_argument("--subset", default="test", type=str)
    parser.add_argument("--num-examples", default=-1, type=int)
    parser.add_argument("--example-idx", default=None, type=str)

    parser.add_argument("--reasoning-fn", default="one-pass", type=str)
    parser.add_argument("--reasoning-args", default=None, type=str)
    parser.add_argument("--answer-type", default=None, type=str)

    parser.add_argument("--init-file", default=None, type=str)
    parser.add_argument("--output-file", default=None, type=str)
    parser.add_argument("--output-key", default="reasoning", type=str)

    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    answer_type = get_answer_type(dataset_name) if args.answer_type is None else args.answer_type
    
    answer_type = "latex"
    dataset_args = {}
    if args.dataset_args is not None:
        dataset_args = eval(args.dataset_args)
    aug_data = dataset_args.pop("aug", None)
    print(f"Create dataset {dataset_name} with config: {args.dataset_name}[{args.subset}] with args {dataset_args}.")

    dataset = build_raw_dataset(args.dataset, split="test", **dataset_args)
    if aug_data is not None:
        aug_data = load(DATASETS / aug_data)
    augment_dataset(dataset, aug_data)

    print(f"Dataset keys: {list(dataset[0].keys())}")
    print(f"Answer type: {answer_type}")

    if args.example_idx is not None:
        dataset = [_ for _ in dataset if _["example_idx"] == args.example_idx]
    login_openai()
    ChatBot.init(args.model)
    max_seq_len = get_max_length(args.model)

    # Build Agent
    if args.reasoning_args is not None:
        reasoning_args = eval(args.reasoning_args)
    reasoning_args.setdefault("n", 16)
    reasoning_args.setdefault("temperature", 0.7)

    if "prompt" in reasoning_args:
        reasoning_args.setdefault("prompt_tag", reasoning_args["prompt"])
        print("Prompt", reasoning_args["prompt"])
        print("Prompt name", reasoning_args["prompt_tag"])

    total_sample_n = sample_n = reasoning_args["n"] if reasoning_args["temperature"] > 0 else 1
    reasoning_args.setdefault("frequency_penalty", 0)
    reasoning_args.setdefault("presence_penalty", 0)
    decoding_args = DecodingArguments(
        max_tokens=max_seq_len,
        n=sample_n,
        temperature=reasoning_args["temperature"],
        frequency_penalty=reasoning_args["frequency_penalty"],
        presence_penalty=reasoning_args["presence_penalty"],
    )
    print("Reasoning function:", args.reasoning_fn)
    print("Reasoning args:", reasoning_args)
    print("#samples:", total_sample_n)

    result_file_suffix = "_train" if args.subset == "train" else ""
    result_folder = RESULTS / dataset_name / args.model

    if args.reasoning_fn == "one-pass":
        reasoning_fn = one_pass_reasoning
        alg_name = "greedy" if reasoning_args["temperature"] == 0 else "sc"
        result_filename = result_folder / f"{alg_name}_{reasoning_args['prompt_tag']}{result_file_suffix}.json"
    elif args.reasoning_fn == "explore":
        decoding_args.n = sample_n
        n_retrievals = reasoning_args.get("n_retrievals", None)

        if "retrieval" in reasoning_args["type"]:
            retrieval_ratio = reasoning_args["retrieval_ratio"]
            if "demo_kwargs" in reasoning_args:
                demos = list_to_dict(build_raw_dataset(**reasoning_args["demo_kwargs"]))
            else:
                demos = load(reasoning_args["demo_path"])
            demos = dict_to_list(demos)
            if "answer_techniques" in demos[0]:
                for example in demos:
                    answer_techniques = [
                        re.sub(r"^[\d.:]+|[\d.:]+$", "", technique).strip() for technique in example.pop("answer_techniques").split("\n")
                    ]
                    if answer_techniques[0] == "Mathematical Techniques":
                        answer_techniques = answer_techniques[1:]
                    example["answer_techniques"] = "\n".join(answer_techniques)

            retrieval_ratio = {key: item for key, item in retrieval_ratio.items() if item > 0}
            demo_embeddings = {}
            for key in retrieval_ratio:
                demo_embeddings[key] = build_dpr_index(demos, key=key)

        alg_name = rstrip_string(reasoning_args["type"], ".txt").replace("-", "_")
        result_filename = result_folder / f"explore_with_{alg_name}{result_file_suffix}.json"
        reasoning_fn = prompt_based_exploration
    
    if args.output_file is not None:
        print("Use given output file instead of default", args.output_file)
        result_filename = Path(args.output_file)
        
    num_call = 0
    results = load(result_filename) if result_filename.exists() else {}
    dataset_keys = [str(item["example_idx"]) for item in dataset]

    if isinstance(results, list):
        results = {str(item["example_idx"]): item for item in results}
    results = {str(key): item for key, item in results.items()}
    print(f"Begin from {len(results)}, but we have {len(dataset)} examples in total!")
    print("Save results to", str(result_filename), result_filename.exists())
    main()
