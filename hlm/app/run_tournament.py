import random, numpy as np, os.path as osp
from itertools import chain
from tqdm import trange
from mmengine import load, dump
from IPython import embed
from collections import defaultdict
from tabulate import tabulate

from hlm.utils.model_utils import ChatBot, DecodingArguments
from hlm.utils.openai_utils import login_openai, get_total_money
from hlm.utils.prompt_utils import apply_prompt_template
from hlm import RESULTS


def tournament(samples_need_check, fast_acc, baseline_results, num=1, prompt_name="0shot/compare_answer.txt", filename=None):
    assert len(fast_acc) + len(samples_need_check) == len(baseline_results)
    results = {} if filename is None or not osp.exists(filename) else load(filename)
    
    baseline_acc_num = np.sum([np.mean(v["majority_corrects"]) for v in baseline_results.values()])
    print("Baseline num", baseline_acc_num)
    remaining = baseline_acc_num - np.sum(list(fast_acc.values()))
    print("Remaining target", remaining)
    
    if filename is not None:
        print(f"Save to {filename}!")
    
    print(len(samples_need_check), len(fast_acc))
    full_indices = sorted(samples_need_check.keys())
    tqdm_obj = trange(len(full_indices))
    
    our_res = defaultdict(list)
    for example_idx in fast_acc:
        question_type = baseline_results[example_idx]["type"]
        our_res[question_type].append(fast_acc[example_idx])
    baseline_res = defaultdict(list)
    for example_idx in baseline_results:
        question_type = baseline_results[example_idx]["type"]
        baseline_res[question_type].append(np.mean(baseline_results[example_idx]["majority_corrects"]))
    
    baseline = save = 0
    num_call = 0
    
    for idx in tqdm_obj:
        example_idx = full_indices[idx]
        example = samples_need_check[example_idx]
        question = example["question"]
        question_type = baseline_results[example_idx]["type"]
        candidates = example["reasoning_outputs"]
        must_correct = example["must_correct"]
        correctness = example["correctness"]
        assert isinstance(candidates, dict)
        
        if example_idx in results:
            result_i = results[example_idx]
        else:
            candidate_answers = sorted(candidates.keys())
            check_results = []
            best = candidate_answers[0]
            
            for i in range(1, len(candidate_answers)):
                big_rerun = 0
                while True:
                    big_rerun += 1
                    if big_rerun > 3:
                        print("It is wield!")
                        embed()
                        exit(0)
                    
                    prompts = []
                    solution_1s = []
                    solution_2s = []
                    for j in range(num):
                        solution_1 = random.choice(candidates[best])
                        solution_2 = random.choice(candidates[candidate_answers[i]])
                        solution_1s.append(solution_1)
                        solution_2s.append(solution_2)
                        sample_i = {"question": question, "solution1": solution_1, "solution2": solution_2}
                        prompt = apply_prompt_template(prompt_name, "MATH", sample_i)
                        prompts.append(prompt)
                    
                    info_i = {"solution1": solution_1s, "solution2": solution_2s}
                    info_i["prompts"] = prompts
                    outputs = ChatBot.call_model(prompts, decoding_args=default_decoding_args, return_list=False)
                    info_i["outputs"] = outputs
                    # print(outputs[0])
                    split_text = "preferred answer index"
                    
                    # embed()
                    rerun_idx = 0
                    sign = True
                    while True:
                        rerun_idx += 1
                        problematic_indices = []
                        for j, output in enumerate(outputs):
                            if split_text in output.lower():
                                better_index = output.lower().split(split_text)[-1].split("\n")[0].strip(":*")
                            else:
                                problematic_indices.append(j)
                        if len(problematic_indices) == 0:
                            break
                        if rerun_idx >= 2:
                            sign = False
                            break
                        tmp_prompts = [prompts[j] for j in problematic_indices]
                        print(f"Rerun for {len(tmp_prompts)}!")
                        tmp_outputs = ChatBot.call_model(tmp_prompts, decoding_args=default_decoding_args, return_list=False)
                        for j, tmp_output in zip(problematic_indices, tmp_outputs):
                            outputs[j] = tmp_output
                    if sign:
                        break
                
                voting = defaultdict(int)
                processed_outputs = []
                for output in outputs:
                    if split_text in output.lower():
                        better_index = output.lower().split(split_text)[-1].split("\n")[0].strip(":*")
                        if "2" in better_index:
                            opt = 2
                        elif "1" in better_index:
                            opt = 1
                        else:
                            opt = 0
                    else:
                        opt = 1
                    processed_outputs.append(better_index)
                    voting[opt] += 1
                info_i["processed_outputs"] = processed_outputs
                # print(processed_outputs, voting)
                info_i["voting"] = dict(voting)
                if voting[2] > voting[1]:
                    better_index = 2
                else:
                    better_index = 1
                info_i["better_index"] = better_index
                check_results.append(info_i)
                if better_index == 2:
                    best = candidate_answers[i]
            num_call += 1
            results[example_idx] = result_i = [check_results, best]
            print(example_idx, must_correct, len(candidates), example["final_answer"], best, correctness[best])
            if filename is not None:
                dump(results, filename, indent=2)
        save += correctness[result_i[1]]
        our_res[question_type].append(correctness[result_i[1]])
        est_total_money = get_total_money() / (idx + 1) * (len(samples_need_check) - idx - 1)
        tqdm_obj.set_postfix_str(f"{save}/{remaining}, Money: {get_total_money():.2f}, Total money: {est_total_money:.2f}.")
        
    names = sorted(our_res.keys()) + ["Overall"]
    our_res["Overall"] = list(chain(*our_res.values()))
    baseline_res["Overall"] = list(chain(*baseline_res.values()))
    stats = defaultdict(list)
    assert len(our_res["Overall"]) == len(baseline_res["Overall"]) == len(baseline_results)
    stats["Methods"] = ["Baseline", "Ours"]
    for key in names:
        stats[key].append(np.mean(baseline_res[key]) * 100)
        stats[key].append(np.mean(our_res[key]) * 100)
    print(tabulate(stats, headers="keys", floatfmt=".2f"))


def run_for_hard():
    results = RESULTS / "math_hard"
    gpt3_folder = results / "reasoning_output/gpt-3.5-turbo"
    baseline_results = load(gpt3_folder / "sc_0shot.json")
    
    ChatBot.init("gpt-4")
    num = 1
    samples_need_check, fast_acc = load(results / "samples/gpt-3-key_techniques.pkl")
    filename = results / "tournament/gpt-4-key_techniques.json"
    tournament(samples_need_check, fast_acc, baseline_results, num=num, prompt_name="0shot/compare_answer.txt", filename=filename)
    samples_need_check, fast_acc = load(results / "samples/gpt-3-retrieval.pkl")
    filename = results / "tournament/gpt-4-retrieval.json"
    tournament(samples_need_check, fast_acc, baseline_results, num=num, prompt_name="0shot/compare_answer.txt", filename=filename)
    

def run_for_ablation():
    results = RESULTS / "math_hard_20"
    gpt3_folder = results / "reasoning_output/gpt-3.5-turbo"
    baseline_results = load(gpt3_folder / "sc_0shot.json")
    samples_need_check, fast_acc = load(results / "samples/gpt-3-retrieval.pkl")
    
    ChatBot.init("gpt-3.5-turbo")
    for num in [1, 3, 5]:
        filename = results / f"ablation/tournament-gpt-3.5-num={num}.json"
        tournament(samples_need_check, fast_acc, baseline_results, num=num, filename=filename)

    ChatBot.init("gpt-4")
    for num in [1, 3, 5]:
        filename = results / f"ablation/tournament-gpt-4-num={num}.json"
        tournament(samples_need_check, fast_acc, baseline_results, num=num, filename=filename)


if __name__ == '__main__':
    login_openai()
    default_decoding_args = DecodingArguments(
        max_tokens=ChatBot.max_seq_len,
        n=1,
        temperature=0.3,
        frequency_penalty=0,
        presence_penalty=0,
    )
    run_for_ablation()
    run_for_hard()
    