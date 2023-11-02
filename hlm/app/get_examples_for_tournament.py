import numpy as np
from itertools import chain
from pathlib import Path
from collections import defaultdict
from mmengine import load, dump

from hlm import RESULTS
from hlm.utils.metric_utils import filter_na_groups, groups_filter_by, get_top_k_voting


def generate_examples_for_tournament(target, baseline, filename=None):
    target = load(target)
    baseline = load(baseline)
    
    example = list(target.values())[0]
    if "subgroup_keys" in example:
        subgroup_keys = example["subgroup_keys"]
        group_size = len(example["reasoning_raw_outputs"][subgroup_keys[0]])
        assert group_size == 16
    else:
        group_size = 16
    
    up_bound, base = {}, {}
    already_fail = 0
    n_must_pass = need_to_check = problem_to_be_checked = 0
    
    samples_need_check = {}
    fast_acc = {}
    
    for example_idx, example in target.items():
        pred_answers = example["normalized_pred_answers"]
        reasoning_outputs = example["reasoning_outputs"]
        q_type = example["type"]
        per_sample_correct = example["per_sample_correct"]
        groups = example["groups"]
        groups = {eval(k): v for k, v in groups.items()}
        base[example_idx] = base_acc = np.mean(baseline[example_idx]["majority_corrects"])
        
        if "subgroup_keys" in example:
            subgroup_keys = example["subgroup_keys"]
            num_groups = len(subgroup_keys)
            assert len(example["reasoning_raw_outputs"][subgroup_keys[0]]) == group_size
        else:
            num_groups = len(pred_answers) // group_size
            assert num_groups == 4
        
        inv_ans_groups = {}
        for key, item in groups.items():
            for _ in item:
                inv_ans_groups[_] = key
                
        majorities = []
        for i in range(num_groups):
            group_i = groups_filter_by(groups, lambda x: i * group_size <= x < (i + 1) * group_size)
            group_i = filter_na_groups(group_i, pred_answers)
            majority_i = get_top_k_voting(group_i, k=1)
            if len(majority_i) >= 5 or len(majority_i) == 0:  # Too many majority or all the answers do not make sense
                continue
            majorities.append(majority_i)
            
        all_possible = list(chain(*majorities))
        if len(all_possible) == 0:
            up_bound[example_idx] = 0
            fast_acc[example_idx] = 0
            continue
        scores = defaultdict(float)
        for majority in majorities:
            for _ in majority:
                scores[inv_ans_groups[_]] += 1 / len(majority)
        sorted_score = sorted(scores.values(), reverse=True)
        sorted_score = np.unique(sorted_score)
        
        first_majority = [key for key in scores if np.abs(scores[key] - sorted_score[0]) < 1e-4]
        if len(sorted_score) > 1:
            second_majority = [key for key in scores if np.abs(scores[key] - sorted_score[1]) < 1e-4]
        else:
            second_majority = []
        
        if len(scores) == 1 or np.max([per_sample_correct[_] for _ in all_possible]) < 1: # Only one candidate or no candidate is correct
            acc = per_sample_correct[first_majority[0]]
            up_bound[example_idx] = acc
            fast_acc[example_idx] = acc
            if acc < base_acc:
                already_fail += base_acc - acc
            continue
        
        candidate_case = first_majority + second_majority
        up_bound[example_idx] = np.max([per_sample_correct[_] for _ in candidate_case])
        
        if np.max([per_sample_correct[_] for _ in candidate_case]) < 1:
            fast_acc[example_idx] = 0
            already_fail += base_acc
            continue
        
        check_example = {
            "question": example["question"],
            "answer": example["answer"],
            "final_answer": example["final_answer"],
            "reasoning_outputs": {pred_answers[idx]: [reasoning_outputs[_] for _ in groups[idx]] for idx in candidate_case},
            "correctness": {pred_answers[idx]: per_sample_correct[idx] for idx in candidate_case},
            "must_correct": base_acc > 0,
        }
        samples_need_check[example_idx] = check_example
            
        need_to_check += len(first_majority + second_majority)
        problem_to_be_checked += 1
        if base_acc > 0:
            n_must_pass += 1
        
    if filename is not None:
        dump([samples_need_check, fast_acc], filename)
    print("================")
    num_questions = len(baseline)
    assert num_questions == len(target)
    
    print("Total number of questions:", num_questions)
    base_num = np.sum(list(base.values()))
    print(f"Baseline acc: {base_num / num_questions * 100:.2f} [{base_num:.2f}/{num_questions}].")
    max_num = np.sum(list(up_bound.values()))
    print(f"Max potential acc: {max_num / num_questions * 100:.2f} [{max_num}/{num_questions}].")
    print(f"Finished acc: {np.sum(list(fast_acc.values()))}/{len(fast_acc)}.")
    print(f"Already fail: {already_fail:.2f}.")
    print(f"Must pass: {n_must_pass}/{len(up_bound) - len(fast_acc)}.")
    print(f"To achieve baseline: {base_num - np.sum(list(fast_acc.values())):.2f}.")

        
if __name__ == '__main__':
    gpt_3_results = RESULTS / "math_hard/reasoning_output/gpt-3.5-turbo/"
    sample_folder = RESULTS / "math_hard/samples"
    
    baseline = gpt_3_results / "sc_0shot.json"
    
    target = gpt_3_results / "explore_with_retrieval_question.json"
    generate_examples_for_tournament(target=target, baseline=baseline, filename=sample_folder / "gpt-3-retrieval.pkl")

    target = gpt_3_results / "explore_with_key_techniques.json"
    generate_examples_for_tournament(target=target, baseline=baseline, filename=sample_folder / "gpt-3-key_techniques.pkl")
