import numpy as np
from tabulate import tabulate
from pathlib import Path
from collections import defaultdict
from mmengine import load


from hlm.utils.metric_utils import filter_na_groups, get_top_k_voting, groups_filter_by
from hlm import RESULTS


def compute_gmr(root, name="GPT-3.5/Retrieval", num=-1):
    example = list(root.values())[0]
    
    if "subgroup_keys" in example:
        subgroup_keys = example["subgroup_keys"]
        group_size = len(example["reasoning_raw_outputs"][subgroup_keys[0]])
        assert group_size == 16
    else:
        group_size = 16
    
    metrics = defaultdict(list)
    for key, example in root.items():
        pred_answers = example["normalized_pred_answers"]
        q_type = example["type"]
        per_sample_correct = example["per_sample_correct"]
        groups = example["groups"]
        groups = {eval(k): v for k, v in groups.items()}
        if "subgroup_keys" in example:
            subgroup_keys = example["subgroup_keys"]
            num_groups = len(subgroup_keys)
            assert len(example["reasoning_raw_outputs"][subgroup_keys[0]]) == group_size
        else:
            num_groups = len(pred_answers) // group_size
            assert num_groups == 4
            
        if num > 0:
            num_groups = min(num_groups, num)
        
        group_splits = []
        gmr = []
        for i in range(num_groups):
            group_i = groups_filter_by(groups, lambda x: i * group_size <= x < (i + 1) * group_size)
            group_i = filter_na_groups(group_i, pred_answers)
            majority_i = get_top_k_voting(group_i, k=1)
            if len(majority_i) == 0:
                maj_correct_i = 0
            else:
                maj_correct_i = np.mean([per_sample_correct[_] for _ in majority_i])
            gmr.append(maj_correct_i)
            group_splits.append(group_i)
        gmr = np.max(gmr)
        metrics[q_type].append(gmr)
        metrics["Overall"].append(gmr)
        
    keys = sorted(metrics.keys())
    keys = [key for key in keys if key != "Overall"] + ["Overall"]
    metrics = {key: [np.mean(metrics[key], axis=0) * 100] for key in keys}
    print(name)
    print(tabulate(metrics, headers="keys", floatfmt=".2f"))


def compute_math_hard_20():
    root = RESULTS / "math_hard_20" / "reasoning_output"
    data = load(root / "gpt-3.5-turbo/sc_0shot.json")
    compute_gmr(data, "GPT-3")
    
    data = load(root / "gpt-3.5-turbo/explore_with_key_techniques.json")
    compute_gmr(data, name="GPT-3.5/Key")
    
    data = load(root / "gpt-3.5-turbo/explore_with_retrieval_question.json")
    compute_gmr(data, name="GPT-3.5/Retrieval")
    print("=" * 128)
    

def compute_math_hard():
    root = RESULTS / "math_hard" / "reasoning_output"
    
    data = load(root / "gpt-3.5-turbo/sc_0shot.json")
    compute_gmr(data, "GPT-3.5")
    
    data = load(root / "gpt-3.5-turbo/explore_with_key_techniques.json")
    compute_gmr(data, name="GPT-3.5/Key")
    
    data = load(root / "gpt-3.5-turbo/explore_with_retrieval_question.json")
    compute_gmr(data, name="GPT-3.5/Retrieval")
    print("=" * 128)
    


if __name__ == '__main__':
    compute_math_hard_20()
    compute_math_hard()
    