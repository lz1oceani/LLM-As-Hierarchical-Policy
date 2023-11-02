import os, warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

os.environ["USE_SYMENGINE"] = "1"
warnings.simplefilter("ignore", SyntaxWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)


import numpy as np, re, time, signal, sympy, scipy
from collections import defaultdict
from numbers import Number
from IPython import embed
from copy import deepcopy
from itertools import chain

# from sympy import Symbol, Eq, simplify, solve
from sympy.parsing.latex import parse_latex
from sympy.core.expr import Expr
from sympy import Interval, conjugate, Abs
from .math_answer_utils import normalize_answer, is_set, is_sympy, is_constant, to_set, is_relation
from math import *


NO_ANSWER = "No answer!"

SKIP_ANSWER_TEMPLATE = [
    "Code cannot be executed!",
    "Code contains infinite loop!",
    "no possible values",
    NO_ANSWER,
]
SKIP_ANSWER_TEMPLATE = SKIP_ANSWER_TEMPLATE + [_.lower() for _ in SKIP_ANSWER_TEMPLATE]

ZERO_ANSWER_TEMPLATE = [
    "doesn't have any money left",
    "used up all of",
]


def check_basics(source, target):
    if not (isinstance(source, (Expr, Number)) and isinstance(target, (Expr, Number))):
        return True
    source_symbols = source.free_symbols if isinstance(source, Expr) else {}
    target_symbols = target.free_symbols if isinstance(target, Expr) else {}

    if source_symbols != target_symbols:
        return False

    try:
        if len(source_symbols) > 0:
            values = {_: np.random.rand() for _ in source_symbols}
            source = source.subs(values)
            target = target.subs(values)
        else:
            source = source.evalf()
            target = target.evalf()
        if not isinstance(source, Number) or not isinstance(target, Number):
            source = abs(source).simplify() if not isinstance(source, Number) else source
            target = abs(target).simplify() if not isinstance(target, Number) else target
        return bool(np.abs(source - target) < 1e-6)
    except:
        pass
    return True


def run_sympy_compare(source, target):
    def has_fn(x):
        for name in ["equals", "compare", "intersect"]:
            if hasattr(x, name):
                return True
        return False

    # print(is_constant(source), is_constant(target))
    # return False
    if is_constant(source) and is_constant(target):
        source = source if isinstance(source, Number) else source.evalf()
        target = target if isinstance(target, Number) else target.evalf()
        try:
            return bool(np.abs(source - target) < 1e-6)
        except:
            return False

    if is_set(source) or is_set(target):
        source = to_set(source)
        target = to_set(target)
    if not has_fn(source):
        source, target = target, source
    assert has_fn(source), [source, target, type(source), type(target)]
    try:
        if hasattr(source, "equals"):  # Work for expressions and points
            if is_relation(source) != is_relation(target):
                return False
            if not is_relation(source) and not check_basics(source, target):
                return False
            ret = source.equals(target)
            ret = False if ret is None else bool(ret)
        elif hasattr(source, "intersect"):
            sign1 = source.intersect(target.complement(sympy.S.Reals)).simplify()
            sign1 = sign1.is_empty or (np.abs(sign1.measure) < 1e-6 and sign1.is_open)
            sign2 = target.intersect(source.complement(sympy.S.Reals)).simplify()
            sign2 = sign2.is_empty or (np.abs(sign2.measure) < 1e-6 and sign2.is_open)
            ret = sign1 and sign2
        elif hasattr(source, "compare"):
            ret = source.compare(target) == 0
    except:
        ret = False
    return bool(ret)


def compare_items(source, target, answer_type="text", need_normalize=True):
    if isinstance(source, (list, tuple)):
        return [compare_items(_, target, answer_type=answer_type, need_normalize=need_normalize) for _ in source]
    if source == "No answer!" or target == "No answer!" or source is None or target is None:
        return False
    if answer_type in ["text", "date", "bool"]:
        return source.lower() == target.lower()

    if isinstance(source, str) and isinstance(target, str):
        if "=" in source and "=" not in target:
            source = source.split("=")[-1]
        if "=" in target and "=" not in source:
            target = target.split("=")[-1]

    if need_normalize:
        source = normalize_answer(source, answer_type)
        target = normalize_answer(target, answer_type)

    if source is None or target is None:
        return (target is None) == (source is None)
    if isinstance(source, str) or isinstance(target, str):
        return str(source).lower() == str(target).lower()

    sympy_compare = lambda: run_sympy_compare(source, target)
    from .misc import timeout_call

    if isinstance(source, Number) and isinstance(target, Number):
        try:
            return bool(np.abs(source - target) < 1e-6)
        except:
            return timeout_call(sympy_compare, timeout=1, default=False)
    elif answer_type == "number":
        return False
    assert answer_type == "latex", f"Invalid answer type: {answer_type}"
    return timeout_call(sympy_compare, timeout=1, default=False)


def minimum_group_representation(groups, answers):
    ret = {}
    for i, indices in groups.items():
        indices = sorted(indices, key=lambda x: len(str(answers[x])))
        ret[indices[0]] = indices
    assert len(list(chain(*ret.values()))) == len(answers), [len(list(chain(*ret.values()))), len(answers)]
    return ret


def group_answers(answers, answer_type="text", need_normalize=False, groups=None):
    # answer = [_ for _ in answers if _ not in SKIP_ANSWER_TEMPLATE]
    if need_normalize:
        answers = [normalize_answer(_, answer_type=answer_type) for _ in answers]

    if answer_type == "text":
        groups = defaultdict(list)
        for idx, ans in enumerate(answers):
            key = "No answer!" if ans.lower() in SKIP_ANSWER_TEMPLATE else ans
            groups[key].append(idx)
        return {item[0]: item for item in groups.values()}

    if groups is None:
        groups = defaultdict(list)
    else:
        tmp_groups = defaultdict(list)
        tmp_groups.update(groups)
        groups = tmp_groups

    processed_indices = list(chain(*groups.values()))
    name_to_index = {str(answers[i]): i for i in groups}
    slow_answers = []
    for i, ans_i in enumerate(answers):
        if i in processed_indices:
            continue
        ans_str_i = str(ans_i) if str(ans_i).lower() not in SKIP_ANSWER_TEMPLATE else NO_ANSWER
        if ans_str_i in name_to_index:
            groups[name_to_index[ans_str_i]].append(i)
            continue
        if len(groups) == 0:
            groups[i].append(i)
            name_to_index[ans_str_i] = i
            continue

        found = False
        slow = False
        existing_indices = sorted(groups.keys(), key=lambda x: len(str(answers[x])))
        
        cnt_i = 0
        for j in existing_indices:
            
            st = time.time()
            sign = compare_items(ans_i, answers[j], answer_type, need_normalize=False)
            time_ij = time.time() - st
            # print("Compare:", ans_str_i, "|", str(answers[j]), "|", sign, "|", time_ij)
            
            if time_ij > 0.15:
                cnt_i += 1
            if cnt_i >= 2 and not sign:  # The answer is too slow to compare
                slow_answers.append(i)
                slow = True
                break
            
            if not sign:
                continue
            groups[j].append(i)
            found = True
            break

        if not found and not slow:
            name_to_index[ans_str_i] = i
            groups[i].append(i)
    for i in slow_answers:
        groups[i].append(i)
    
    assert len(list(chain(*groups.values()))) == len(answers), [len(list(chain(*groups.values()))), len(answers)]
    return minimum_group_representation(groups, answers)


def get_top_k_voting(groups, k=1, complex_groups=True):
    group_keys = sorted(groups.keys())
    group_sizes = [len(groups[key]) for key in group_keys]
    if not complex_groups:
        sorted_idx = np.argsort(group_sizes, kind="stable")[::-1]
        while k < len(group_keys):
            if group_sizes[sorted_idx[k]] != group_sizes[sorted_idx[k - 1]]:
                break
            k += 1
        return [group_keys[_] for _ in sorted_idx[:k]]
    else:
        top_group_size = np.sort(np.unique(group_sizes))[::-1][:k].tolist()
        ret = [[] for i in range(k)]
        for i, size in enumerate(group_sizes):
            if size in top_group_size:
                ret[top_group_size.index(size)].append(group_keys[i])
        return ret[0] if k == 1 else ret
        

def majority_voting(answers, answer_type="text", need_normalize=False, groups=None, groups_filter_fn=None):
    groups = group_answers(answers, answer_type, need_normalize=need_normalize, groups=groups)
    # groups = groups_filter_fn(ret_groups) if groups_filter_fn is not None else ret_groups
    filtered_groups = groups_filter_fn(groups) if groups_filter_fn is not None else groups
    filtered_groups = {key: item for key, item in groups.items() if str(answers[key]).lower() not in SKIP_ANSWER_TEMPLATE}

    if len(filtered_groups) == 0:
        majority_indices, majority_count = [0], len(answers)
    else:
        majority_indices = get_top_k_voting(filtered_groups, 1)
        majority_count = len(filtered_groups[majority_indices[0]])
    return majority_indices, majority_count, groups


def compute_majority_metric(pred_answers, final_answer, answer_type="text", groups=None, groups_filter_fn=None):
    pred_answers = [normalize_answer(_, answer_type=answer_type) for _ in pred_answers]
    final_answer = normalize_answer(final_answer, answer_type=answer_type)

    majority_indices, majority_count, groups = majority_voting(
        pred_answers, answer_type, need_normalize=False, groups=groups, groups_filter_fn=groups_filter_fn
    )
    per_sample_correct = compare_items(pred_answers, final_answer, answer_type, need_normalize=False)

    majority_results = [str(pred_answers[_]) for _ in majority_indices]
    majority_corrects = [per_sample_correct[_] for _ in majority_indices]

    na_count = [len(groups[_]) for _ in groups.keys() if str(pred_answers[_]).lower() in SKIP_ANSWER_TEMPLATE]
    na_count = np.sum(na_count) if len(na_count) > 0 else 0
    return {
        "normalized_pred_answers": [str(_) for _ in pred_answers],
        "normalized_final_answer": str(final_answer),
        "per_sample_correct": per_sample_correct,
        "majority_results": majority_results,
        "majority_corrects": majority_corrects,
        "majority_count": majority_count,
        "na_count": na_count,
        "groups": groups,
    }


def groups_filter_by(groups, fn):
    ret = {key: [_ for _ in item if fn(_)] for key, item in groups.items()}
    return {item[0]: item for key, item in ret.items() if len(item) > 0}


def filter_na_groups(groups, pred_answers):
    return {key: item for key, item in groups.items() if pred_answers[key].lower() not in SKIP_ANSWER_TEMPLATE}


def groups_filter_by_sample_n(groups, sample_n):
    return groups_filter_by(groups, lambda _: _ < sample_n)

