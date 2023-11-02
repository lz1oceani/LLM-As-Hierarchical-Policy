import os, os.path as osp, re
from copy import deepcopy
from .text_utils import load_text
from hlm import PROMPTS


PROMPT_CACHE = {}


def apply_prompt_template(prompt_name, dataset_name, sample, prompt=None):
    sample = deepcopy(sample)
    if "plan_outputs" in sample:
        sample["plan"] = [re.split(r"\nPlan\: \[(?:[^\[\]])*?\]\n", _)[-1] for _ in sample["plan_outputs"]]

    if prompt_name == "0shot":
        prompt_name += ".txt"

    query_key = prompt_name
    if query_key not in PROMPT_CACHE:
        filename = PROMPTS / prompt_name
        print("Loading prompt", str(filename))
        PROMPT_CACHE[query_key] = load_text(filename)
    if prompt is None:
        prompt = PROMPT_CACHE[query_key]

    for key in sample:
        regex_template = "{" + f"({key}(?:\[(?:[^\[\]]*?)\])*)" + "}"
        for pattern in re.findall(regex_template, prompt):
            target_key = "&left&" + pattern + "&right&"
            prompt = prompt.replace("{" + pattern + "}", target_key)
            if isinstance(sample[key], list):
                if key in ["answer_techniques"]:
                    sample[key] = ", ".join(sample[key])

    prompt = prompt.replace("{", "-left-")
    prompt = prompt.replace("}", "-right-")

    prompt = prompt.replace("&left&", "{")
    prompt = prompt.replace("&right&", "}")

    prompt = prompt.format(**sample)

    prompt = prompt.replace("-left-", "{")
    prompt = prompt.replace("-right-", "}")
    return prompt

