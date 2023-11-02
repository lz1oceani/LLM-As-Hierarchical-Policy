import os, os.path as osp, json, re, wget, tarfile, random, shutil, wget
import numpy as np

from pathlib import Path
from mmengine import load, dump
from copy import deepcopy
from collections import defaultdict
from hlm import PROCESSED_RAW, DATASETS
from .text_utils import rstrip_string, filter_stripped_lines, unique_texts


def augment_dataset(dataset, aug_data=None):
    if aug_data is not None:
        assert isinstance(aug_data, dict)
        aug_item = list(aug_data.values())[0]
        dataset_item = dataset[0]
        aug_keys = [key for key in aug_item if key not in dataset_item]
        print(f"Update dataset with keys: {aug_keys}!")
        ret = []
        for item in dataset:
            item.update(aug_data[item["example_idx"]])
            ret.append(item)
        from datasets import Dataset
        dataset = Dataset.from_list(ret)
    return dataset


def list_to_dict(data):
    if isinstance(data, dict):
        return data

    assert isinstance(data, (list, tuple)), f"Unknown data type: {type(data)}"
    assert isinstance(data[0], dict), f"Unknown data[0] type: {type(data[0])}"
    assert "example_idx" in data[0], "Missing key: example_idx"

    return {_["example_idx"]: _ for _ in data}


def dict_to_list(data):
    if isinstance(data, (list, tuple)):
        return data
    ret = []
    for key, item in data.items():
        if "example_idx" not in item:
            item["example_idx"] = key
        ret.append(item)
    return ret


def filter_by(data, fn):
    if isinstance(data, dict):
        ret = {key: item for key, item in data.items() if fn(item)}
    else:
        ret = [item for item in data if fn(item)]
    return ret


def MATH(split="test", reprocess=False, subsample=-1, include_asy=True, levels=None, types=None):
    math_root = (DATASETS / "MATH").absolute()
    if not math_root.exists():
        os.makedirs(math_root.parent, exist_ok=True)
        url = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
        filename = str(DATASETS / "MATH.tar")
        print(f"Download MATH dataset to {str(math_root)}")
        wget.download(url, filename)
        with tarfile.open(filename, "r") as f:
            f.extractall(str(DATASETS))
            os.remove(filename)

    if levels is not None and isinstance(levels, int):
        levels = [levels] if levels >= 0 else None

    dataset_path = math_root / split
    tag = f"_split={split}"
    if subsample > 0:
        tag += f"_sample={subsample}"
    if levels is not None:
        tag += f"_levels={levels}"
    tag += f"_asy={include_asy}"

    processed_path = PROCESSED_RAW / f"MATH{tag}.json"

    old_processed_path = PROCESSED_RAW / f"MATH_{split}_{subsample}.json"
    if old_processed_path.exists() and not processed_path.exists():
        print(f"Move {old_processed_path} to {processed_path}")
        shutil.move(old_processed_path, processed_path)

    if processed_path.exists() and not reprocess:
        print("Hit cache of MATH dataset.")
        return load(processed_path)

    data = []
    category_paths = sorted(list(os.listdir(dataset_path)), key=lambda x: x)
    unique_example_idx = 0
    for category_path in category_paths:
        if category_path in ["imgs"] or not (dataset_path / category_path).is_dir():
            continue
        file_paths = sorted(list(os.listdir(dataset_path / category_path)), key=lambda x: int(x.split(".")[0]))
        file_idx = np.arange(len(file_paths))
        np.random.seed(42)
        np.random.shuffle(file_idx)

        count = defaultdict(int)
        for i in file_idx:
            file_path = dataset_path / category_path / file_paths[i]
            data_i = load(file_path)
            try:
                level = int(data_i["level"].split(" ")[-1].strip())
            except:
                print(data_i["level"])
                continue
            if types is not None and data_i["type"] not in types:
                continue

            if levels is not None and level not in levels:
                continue

            answer = data_i.pop("solution").replace(",\\!", "").strip()
            question = data_i.pop("problem").replace(",\\!", "").strip()

            if subsample > 0 and level in count and count[level] >= subsample:
                continue
            if ("[asy]" in answer or "[asy]" in question) and not include_asy:
                # It is better to test VLM..
                continue

            data_i["level"] = level
            data_i["question"] = question
            data_i["answer"] = answer
            data_i["example_idx"] = str(file_path.relative_to(math_root))

            from .math_answer_utils import unwrap_latex_env, remove_starting_text, clean_up_latex_answer
            
            final_answer = unwrap_latex_env(answer, env_name="boxed")
            final_answer = [clean_up_latex_answer(_) for _ in final_answer]
            final_answer = filter_stripped_lines(final_answer)
            final_answer = unique_texts(final_answer)

            if len(final_answer) == 0:
                continue
            elif len(final_answer) > 1:
                continue
            elif len(final_answer) == 1:
                final_answer = final_answer[0]

            count[level] += 1
            data_i["final_answer"] = final_answer
            data.append(data_i)
        unique_example_idx += len(file_paths)
    print(f"Save {len(data)} examples to {processed_path}.")
    dump(data, processed_path, indent=2)
    return data


def build_raw_dataset(dataset_name, **kwargs):
    if osp.exists(dataset_name):
        assert osp.isfile(dataset_name)
        print(f"Load local dataset from {dataset_name}.")
        return load(dataset_name)
    
    create_fn = eval(dataset_name)
    return create_fn(**kwargs)
