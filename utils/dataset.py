import json
import os

from json_repair import repair_json
from tqdm import tqdm

from .file_utils import read_results


def extract_json_string(text):
    if not text:
        return None
    text = text.replace("\\", "\\\\")
    start = text.find("[")
    if start == -1:
        return None
    stack = []
    end = -1
    for i in range(start, len(text)):
        if text[i] == "[":
            stack.append(i)
        elif text[i] == "]":
            if stack:
                stack.pop()
                if not stack:
                    end = i
                    break
    if end == -1:
        return None
    json_str = text[start : end + 1]
    return repair_json(json_str)


# make prompt for recall, precision
def make_prompt(name, c, prompt):
    # make ground truth information
    gt_set = []
    if name == "recall":
        cnt = 0
        for f in sorted(c["key_annotation_steps"]):
            if c["key_annotation_steps"][f] is None:
                continue
            for step in c["key_annotation_steps"][f]["logical_conclusion"]:
                if step.strip() == "":
                    continue
                gt_set.append(dict(step_index=cnt, content=step.strip()))
                cnt += 1
    else:
        gt_conclusion = []
        for f in c["key_annotation_steps"]:
            if c["key_annotation_steps"][f] is None:
                continue
            gt_conclusion.extend(c["key_annotation_steps"][f]["logical_conclusion"])

        gt_set = gt_conclusion
        gt_set = [l.strip() for l in gt_set if l.strip() != ""]
    # add information
    prompt = prompt.format(
        question=c["question"],
        answer=c["answer"],
        solution=c["prediction"],
        gt_annotation=json.dumps(gt_set),
    )
    return prompt


# make prompt for recall, precision
def make_prompt_efficiency(c, prompt):
    # add information
    prompt = prompt.format(
        question=c["question"],
        answer=c["answer"],
        think_content=c["think_content"],
    )
    return prompt


# make prompt for recall, precision
def make_prompt_reflection_quality(c, prompt):
    gt_conclusion = []
    for f in c["key_annotation_steps"]:
        if c["key_annotation_steps"][f] is None:
            continue
        gt_conclusion.extend(c["key_annotation_steps"][f]["logical_conclusion"])
    gt_set = gt_conclusion
    gt_set = [l.strip() for l in gt_set if l.strip() != ""]
    # add information
    prompt = prompt.format(
        question=c["question"],
        answer=c["answer"],
        gt_annotation=json.dumps(gt_set),
        think_content=c["think_content"],
    )
    return prompt


def get_dataset_by_path(name, dataset_args):
    # load all the result and its index
    results = read_results(dataset_args["data_path"])  # read either from xlsx or json

    # filter what have already collected in cache
    cached_index = []
    for file in os.listdir(dataset_args["cache_dir"]):
        cached_index.append(int(os.path.splitext(file)[0]))
    filtered_results = []
    for c in results:
        if int(c["index"]) not in cached_index:
            filtered_results.append(c)
    results = filtered_results

    # read the prompt
    with open(dataset_args["prompt_path"], "r") as f:
        prompt = f.read().strip()

    if name in [
        "recall",
        "precision",
    ]:
        # return all uncached data
        return_list = []
        for c in tqdm(results, desc="Processing data"):
            try:
                c["query_input"] = [
                    {"type": "text", "text": make_prompt(name, c, prompt)}
                ]
                return_list.append(c)
            except Exception as e:
                print(
                    f"Error processing data with index {c['index']}: {e}. Skipping this data."
                )
    elif name == "reflection_quality":
        return_list = []
        for c in tqdm(results, desc="Processing data"):
            try:
                c["query_input"] = [
                    {"type": "text", "text": make_prompt_reflection_quality(c, prompt)}
                ]
                return_list.append(c)
            except Exception as e:
                print(
                    f"Error processing data with index {c['index']}: {e}. Skipping this data."
                )
    else:
        raise NotImplementedError

    return return_list
