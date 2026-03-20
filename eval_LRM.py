import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from utils.query_utils import query


# read json file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# save to cache
def save_to_cache(item, model_name, cache_dir="cache"):
    cache_dir = os.path.join(cache_dir, model_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    index = item["index"]
    cache_file = os.path.join(cache_dir, f"{index}.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        print(f"save to {cache_file} error: {e}")


# load all cached items
def load_cache(model_name, cache_dir="cache"):
    cache_dir = os.path.join(cache_dir, model_name)
    cached_items = []
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(cache_dir, filename)
                try:
                    item = load_json(file_path)
                    cached_items.append(item)
                except Exception as e:
                    print(f"load cache file {file_path} error: {e}")
    return cached_items


def _backend_from_model(model_name):
    m = model_name.lower()
    if "deepseek" in m and "distill" not in m:
        return "deepseek"
    if "distill" in m and "deepseek" in m:
        return "deepseek_distill"
    if "qwq" in m:
        return "qwq"
    if "qwen3" in m:
        return "qwen3"
    if "claude" in m:
        return "claude"
    if "ernie" in m:
        return "ernie"
    if "grok" in m:
        return "grok3"
    if "glm" in m:
        return "glm"
    return None


# query llm
def query_llm(inputs, args):
    backend = _backend_from_model(args.model)
    return query(inputs, args, backend)


# process single question
def process_question(item, prompt, args):
    question = item["question"]
    query_input = prompt + question if prompt else question
    prediction, think_content = query_llm(
        {"query_input": query_input, "index": item["index"]}, args
    )
    item["prediction"] = prediction
    item["think_content"] = think_content
    if prediction == "" or think_content == "":
        print(f"ERROR: output seems None: {item['index']}")
    if prediction is not None and think_content is not None:
        save_to_cache(item, args.model, args.cache_dir)
    return item


# process all questions in parallel
def process_all_questions(data, prompt, args):
    cached_items = load_cache(args.model, args.cache_dir)
    cached_indices = {item["index"] for item in cached_items}
    new_data = [item for item in data if item["index"] not in cached_indices]
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for _ in tqdm(
            as_completed(
                [executor.submit(process_question, x, prompt, args) for x in new_data]
            ),
            total=len(new_data),
            desc="Generating answers for questions",
            unit="question",
        ):
            pass


# save output
def save_output(results, file_name="output.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=r"Think-Bench.json",
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/generation",
        help="Path to the cache directory",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default="http://127.0.0.1:8080",  # default to llama.cpp server
        help="URL of the LLM API",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="model to use",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        default=False,
        help="Whether to use prompt engineering",
    )
    parser.add_argument(
        "--num_threads", type=int, default=16, help="Number of threads."
    )
    args = parser.parse_args()
    # Non-trivial defaults
    if not hasattr(args, "output_file"):
        args.output_file = f"outputs/{args.model}.json"
    # load data
    file_path = args.input_file
    data = load_json(file_path)
    # prompt engineering
    if args.prompt:
        prompt = "Answer this question with minimal thought:"
    else:
        prompt = ""
    # process all questions
    process_all_questions(data, prompt, args)
    processed_data = load_cache(args.model, args.cache_dir)
    # save output
    save_output(processed_data, args.output_file)
