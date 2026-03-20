import argparse
import json
import os
import re
from functools import lru_cache
from typing import List

import pandas as pd
from evalscope.third_party.thinkbench.tools.llm import request_url
from evalscope.third_party.thinkbench.tools.utils import extract_answer
from evalscope.utils.io_utils import dict_to_json, dump_jsonl_data, jsonl_to_list
from tqdm.contrib.concurrent import thread_map
from transformers import AutoTokenizer

cur_path = os.path.dirname(os.path.abspath(__file__))


class EvalThink:
    def __init__(
        self,
        json_path,
        tokenizer_path,
        split_strategies="llm",
        judge_config=None,
        cache_dir=None,
    ):
        self.cache_dir = cache_dir
        self.json_path = json_path
        self.reformat_template = open(
            os.path.join(cur_path, "prompt/prompt_reformat.txt"), "r"
        ).read()
        self.critique_template = open(
            os.path.join(cur_path, "prompt/prompt_critique.txt"), "r"
        ).read()
        self.switch_tokens = [
            "alternatively",
            "but wait",
            "let me reconsider",
            "another way",
            "another approach",
            "another method",
            "another angle",
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.metrics = [
            "reasoning_tokens",
            "first_correct_tokens",
            "reflection_tokens",
            "token_efficiency",
            "thought_num",
        ]
        self.split_strategies = split_strategies  # split by llm, keywords, separator
        self.judge_config = judge_config
        self.model_parse_file_path = os.path.join(self.cache_dir, "answer_index.jsonl")

    @lru_cache(maxsize=None)
    def cal_tokens(self, text: str):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def process_choice(self, think_part, answer, problem):
        tokens = self.cal_tokens(think_part)
        switch_count = sum(
            think_part.lower().count(token) for token in self.switch_tokens
        )
        useful_tokens = self.cal_tokens(
            self.get_first_correct(think_part, problem, answer)
        )
        reflection_tokens = tokens - useful_tokens
        return tokens, switch_count, useful_tokens, reflection_tokens

    def process_item(self, item):
        try:
            problem = item["question"]
            think_content = item["think_content"]
            answer = item["answer"]
            total_tokens, switch_counts, useful_tokens, reflection_tokens = (
                self.process_choice(think_content, answer, problem)
            )

            avg_tokens = total_tokens  # think content total tokens
            avg_thought_num = switch_counts  # think content switch counts
            avg_token_efficiency = (
                useful_tokens / total_tokens
            )  # efficiency of think content
            avg_useful_tokens = useful_tokens  # first correct tokens
            avg_reflection_tokens = reflection_tokens  # reflection tokens

            result = {
                "tokens": avg_tokens,
                "thought_num": avg_thought_num,
                "token_efficiency": avg_token_efficiency,
                "useful_tokens": avg_useful_tokens,
                "reflection_tokens": avg_reflection_tokens,
                "index": item["index"],
            }
            result.update(item)
            if self.cache_dir is not None:
                json.dump(
                    result,
                    open(f"./{self.cache_dir}/{result['index']}.json", "w"),
                    indent=4,
                )
            return (
                avg_tokens,
                avg_thought_num,
                avg_token_efficiency,
                avg_useful_tokens,
                avg_reflection_tokens,
                item["category"],
            )
        except Exception as e:
            return None, None, None, None, None, None, item["index"], str(e)

    def split_by_llm(self, response, problem) -> List[str]:
        response = response.replace("\n", " ")  # remove newline characters
        prompt = self.reformat_template.format(problem=problem, response=response)
        llm_response = request_url(self.judge_config, prompt)
        return llm_response.split("\n\n")

    def split_by_keywords(self, text) -> List[str]:
        pattern = r"(?=\b(?:{})\b)".format("|".join(map(re.escape, self.switch_tokens)))
        segments = re.split(pattern, text)
        # remove empty segments
        segments = [segment.strip() for segment in segments if segment.strip()]

        return segments if segments else [text]

    def split_by_separator(self, text) -> List[str]:
        return text.split("\n\n")

    def get_answer_index(self, response: List[str], problem: str, answer: str) -> int:
        tagged_response = ""
        for sdx, step in enumerate(response):
            tagged_response += f"<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n"
        tagged_response = tagged_response.strip()

        prompt = self.critique_template.format(
            problem=problem, answer=answer, tagged_response=tagged_response
        )

        llm_response = request_url(self.judge_config, prompt)
        if not llm_response:
            answer_index = -1
        else:
            answer_index = extract_answer(llm_response)

        dump_jsonl_data(
            {"prompt": prompt, "response": llm_response, "answer_index": answer_index},
            self.model_parse_file_path,
            dump_mode="append",
        )

        try:
            answer_index = int(answer_index)
        except Exception:
            answer_index = -1
        return answer_index

    def get_first_correct(self, response: str, problem: str, answer: str) -> str:
        if self.split_strategies == "llm":
            text_list = self.split_by_llm(response, problem)
        elif self.split_strategies == "keywords":
            text_list = self.split_by_keywords(response)
        else:
            text_list = self.split_by_separator(response)

        answer_index = self.get_answer_index(text_list, problem, answer)

        if answer_index == -1:  # no correct answer found
            first_correct = ""
        else:
            first_correct = "\n\n".join(text_list[:answer_index])
        return first_correct

    def evaluate(self, output_dir, workers=128):
        cached_index = []
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith(".json"):
                    cached_index.append(int(os.path.splitext(file)[0]))
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        new_data = [item for item in data if item["index"] not in cached_index]

        _ = list(thread_map(self.process_item, new_data, max_workers=workers))

        # from cache_dir to read existing results
        all_results = []
        for file in os.listdir(self.cache_dir):
            if file.endswith(".json"):
                file_path = os.path.join(self.cache_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result = {
                        "avg_tokens": data.get("tokens"),
                        "avg_thought_num": data.get("thought_num"),
                        "avg_token_efficiency": data.get("token_efficiency"),
                        "avg_useful_tokens": data.get("useful_tokens"),
                        "avg_reflection_tokens": data.get("reflection_tokens"),
                        "category": data.get("category"),
                        "question_type": data.get("question_type"),
                    }
                    all_results.append(result)

        # convert results to dataframe
        df = pd.DataFrame(
            all_results,
            columns=[
                "avg_tokens",
                "avg_thought_num",
                "avg_token_efficiency",
                "avg_useful_tokens",
                "avg_reflection_tokens",
                "category",
                "question_type",
            ],
        )

        # calculate overall metrics
        overall_results = (
            df.drop(columns=["category", "question_type"]).mean().to_dict()
        )

        # calculate category metrics
        category_results = (
            df.drop(columns=["question_type"])
            .groupby("category")
            .mean()
            .to_dict(orient="index")
        )

        # calculate qustion_type metrics
        question_type_results = (
            df.drop(columns=["category"])
            .groupby("question_type")
            .mean()
            .to_dict(orient="index")
        )

        # merge overall and category results
        final_results = {
            "overall": overall_results,
            "category": category_results,
            "question_type": question_type_results,
        }
        # save results to json
        dict_to_json(
            final_results, os.path.join(output_dir, f"think_eval_results.json")
        )

        return final_results


def run_task(config, output_dir="outputs", workers=128):
    evaluator = EvalThink(
        **config,
    )
    results = evaluator.evaluate(output_dir, workers)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for judge model",
    )
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for judge model",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="gpt-5.4-mini",
        help="Model name for judge model",
    )
    parser.add_argument(
        "--json_path", type=str, required=True, help="Model output json path"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="openai/gpt-oss-20b",
        help="The tokenizer path",
    )
    parser.add_argument(
        "--split_strategies",
        type=str,
        default="separator",
        choices=["llm", "keywords", "separator"],
        help="The policy to split the response",
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of workers for evaluation"
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    judge_config = dict(
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        model_name=args.judge_model_name,
    )

    model_config = dict(
        json_path=args.json_path,
        tokenizer_path=args.tokenizer_path,
        split_strategies=args.split_strategies,
        judge_config=judge_config,
        cache_dir=args.cache_dir,
    )

    # evaluate the model efficiency
    run_task(model_config, output_dir=args.output_dir, workers=args.workers)
