"""
file utils
"""

import json
import os
import time

import openai
import pandas as pd
from json_repair import repair_json
from openai import OpenAI


def query_gpt(inputs, args):
    """
    Query the GPT API with the given inputs.
    Returns:
        Response (dict[str, str]): the response from GPT API.
        Input ID (str): the id that specifics the input.
    """

    messages = [
        {
            "role": "user",
            "content": inputs["query_input"],
        }
    ]

    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)

    succuss = True
    while succuss:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
            )
            succuss = False
        except openai.RateLimitError as e:
            time.sleep(60)
        except openai.APIConnectionError as e:
            time.sleep(10)
        except openai.OpenAIError as e:
            print(f"ERROR: {e}")
            return f"Unsuccessful: {e.message}"

    return response, inputs["index"]


def save_output(results, dataset_name, file_name="output.json"):
    output_folder = os.path.join("./output", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    json.dump(results, open(os.path.join(output_folder, file_name), "w"))


def read_results(data_path):
    if data_path.endswith(".xlsx"):
        results = pd.read_excel(data_path)
        results = results.to_dict(orient="records")
    elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                results = json.loads(line)  # NOTE: It doesn't work for jsonl
        return results
    else:
        raise ValueError(f"Unsupported file type: {data_path}")


def extract_json_string(text):
    """
    Extract and process JSON string from text.
    Returns None if invalid format.
    """
    if not text:
        return None
    text = text.replace("\\", "\\\\")
    start = text.find("[")
    if start == -1:
        return None
    stack = []
    end = -1
    in_quotes = False
    for i in range(start, len(text)):
        if text[i] == '"':
            if i == 0 or text[i - 1] != "\\":
                in_quotes = not in_quotes
        if in_quotes:
            continue
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
