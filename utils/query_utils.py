import re
import time

import openai
from openai import OpenAI

openai.timeout = 3600

SYSTEM_MESSAGE = "You are an AI assistant that helps people solve their questions."

# Backend behavior: stream, reasoning from API field (vs <think> tags), optional extra_body
BACKENDS = {
    "deepseek": {"stream": False, "reasoning_from_api": True},
    "deepseek_distill": {"stream": True, "reasoning_from_api": True},
    "qwq": {"stream": True, "reasoning_from_api": True},
    "qwen3": {
        "stream": True,
        "reasoning_from_api": True,
        "extra_body": {"enable_thinking": True},
    },
    "claude": {"stream": False, "reasoning_from_api": False},
    "grok3": {"stream": False, "reasoning_from_api": True},
    "ernie": {"stream": True, "reasoning_from_api": True},
    "glm": {"stream": True, "reasoning_from_api": False},
}


def extract_think_content(content):
    """Parse <think>...</think> from text; return (content_without_think, reasoning)."""
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(content)
    if match:
        reasoning_content = match.group(1)
        content = think_pattern.sub("", content).strip()
        return content, reasoning_content
    return content, ""


def _call_api(client, messages, args, stream, extra_body=None):
    """Call chat completions with retries. Returns (response or iterator, None) or (None, None) on fatal error."""
    kwargs = {"model": args.model, "messages": messages}
    if stream:
        kwargs["stream"] = True
    if extra_body:
        kwargs["extra_body"] = extra_body

    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            if not stream and response.choices[0].message.content is not None:
                return response, None
            if stream:
                return response, None
        except openai.RateLimitError as e:
            print("Rate limit exceeded, waiting for 60 seconds...")
            print(f"ERROR: {e}")
            time.sleep(60)
        except openai.APIConnectionError as e:
            print("API connection error, waiting for 10 seconds...")
            print(f"ERROR: {e}")
            time.sleep(10)
        except Exception as e:
            if "RequestTimeOut" in str(e):
                print(f"ERROR: {e}")
                time.sleep(5)
            else:
                print(f"ERROR: {e}")
                return None, None


def _parse_nonstream(response, reasoning_from_api):
    """Extract (answer, reasoning) from a non-streaming response."""
    msg = response.choices[0].message
    if reasoning_from_api:
        return msg.content, getattr(msg, "reasoning_content", None) or ""
    return extract_think_content(msg.content or "")


def _parse_stream(completion, reasoning_from_api, inputs_index=None):
    """Extract (answer, reasoning) from a streaming completion."""
    reasoning_content = ""
    answer_content = ""
    is_answering = False

    try:
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if reasoning_from_api:
                if getattr(delta, "reasoning_content", None):
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content and not is_answering:
                        is_answering = True
                    answer_content += delta.content or ""
            else:
                answer_content += delta.content or ""
        if not reasoning_from_api:
            answer_content, reasoning_content = extract_think_content(answer_content)
    except Exception as e:
        if inputs_index is not None:
            print(inputs_index)
        print(f"ERROR: {e}")
        return None, None
    return answer_content, reasoning_content


def query(inputs, args, backend):
    """
    Single entry point for all backends.
    backend: one of deepseek, deepseek_distill, qwq, qwen3, claude, grok3, ernie, glm, or None for general backend.
    Returns (answer_content, reasoning_content).
    """
    if backend is None:
        cfg = {
            "stream": False,
            "reasoning_from_api": True,
            "extra_body": None,
        }
    else:
        cfg = BACKENDS.get(backend)
        if not cfg:
            raise ValueError(f"Backend {backend} not supported")

    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": inputs["query_input"]},
    ]
    extra_body = cfg.get("extra_body")
    stream = cfg["stream"]
    reasoning_from_api = cfg["reasoning_from_api"]

    response, _ = _call_api(
        client, messages, args, stream=stream, extra_body=extra_body
    )
    if response is None:
        return None, None

    if stream:
        return _parse_stream(response, reasoning_from_api, inputs.get("index"))
    return _parse_nonstream(response, reasoning_from_api)
