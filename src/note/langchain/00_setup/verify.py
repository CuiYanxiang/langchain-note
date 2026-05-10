"""
# uv add "langchain[anthropic]" langchain-openai python-dotenv
# uv run python 00_setup/verify.py
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as pkg_version

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from openai import base_url

MIN_MAJOR = 1
MIN_MINOR = 2


def parse_major_minor(version_text: str) -> tuple[int, int]:
    parts = version_text.split(".")
    if len(parts) < 2:
        raise ValueError(f"Unexpected version string: {version_text}")
    return int(parts[0]), int(parts[1])


def ensure_langchain_version() -> str:
    version_text = pkg_version("langchain")
    major, minor = parse_major_minor(version_text)
    if (major, minor) < (MIN_MAJOR, MIN_MINOR):
        raise RuntimeError(
            f"LangChain version must be >= {MIN_MAJOR}.{MIN_MINOR}, current: {version_text}"
        )
    return version_text


def choose_model() -> tuple[str, str]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if anthropic_key:
        return "anthropic", os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    if openai_key:
        return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    raise RuntimeError(
        "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env."
    )


def main() -> None:
    load_dotenv()

    langchain_version = ensure_langchain_version()
    provider, model_name = choose_model()

    model = init_chat_model(
        model=model_name,
        model_provider=provider,
        base_url=os.getenv("BASE_URL"),
        temperature=0,
        max_tokens=64,
    )

    response = model.invoke("只回答一个词：乒乓")

    print(f"LangChain version: {langchain_version}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Response content: {response.content}")
    print(f"Response metadata: {response.response_metadata}")
    print(f"Usage metadata: {response.usage_metadata}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
