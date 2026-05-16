# ============================================================
# 流式响应深度演示
# ============================================================
# 安装命令：
#   uv add langchain langchain-openai python-dotenv
# 运行命令：
#   uv run python 01_models/streaming.py
# ============================================================

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    max_tokens=128,  # 限制输出长度
)


# ============================================================
# 模式 1：model.stream() — 同步逐块输出
# ============================================================
def sync_stream_demo():
    print("=" * 60)
    print("模式 1：model.stream() — 同步逐块输出")
    print("=" * 60)

    prompt = "请用四句话介绍 LangChain 框架的核心功能。"
    print(f"提问: {prompt}")
    print("-" * 40)

    full_text = ""
    chunk_count = 0
    final_usage = None

    start = time.time()
    for chunk in model.stream(prompt):
        chunk_count += 1

        # 提取文本内容
        text = chunk.text if hasattr(chunk, "text") else chunk.content
        if text:
            print(text, end="", flush=True)
            full_text += text

        # 检查是否有 usage_metadata（通常在最后一个 chunk）
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            final_usage = chunk.usage_metadata

    elapsed = time.time() - start
    print()  # 换行
    print(f"\n统计: {chunk_count} 个块, 耗时 {elapsed:.2f}s")
    if final_usage:
        print(f"Token 用量: {final_usage}")
    print(f"完整文本长度: {len(full_text)} 字符")
    print()


# ============================================================
# 模式 2：model.astream() — 异步逐块输出
# ============================================================
async def async_stream_demo():
    print("=" * 60)
    print("模式 2：model.astream() — 异步逐块输出")
    print("=" * 60)

    prompt = "什么是 RAG？请用两句话解释。"
    print(f"提问: {prompt}")
    print("-" * 40)

    full_text = ""
    chunk_count = 0

    start = time.time()
    async for chunk in model.astream(prompt):
        chunk_count += 1
        text = chunk.text if hasattr(chunk, "text") else chunk.content
        if text:
            print(text, end="", flush=True)
            full_text += text

    elapsed = time.time() - start
    print()
    print(f"\n统计: {chunk_count} 个块, 耗时 {elapsed:.2f}s")
    print(f"完整文本长度: {len(full_text)} 字符")
    print()


# ============================================================
# 模式 3：model.invoke() — 非流式对比
# ============================================================
def invoke_demo():
    print("=" * 60)
    print("模式 3：model.invoke() — 非流式（对比参考）")
    print("=" * 60)

    prompt = "1 + 1 = ?"
    print(f"提问: {prompt}")

    start = time.time()
    response = model.invoke(prompt)
    elapsed = time.time() - start

    print(f"响应: {response.content.strip()}")
    print(f"耗时: {elapsed:.2f}s （需等待完整响应后才能看到输出）")
    print()


# ============================================================
# 模式 4：多消息 + 流式
# ============================================================
def multi_message_stream_demo():
    print("=" * 60)
    print("模式 4：多消息 + 流式")
    print("=" * 60)

    messages = [
        SystemMessage(content="你是一个翻译助手，只做翻译，不做解释。"),
        HumanMessage(
            content='将以下英文翻译为中文: "The future belongs to those who believe in the beauty of their dreams."'
        ),
    ]

    print("系统提示: 你是一个翻译助手")
    print(
        "用户输入: The future belongs to those who believe in the beauty of their dreams."
    )
    print("-" * 40)

    for chunk in model.stream(messages):
        text = chunk.text if hasattr(chunk, "text") else chunk.content
        if text:
            print(text, end="", flush=True)

    print()
    print()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    sync_stream_demo()
    asyncio.run(async_stream_demo())
    invoke_demo()
    multi_message_stream_demo()

    print("=" * 60)
    print("🎉 流式响应演示全部完成！")
    print("=" * 60)
    print("\n💡 关键总结：")
    print("  - stream():  同步逐块输出，适合 CLI 交互")
    print("  - astream(): 异步逐块输出，适合 Web 服务（FastAPI SSE）")
    print("  - invoke():  等待完整响应，适合后台处理")
    print("  - stream_usage=True: 在流式末尾获取 token 用量统计")
