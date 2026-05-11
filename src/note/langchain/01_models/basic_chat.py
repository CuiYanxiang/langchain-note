# ============================================================
# 第 1 阶段：核心组件 — 模型
# ============================================================
# 安装命令：
#   uv add langchain "langchain[anthropic]" "langchain[openai]" python-dotenv
# 运行命令：
#   uv run python 01_models/basic_chat.py
# ============================================================

from __future__ import annotations

import os
import sys
from pprint import pprint
import json
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# 1.1 & 1.2 init_chat_model —— 统一初始化 & Provider
# ============================================================
def demo_basic_chat():
    """基础聊天调用"""
    print("=" * 60)
    print("💬 1.1 & 1.2  init_chat_model 统一初始化 & Provider")
    print("=" * 60)

    # 初始化模型（统一入口）
    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,  # 创造性适中
        max_tokens=500,  # 限制输出长度
    )

    # 构建消息（System + Human）
    messages = [
        SystemMessage(content="你是一个乐于助人的 AI 助手，请用中文回答。"),
        HumanMessage(content="请用 3 个要点总结人工智能的三大伦理挑战。")
    ]

    # 调用模型
    response = model.invoke(messages)

    # 打印结果
    print(f"\n✅ 模型回复:\n{response.content}")
    print(f"\n📊 响应元数据:")
    print(f"   - 模型名称: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"   - 完成原因: {response.response_metadata.get('finish_reason', 'N/A')}")
    print(f"\n💰 Token 消耗:")
    print(f"   - 输入: {response.usage_metadata.get('input_tokens', 'N/A')} tokens")
    print(f"   - 输出: {response.usage_metadata.get('output_tokens', 'N/A')} tokens")
    print(f"   - 总计: {response.usage_metadata.get('total_tokens', 'N/A')} tokens")

    chat_model = ChatOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), model=os.getenv("OPENAI_MODEL"))

    print(f"✅ 初始化 ChatOpenAI 模型: {chat_model.model_name}")

    print()


# ============================================================
# 1.3 模型参数：temperature、max_tokens、streaming
# ============================================================
def demo_model_params():
    print("=" * 60)
    print("1.3  模型参数调优")
    print("=" * 60)

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,  # 0 = 确定性输出，适合事实性任务
        max_tokens=100,  # 限制输出长度
        # model_kwargs={"top_p": 0.9},  # 其他模型特定参数可通过 model_kwargs 传入
    )

    print(f"模型配置: temperature={model.temperature}, max_tokens={model.max_tokens}")
    print()

    # 低 temperature（确定性）
    resp_low = model.invoke("请用一个词回答：天空是什么颜色的？")
    print(f"temperature=0 → {resp_low.content.strip()}")

    # 高 temperature（创造性）
    model_creative = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=1.0,
        max_tokens=100,
    )

    resp_high = model_creative.invoke("请用一句话写一句诗，主题是秋天。")
    print(f"temperature=1.0 → {resp_high.content.strip()}")
    print()


# ============================================================
# 1.4 AIMessage 对象解析
# ============================================================
def demo_aimessage_parse():
    print("=" * 60)
    print("1.4  AIMessage 对象解析")
    print("=" * 60)

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        max_tokens=100,
    )

    # 发送包含 SystemMessage 的多消息请求
    messages = [
        SystemMessage(content="你是一个简洁的助手，回答不超过 50 个字。"),
        HumanMessage(content="用一句中文介绍 LangChain 1.x 的定位。"),
    ]
    response = model.invoke(messages)

    # --- .content / .text：模型生成的文本内容 ---
    # LangChain 1.x 中 .text 是 .content 的别名，两者等价
    print(f"content  → {response.content!r}")
    print(f"text     → {response.text!r}")  # 1.x 新增别名
    print(f"content is text → {response.content is response.text}")

    # --- .response_metadata：模型级元数据 ---
    # 包含 model name、stop reason、request id 等
    print(f"\nresponse_metadata:")
    for key, value in response.response_metadata.items():
        print(f"  {key}: {value}")

    # --- .usage_metadata：Token 用量统计 ---
    # 这是 LangChain 1.x 的标准化 token 统计格式
    if response.usage_metadata:
        print(f"\nusage_metadata:")
        print(f"  input_tokens:  {response.usage_metadata.get('input_tokens')}")
        print(f"  output_tokens: {response.usage_metadata.get('output_tokens')}")
        print(f"  total_tokens:  {response.usage_metadata.get('total_tokens')}")

    # --- .id：消息唯一标识符 ---
    print(f"\nmessage id: {response.id}")

    # --- .type：消息类型 ---
    print(f"message type: {response.type}")

    print()


# ============================================================
# 1.5 同步流式响应：model.stream()
# ============================================================
def demo_sync_stream():
    print("=" * 60)
    print("1.5  同步流式响应：model.stream()")
    print("=" * 60)

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,
        max_tokens=100,
        stream_usage=True,  # 1.x 新参数：在流式末尾返回 token 用量
    )

    print("提问：请用三句话介绍 Python 语言的特点。")
    print("-" * 40)

    # stream() 返回一个生成器，逐块产出 AIMessageChunk
    collected_chunks = []
    for i, chunk in enumerate(model.stream("请用三句话介绍 Python 语言的特点。")):
        # 每个 chunk 也是 AIMessage 类型的子类（AIMessageChunk）
        text_piece = chunk.text if hasattr(chunk, 'text') else chunk.content
        if text_piece:
            print(text_piece, end="", flush=True)
            collected_chunks.append(text_piece)

    print()  # 换行

    # 最后一个 chunk 通常包含 usage_metadata
    last_chunk = collected_chunks[-1] if collected_chunks else None
    print(f"\n收到 {len(collected_chunks)} 个文本块")

    print()


# ============================================================
# 1.6 异步流式响应：model.astream()
# ============================================================
async def demo_async_stream():
    print("=" * 60)
    print("1.6  异步流式响应：model.astream()")
    print("=" * 60)

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,
        max_tokens=200,
        stream_usage=True,
    )

    print("提问：什么是异步编程？请简短回答。")
    print("-" * 40)

    # astream() 返回异步生成器
    chunk_count = 0
    async for chunk in model.astream("什么是异步编程？请简短回答。"):
        text_piece = chunk.text if hasattr(chunk, 'text') else chunk.content
        if text_piece:
            print(text_piece, end="", flush=True)
            chunk_count += 1

    print()
    print(f"\n收到 {chunk_count} 个异步文本块")
    print()


# ============================================================
# 额外演示：使用 SystemMessage + 多消息构建请求
# ============================================================
def demo_multi_message():
    print("=" * 60)
    print("额外演示：SystemMessage + HumanMessage 构建请求")
    print("=" * 60)

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        max_tokens=200,
    )

    messages = [
        SystemMessage(content=(
            "你是一个专业的 Python 导师。"
            "回答时使用中文，保持简洁，每个回答不超过 3 句话。"
        )),
        HumanMessage(content="解释一下列表推导式。"),
    ]
    response = model.invoke(messages)
    print(f"回复: {response.content}")

    # Token 用量
    if response.usage_metadata:
        um = response.usage_metadata
        print(f"Token 用量: 输入={um['input_tokens']}, "
              f"输出={um['output_tokens']}, "
              f"总计={um['total_tokens']}")

    print()


def model_demo():
    # 1. 使用 init_chat_model 统一入口初始化模型 (Static Agent 模型配置方式)
    # 通过修改 model_provider，你可以极其轻松地将应用平滑迁移到其他厂商

    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,  # 控制随机性，Data+AI 场景通常调低以保证稳定性
        max_tokens=256,  # 限制最大输出长度
    )

    # 注: 如果你想动态切换，只需改变参数：
    # model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # 2. 同步调用 (invoke) 与 AIMessage 对象解析
    print("========== 1. 同步调用与 AIMessage 解析 ==========")
    prompt = "简述在企业级应用中，相比于单纯调用大模型API，引入类似 LangChain4j 或 LangGraph 这样的 AI Agent 框架的核心优势是什么？一句话概括。"

    response = model.invoke([HumanMessage(content=prompt)])

    print("💬 文本内容 (.content):\n", response.content)
    print("\n📊 响应元数据 (.response_metadata):\n", response.response_metadata)
    print("\n💰 Token 用量 (.usage_metadata):\n", response.usage_metadata)

    # 3. 流式响应 (stream)
    print("\n========== 2. 流式调用 (Streaming) ==========")
    prompt_stream = "请列出3个目前主流的开源大语言模型名称。"
    print("🌊 流式输出: ", end="", flush=True)

    # stream() 会返回一个迭代器，每次吐出 AIMessageChunk
    for chunk in model.stream([HumanMessage(content=prompt_stream)]):
        print(chunk.content, end="", flush=True)
    print("\n")

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    demo_basic_chat()

    demo_model_params()

    demo_aimessage_parse()

    demo_sync_stream()

    asyncio.run(demo_async_stream())

    demo_multi_message()

    model_demo()

    print("\n" + "=" * 60)
    print("🎉 第 1 阶段全部演示完成！")
    print("=" * 60)
