"""
核心模型参数演示
运行: uv run python model_params.py

所需依赖:
# uv add langchain langchain-openai python-dotenv
"""

import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# ── 初始化一个高创意、短输出的模型实例 ──────────────
model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9,
    max_tokens=128,
)

# ── 同步调用（非流式）──────────────────────────────
print("【同步调用】等待完整响应...")
start = time.time()
response = model.invoke("写一句科幻风格的短诗")
print(f"耗时: {time.time() - start:.2f}s")
print(f"内容: {response.content}")
print(f"Token: {response.usage_metadata}\n")

# ── 流式调用（streaming=True 在 invoke 时生效）─────
print("【流式调用】逐字输出：")
start = time.time()
full_text = ""
for chunk in model.stream("写一句科幻风格的短诗"):
    print(chunk.content, end="", flush=True)
    full_text += chunk.content
print(f"\n耗时: {time.time() - start:.2f}s")


question = "写一句关于春天的诗"
# ----------------------------------------------------------
# temperature 对比：0（确定性） vs 1（创造性）
# ----------------------------------------------------------
print("=" * 50)
print("temperature 对比")
print("=" * 50)

for temp in [0.0, 1.0]:
    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temp,
        max_tokens=100,
    )
    response = model.invoke([HumanMessage(content=question)])
    print(f"\n[temperature={temp}]")
    print(f"  {response.content}")

# ----------------------------------------------------------
# max_tokens 控制输出长度
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("max_tokens 对比")
print("=" * 50)

for max_t in [20, 200]:
    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        max_tokens=max_t
    )
    response = model.invoke([HumanMessage(content="详细介绍 LangChain 的架构")])
    print(f"\n[max_tokens={max_t}]")
    print(f"  长度: {len(response.content)} 字符")
    print(f"  内容: {response.content[:100]}...")

# ----------------------------------------------------------
# streaming=True：启用流式（后续 1.6 详细讲解）
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("streaming 参数")
print("=" * 50)

streaming_model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 即使用 invoke，内部也启用了流式通道（对 Callback 更友好）
response = streaming_model.invoke([HumanMessage(content="说一句名言")])
print(f"  响应: {response.content}")
