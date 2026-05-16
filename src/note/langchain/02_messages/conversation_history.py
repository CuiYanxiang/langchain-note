"""
多轮对话历史构建
运行: uv run python 02_messages/conversation_history.py
所需依赖:
# uv add langchain langchain-openai python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ── 初始化对话历史（SystemMessage 只在开头放一次）───
history = [SystemMessage(content="你是一位耐心的编程导师，用中文回答。")]

# ── 模拟 3 轮对话 ──────────────────────────────────
conversations = [
    "什么是 Python 的装饰器？",
    "能给一个带参数的装饰器例子吗？",
    "这个例子中 functools.wraps 的作用是什么？",
]

for i, user_input in enumerate(conversations, 1):
    print(f"\n{'=' * 50}")
    print(f"🧑 第 {i} 轮用户: {user_input}")

    # 1. 追加用户消息
    history.append(HumanMessage(content=user_input))

    # 2. 调用模型（传入完整历史）
    response = model.invoke(history)

    # 3. 追加模型回复到历史
    history.append(AIMessage(content=response.content))

    print(f"🤖 模型回复: {response.content[:120]}...")
    print(f"📊 当前历史长度: {len(history)} 条消息")

# ── 打印完整对话历史 ───────────────────────────────
print(f"\n{'=' * 50}")
print("📜 完整 3 轮对话历史")
print("=" * 50)
for msg in history:
    prefix = "🧑" if msg.type == "human" else "🤖" if msg.type == "ai" else "⚙️"
    print(f"{prefix} [{msg.type.upper():8}] {msg.content[:80]}...")
