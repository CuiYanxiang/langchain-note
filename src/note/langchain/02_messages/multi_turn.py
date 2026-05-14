# ============================================================
# 多轮对话历史构建
# ============================================================
# 安装依赖:
#   uv add langchain "langchain-openai" python-dotenv
# 运行:
#   uv run 02_messages/multi_turn.py
# ============================================================

from dotenv import load_dotenv

import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(model=os.getenv("OPENAI_MODEL"), model_provider="openai", base_url=os.getenv("OPENAI_BASE_URL"))

# ----------------------------------------------------------
# 手动构建多轮对话历史
# ----------------------------------------------------------
# 对话历史是一个消息列表，按照 时间顺序 排列
conversation_history = [
    # 第 0 轮：系统指令
    SystemMessage(content="你是一个友好的旅行顾问，专门推荐中国的旅游目的地。"),

    # 第 1 轮
    HumanMessage(content="我想去一个温暖的地方旅游，有什么推荐？"),
]

# 调用模型 —— 传入完整历史
response1 = model.invoke(conversation_history)
print("=== 第 1 轮 ===")
print(f"用户: {conversation_history[-1].content}")
print(f"助手: {response1.content}\n")

# ----------------------------------------------------------
# 将模型回复追加到历史，构建第 2 轮
# ----------------------------------------------------------
conversation_history.append(AIMessage(content=response1.content))
conversation_history.append(HumanMessage(content="那里有什么特色美食？"))

response2 = model.invoke(conversation_history)
print("=== 第 2 轮 ===")
print(f"用户: {conversation_history[-1].content}")
print(f"助手: {response2.content}\n")

# ----------------------------------------------------------
# 继续第 3 轮 —— 模型能记住之前的推荐
# ----------------------------------------------------------
conversation_history.append(AIMessage(content=response2.content))
conversation_history.append(HumanMessage(content="预算大概需要多少？帮我做个 5 天的行程规划"))

response3 = model.invoke(conversation_history)
print("=== 第 3 轮 ===")
print(f"用户: {conversation_history[-1].content}")
print(f"助手: {response3.content}\n")

# ----------------------------------------------------------
# 查看当前对话历史的结构
# ----------------------------------------------------------
conversation_history.append(AIMessage(content=response3.content))
print("=== 对话历史结构 ===")
for i, msg in enumerate(conversation_history):
    preview = msg.content[:40].replace('\n', ' ') if isinstance(msg.content, str) else str(msg.content)[:40]
    print(f"  [{i}] {msg.type:>8}: {preview}...")

print(f"\n  总消息数: {len(conversation_history)}")
