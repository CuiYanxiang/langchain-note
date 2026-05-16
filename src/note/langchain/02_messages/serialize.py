"""
消息序列化与反序列化
运行: uv run python 02_messages/serialize.py
所需依赖:
# uv add langchain python-dotenv
"""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_to_dict,
    messages_from_dict,
)
import json

# ── 1. 构建一段含所有类型的对话历史 ─────────────────
messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="北京今天天气如何？"),
    AIMessage(
        content="我来查询一下天气信息。",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "北京"}, "id": "call_weather_001"}
        ],
    ),
    ToolMessage(
        content="北京今天晴，25°C，空气质量优。",
        tool_call_id="call_weather_001",
    ),
    AIMessage(content="北京今天晴天，气温 25°C，空气质量优，适合外出。"),
]

print("【原始消息列表】")
for msg in messages:
    print(f"  {msg.__class__.__name__}: {msg.content[:40]}...")

# ── 2. 序列化为字典列表 ────────────────────────────
dicts = messages_to_dict(messages)
print(f"\n【序列化为 dict】")
print(json.dumps(dicts, indent=2, ensure_ascii=False))

# ── 3. 模拟：存入 JSON 文件（持久化）───────────────
with open("/tmp/chat_history.json", "w", encoding="utf-8") as f:
    json.dump(dicts, f, ensure_ascii=False, indent=2)
print("\n💾 已持久化到 /tmp/chat_history.json")

# ── 4. 从文件读取并反序列化 ────────────────────────
with open("/tmp/chat_history.json", "r", encoding="utf-8") as f:
    loaded_dicts = json.load(f)

restored_messages = messages_from_dict(loaded_dicts)

print(f"\n【反序列化后还原】")
for msg in restored_messages:
    print(f"  {msg.__class__.__name__}: {msg.content[:40]}...")
    # 验证 tool_calls 也被正确还原
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"    → tool_calls: {msg.tool_calls}")

# ── 5. 验证类型一致性 ──────────────────────────────
print(f"\n✅ 类型一致性检查:")
for orig, restored in zip(messages, restored_messages):
    match = type(orig) == type(restored)
    print(
        f"   {orig.__class__.__name__:15} → {restored.__class__.__name__:15} {'✅' if match else '❌'}"
    )
