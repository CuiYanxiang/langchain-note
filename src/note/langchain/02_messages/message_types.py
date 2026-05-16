# ============================================================
# 消息类型
# ============================================================
# 安装命令：
#   uv add langchain langchain-openai python-dotenv
# 运行命令：
#   uv run python 02_messages/message_types.py
# ============================================================

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, ChatMessage

# 演示所有消息类型
print("=" * 70)
print("💬 消息类型详解")
print("=" * 70)
# 1. SystemMessage — 定义模型角色和行为规则
system_msg = SystemMessage(
    content="你是一个专业的 Python 导师，回答简洁，每次不超过 3 句话。"
)
print("=== SystemMessage ===")
print(f"  type:    {system_msg.type}")  # "system"
print(f"  content: {system_msg.content}")
print()

# 2. HumanMessage — 用户输入
human_msg = HumanMessage(
    content="什么是装饰器？"
)
print("=== HumanMessage ===")
print(f"  type:    {human_msg.type}")  # "human"
print(f"  content: {human_msg.content}")
print(f"  - 示例: {human_msg.model_dump()}")
print()

# 3. AIMessage — 模型回复（由模型生成，也可手动构造）
ai_msg = AIMessage(
    content="装饰器是一个接受函数作为参数并返回新函数的高阶函数，"
            "用于在不修改原函数代码的情况下扩展其功能。"
)
print("=== AIMessage ===")
print(f"  type:    {ai_msg.type}")  # "ai"
print(f"  content: {ai_msg.content[:50]}...")
print(f"  token 用量: {ai_msg.usage_metadata}...")
print()

# 4. AIMessage — 携带 tool_calls（模型决定调用工具时）
ai_with_tools = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call_abc123",
        }
    ],
)
print(f"=== AIMessage (带工具调用)===")
print(f"  content    = {ai_with_tools.content!r}")
print(f"  tool_calls = {ai_with_tools.tool_calls}")
print(f"  有工具调用   = {ai_with_tools.tool_calls is not None}")
print()

# 5. ToolMessage — 工具执行结果（第 3 阶段详细讲解）
tool_msg = ToolMessage(
    content='{"temperature": 22, "unit": "celsius", "city": "Beijing"}',
    tool_call_id="call_abc123",  # 对应 AIMessage 中的 tool_call id
    name="get_weather",  # 工具名称
)
print("=== ToolMessage ===")
print(f"  type:         {tool_msg.type}")  # "tool"
print(f"  name:         {tool_msg.name}")
print(f"  tool_call_id: {tool_msg.tool_call_id}")
print(f"  content:      {tool_msg.content}")
print()

# 6. ChatMessage - 通用聊天消息（指定角色）
chat_msg = ChatMessage(content="这是自定义角色的消息", role="expert")
print(f"=== ChatMessage (自定义角色) ===")
print(f"   - content: {chat_msg.content}")
print(f"   - role: {chat_msg.role}")

# ----------------------------------------------------------
# 6. 多模态消息 — HumanMessage 支持图片等非文本内容
# ----------------------------------------------------------
multimodal_msg = HumanMessage(
    content=[
        {"type": "text", "text": "这张图片里有什么？"},
        # 支持 base64 图片或 URL
        # {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]
)
print("=== Multimodal HumanMessage ===")
print(f"  content 类型: {type(multimodal_msg.content)}")  # list
print(f"  content:      {multimodal_msg.content}")
print()

# ----------------------------------------------------------
# 7. 所有消息类型的 type 属性一览
# ----------------------------------------------------------
print("=== 消息类型速查 ===")
print(f"  SystemMessage.type  → '{SystemMessage(content='').type}'")
print(f"  HumanMessage.type   → '{HumanMessage(content='').type}'")
print(f"  AIMessage.type      → '{AIMessage(content='').type}'")
print(f"  ToolMessage.type    → '{ToolMessage(content='', tool_call_id='x').type}'")

print("\n" + "=" * 70)
print("✅ 消息类型 完成！")
print("=" * 70)
print("\n💡 关键知识点:")
print("   1. HumanMessage - 用户输入")
print("   2. AIMessage - 模型回复（含 usage_metadata）")
print("   3. SystemMessage - 系统指令")
print("   4. ToolMessage - 工具调用结果（推荐使用）")
