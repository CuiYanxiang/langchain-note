"""
LangChain AIMessage 对象深度解析
安装命令: uv add langchain langchain-openai python-dotenv
"""

import os
import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 加载环境变量
load_dotenv()


def get_model():
    model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.5,
        max_tokens=500,
    )

    return model


def demo_aimessage_structure():
    """AIMessage 对象结构解析"""
    print("=" * 70)
    print("📦 Demo 1: AIMessage 对象结构解析")
    print("=" * 70)

    model = get_model()
    messages = [HumanMessage(content="介绍一下 LangChain 的核心组件。")]

    response: AIMessage = model.invoke(messages)

    print("\n" + "─" * 70)
    print("1️⃣  .content - 模型生成的文本内容")
    print("─" * 70)
    print(
        response.content[:200] + "..."
        if len(response.content) > 200
        else response.content
    )

    print("\n" + "─" * 70)
    print("2️⃣  .response_metadata - 响应元数据")
    print("─" * 70)
    print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))

    print("\n" + "─" * 70)
    print("3️⃣  .usage_metadata - Token 消耗统计")
    print("─" * 70)
    print(json.dumps(response.usage_metadata, indent=2))

    print("\n" + "─" * 70)
    print("4️⃣  完整 AIMessage 对象（序列化）")
    print("─" * 70)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False)[:500] + "...")


def demo_message_types():
    """不同消息类型的对比"""
    print("\n" + "=" * 70)
    print("💬 Demo 2: 不同消息类型对比")
    print("=" * 70)

    # HumanMessage - 用户输入
    human_msg = HumanMessage(content="你好，我想了解 AI。")
    print(f"\n👤 HumanMessage:")
    print(f"   - content: {human_msg.content}")
    print(f"   - type: {human_msg.type}")

    # AIMessage - 模型回复
    model = get_model()
    ai_msg: AIMessage = model.invoke([human_msg])
    print(f"\n🤖 AIMessage:")
    print(f"   - content: {ai_msg.content[:50]}...")
    print(f"   - type: {ai_msg.type}")
    print(f"   - has tool_calls: {ai_msg.tool_calls is not None}")

    # ToolMessage - 工具调用结果（模拟）
    tool_msg = ToolMessage(
        content="查询成功：北京天气晴朗，25℃",
        tool_call_id="tool_123",
        name="weather_tool",
    )
    print(f"\n🔧 ToolMessage:")
    print(f"   - content: {tool_msg.content}")
    print(f"   - tool_call_id: {tool_msg.tool_call_id}")
    print(f"   - name: {tool_msg.name}")


def demo_message_serialization():
    """消息序列化与反序列化"""
    print("\n" + "=" * 70)
    print("🔄 Demo 3: 消息序列化与反序列化")
    print("=" * 70)

    model = get_model()
    messages = [
        HumanMessage(content="推荐 3 本关于机器学习的书。"),
    ]

    response: AIMessage = model.invoke(messages)

    # 序列化为 dict
    serialized = response.model_dump()
    print("\n📤 序列化为 dict:")
    print(json.dumps(serialized, indent=2, ensure_ascii=False)[:300] + "...")

    # 从 dict 重建 AIMessage
    reconstructed = AIMessage(**serialized)
    print("\n📥 从 dict 重建 AIMessage:")
    print(f"   - content 匹配: {response.content == reconstructed.content}")
    print(
        f"   - usage_metadata 匹配: {response.usage_metadata == reconstructed.usage_metadata}"
    )


def demo_token_analysis():
    """Token 消耗深度分析"""
    print("\n" + "=" * 70)
    print("💰 Demo 4: Token 消耗深度分析")
    print("=" * 70)

    model = get_model()

    # 测试不同长度的输入
    test_cases = [
        ("短文本", "你好"),
        ("中等文本", "请解释什么是深度学习，包括其基本概念和应用场景。"),
        (
            "长文本",
            "写一篇关于人工智能发展历史的文章，涵盖从 1950 年代到现在的关键里程碑、重要人物和未来趋势。",
        ),
    ]

    print("\n📊 Token 消耗对比:")
    print(f"{'─' * 70}")
    print(
        f"{'输入类型':<12} {'输入长度':<10} {'输入 Token':<12} {'输出 Token':<12} {'总计':<10}"
    )
    print(f"{'─' * 70}")

    for name, content in test_cases:
        messages = [HumanMessage(content=content)]
        response: AIMessage = model.invoke(messages)

        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
        total_tokens = response.usage_metadata.get("total_tokens", 0)

        print(
            f"{name:<12} {len(content):<10} {input_tokens:<12} {output_tokens:<12} {total_tokens:<10}"
        )

    print(f"{'─' * 70}")


if __name__ == "__main__":
    demo_aimessage_structure()
    demo_message_types()
    demo_message_serialization()
    demo_token_analysis()

    print("\n" + "=" * 70)
    print("\n💡 关键知识点:")
    print("   1. AIMessage.content - 获取模型回复文本")
    print("   2. AIMessage.response_metadata - 查看模型元数据")
    print("   3. AIMessage.usage_metadata - 统计 Token 消耗")
    print("   4. AIMessage.dict() - 序列化消息对象")
