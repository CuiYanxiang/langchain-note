# ============================================================
# MessagePlaceholder 与动态消息插入
# ============================================================
# 安装依赖:
#   uv add langchain langchain-openai" python-dotenv
# 运行:
#   uv run 02_messages/message_placeholder.py
# ============================================================

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL")
)


def demo_basic_placeholder():
    """基础 MessagePlaceholder 使用"""
    print("=" * 70)
    print("📌 Demo 1: 基础 MessagePlaceholder 使用")
    print("=" * 70)

    # 创建包含占位符的 Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 动态插入对话历史
        ("human", "{input}")  # 用户当前输入
    ])

    print(f"\n📝 Prompt Template 结构:")
    print(f"   1. SystemMessage: 固定系统指令")
    print(f"   2. MessagesPlaceholder: 动态对话历史")
    print(f"   3. HumanMessage: 当前用户输入")

    # 构建对话历史
    chat_history = [
        HumanMessage(content="你好，我叫小明"),
        AIMessage(content="你好小明！很高兴认识你。有什么我可以帮助的吗？"),
        HumanMessage(content="我想学习 Python"),
        AIMessage(content="太好了！Python 是一门很棒的编程语言。你想从哪里开始学起？")
    ]

    # 格式化 Prompt
    formatted = prompt.format_messages(
        chat_history=chat_history,
        input="推荐一些学习资源"
    )

    print(f"\n📊 格式化后的完整 Prompt:")
    for i, msg in enumerate(formatted):
        role = "👤" if isinstance(msg, HumanMessage) else "🤖" if isinstance(msg, AIMessage) else "⚙️"
        content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
        print(f"  {i}. {role} {type(msg).__name__}: {content_preview}...")


def demo_few_shot_with_placeholder():
    """Few-shot 学习示例"""
    print("\n" + "=" * 70)
    print("🎯 Demo 2: Few-shot 学习示例")
    print("=" * 70)

    # Few-shot 示例
    few_shot_examples = [
        HumanMessage(content="法国的首都是哪里？"),
        AIMessage(content="{\"country\": \"法国\", \"capital\": \"巴黎\"}"),
        HumanMessage(content="日本的首都是哪里？"),
        AIMessage(content="{\"country\": \"日本\", \"capital\": \"东京\"}"),
        HumanMessage(content="巴西的首都是哪里？"),
        AIMessage(content="{\"country\": \"巴西\", \"capital\": \"巴西利亚\"}")
    ]

    # 创建 Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个国家首都查询助手。请用 JSON 格式回答。"),
        MessagesPlaceholder(variable_name="examples"),  # Few-shot 示例
        ("human", "{question}")
    ])

    print(f"\n📚 Few-shot 示例:")
    for i in range(0, len(few_shot_examples), 2):
        print(f"  Q: {few_shot_examples[i].content}")
        print(f"  A: {few_shot_examples[i + 1].content}")

    # 格式化新问题
    formatted = prompt.format_messages(
        examples=few_shot_examples,
        question="加拿大的首都是哪里？"
    )

    print(f"\n📝 格式化后的新问题 Prompt:")
    for msg in formatted:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖" if isinstance(msg, AIMessage) else "⚙️"
        print(f"  {role} {msg.content[:60]}...")

    # 调用模型
    response = model.invoke(formatted)
    print(f"\n🤖 模型回复: {response.content}")


def demo_conditional_history():
    """条件性对话历史插入"""
    print("\n" + "=" * 70)
    print("🔀 Demo 3: 条件性对话历史插入")
    print("=" * 70)

    # 根据对话长度决定是否插入历史
    def get_prompt_with_conditional_history(user_input, chat_history, max_history_turns=3):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # 条件：只在历史超过 2 轮时插入
        if len(chat_history) > 2:
            # 只取最近的 N 轮
            history_to_use = chat_history[-(max_history_turns * 2):]
            print(f"   ✅ 插入最近 {len(history_to_use)} 条历史")
        else:
            history_to_use = []
            print(f"   ⚠️  历史太短，不插入上下文")

        return prompt.format_messages(
            history=history_to_use,
            input=user_input
        )

    # 短历史
    print(f"\n📋 场景 1: 短对话历史")
    short_history = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么我可以帮助的吗？")
    ]
    prompt_1 = get_prompt_with_conditional_history("今天天气怎么样？", short_history)
    print(f"   Prompt 消息数: {len(prompt_1)}")

    # 长历史
    print(f"\n📋 场景 2: 长对话历史")
    long_history = [
        HumanMessage(content="我想订机票"),
        AIMessage(content="好的，请问您的出发地和目的地是哪里？"),
        HumanMessage(content="从北京到上海"),
        AIMessage(content="明白了。请问您的出行日期？"),
        HumanMessage(content="下周五"),
        AIMessage(content="下周五是 12 月 20 日。请问需要经济舱还是商务舱？"),
        HumanMessage(content="经济舱"),
        AIMessage(content="好的。请问有其他特殊需求吗？")
    ]
    prompt_2 = get_prompt_with_conditional_history("帮我查询价格", long_history)
    print(f"   Prompt 消息数: {len(prompt_2)}")
    print(f"   包含的历史:")
    for msg in prompt_2[1:-1]:  # 排除 system 和当前输入
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"     {role} {msg.content[:40]}...")


def demo_multi_placeholder():
    """多个 MessagePlaceholder"""
    print("\n" + "=" * 70)
    print("📋 Demo 4: 多个 MessagePlaceholder")
    print("=" * 70)

    # 创建包含多个占位符的复杂 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "角色: {role}\n任务: {task}"),
        MessagesPlaceholder(variable_name="background_info"),  # 背景信息
        ("system", "之前的讨论:"),
        MessagesPlaceholder(variable_name="discussion_history"),  # 讨论历史
        ("human", "当前问题: {current_question}")
    ])

    print(f"\n📝 复杂 Prompt Template:")
    print(f"   - 系统角色和任务（变量）")
    print(f"   - 背景信息（MessagePlaceholder）")
    print(f"   - 讨论历史（MessagePlaceholder）")
    print(f"   - 当前问题（变量）")

    # 准备数据
    background = [
        HumanMessage(content="项目背景: 我们正在开发一个 AI 客服系统"),
        AIMessage(content="了解。这是一个很有前景的项目。")
    ]

    discussion = [
        HumanMessage(content="我们需要支持多轮对话"),
        AIMessage(content="建议使用 LangChain 的 ChatMessageHistory"),
        HumanMessage(content="还需要支持工具调用"),
        AIMessage(content="可以集成 LangChain 的 Tools 框架")
    ]

    # 格式化
    formatted = prompt.format_messages(
        role="AI 架构师",
        task="提供技术建议",
        background_info=background,
        discussion_history=discussion,
        current_question="如何实现会话持久化？"
    )

    print(f"\n📊 格式化后的完整 Prompt:")
    for i, msg in enumerate(formatted):
        role = "👤" if isinstance(msg, HumanMessage) else "🤖" if isinstance(msg, AIMessage) else "⚙️"
        content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
        print(f"  {i}. {role} {type(msg).__name__}: {content_preview}...")


def demo_dynamic_persona():
    """动态角色切换"""
    print("\n" + "=" * 70)
    print("🎭 Demo 5: 动态角色切换")
    print("=" * 70)

    # 不同角色的对话历史
    teacher_history = [
        HumanMessage(content="什么是机器学习？"),
        AIMessage(content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律...")
    ]

    comedian_history = [
        HumanMessage(content="讲个笑话"),
        AIMessage(content="为什么程序员分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25！")
    ]

    # 动态选择角色
    def get_role_specific_prompt(role, chat_history, user_input):
        role_prompts = {
            "teacher": "你是一个严谨的大学教授，用专业术语回答。",
            "comedian": "你是一个幽默的喜剧演员，用轻松搞笑的方式回答。",
            "friend": "你是一个亲密的朋友，用随意的口吻聊天。"
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", role_prompts.get(role, "你是一个有帮助的助手。")),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        return prompt.format_messages(
            history=chat_history,
            input=user_input
        )

    print(f"\n👨‍🏫 角色 1: 教授")
    prompt_teacher = get_role_specific_prompt("teacher", teacher_history, "解释一下神经网络")
    response_teacher = model.invoke(prompt_teacher)
    print(f"   🤖 {response_teacher.content[:80]}...")

    print(f"\n🎭 角色 2: 喜剧演员")
    prompt_comedian = get_role_specific_prompt("comedian", comedian_history, "为什么天空是蓝色的？")
    response_comedian = model.invoke(prompt_comedian)
    print(f"   🤖 {response_comedian.content[:80]}...")

    print(f"\n👫 角色 3: 朋友")
    prompt_friend = get_role_specific_prompt("friend", [], "今天过得怎么样？")
    response_friend = model.invoke(prompt_friend)
    print(f"   🤖 {response_friend.content[:80]}...")


if __name__ == "__main__":
    demo_basic_placeholder()
    demo_few_shot_with_placeholder()
    demo_conditional_history()
    demo_multi_placeholder()
    demo_dynamic_persona()

    print("\n" + "=" * 70)
    print("✅ 第 2 阶段 Demo 3 完成！")
    print("=" * 70)
    print("\n💡 关键知识点:")
    print("   1. MessagesPlaceholder - 动态插入消息列表")
    print("   2. ChatPromptTemplate - 构建灵活的提示模板")
    print("   3. Few-shot 学习 - 通过示例引导模型")
    print("   4. 条件性历史插入 - 根据上下文长度动态调整")
    print("   5. 多占位符组合 - 构建复杂提示结构")
