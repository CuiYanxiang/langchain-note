"""
Static Agent —— 固定工具列表
运行: uv run python 04_agents/static_agent.py

所需依赖:
# uv add langchain langchain-openai python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# ── 定义固定工具集（企业级场景：财务分析工具箱）───
@tool
def query_revenue(year: int) -> str:
    """查询指定年份的公司营收数据（单位：亿元）"""
    data = {2023: 150.5, 2024: 210.8, 2025: 285.3}
    return f"{year} 年营收: {data.get(year, '暂无数据')} 亿元"


@tool
def query_profit(year: int) -> str:
    """查询指定年份的公司净利润数据（单位：亿元）"""
    data = {2023: 22.3, 2024: 35.6, 2025: 48.1}
    return f"{year} 年净利润: {data.get(year, '暂无数据')} 亿元"


@tool
def calculate_growth(current: float, previous: float) -> str:
    """计算同比增长率"""
    if previous == 0:
        return "上年数据为 0，无法计算增长率"
    rate = (current - previous) / previous * 100
    return f"同比增长率: {rate:.2f}%"


# ── 构建 Static Agent ─────────────────────────────
model = init_chat_model(
        model=os.getenv("OPENAI_MODEL"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

# 工具列表在创建时固定，运行时不改变
STATIC_TOOLS = [query_revenue, query_profit, calculate_growth]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是公司财务分析师，只能使用提供的财务工具回答问题。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(model, STATIC_TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=STATIC_TOOLS, verbose=True)

# ── 运行 ──────────────────────────────────────────
print("=" * 50)
print("📊 Static Agent: 财务分析")
print("=" * 50)

result = agent_executor.invoke(
    {
        "input": "公司 2024 年和 2025 年的营收和净利润分别是多少？2025 年营收相比 2024 年增长了多少？"
    }
)

print(f"\n📋 最终报告:\n{result['output']}")
