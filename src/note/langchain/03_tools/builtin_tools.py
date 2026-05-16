# ============================================================
# StructuredTool 与内置工具（DuckDuckGo / Wikipedia / Shell）
# ============================================================
# 安装依赖:
#   uv add langchain langchain-community
# 运行:
#   uv run 03_tools/builtin_tools.py
# ============================================================
from __future__ import annotations

import json
from typing import List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def demo_structured_tool():
    """StructuredTool 创建复杂工具"""
    print("=" * 70)
    print("🔧 Demo 1: StructuredTool 复杂工具")
    print("=" * 70)

    # 定义 Pydantic 模型
    class TravelPlanInput(BaseModel):
        """旅行计划输入"""

        destination: str = Field(description="目的地城市")
        duration_days: int = Field(description="旅行天数", ge=1, le=30)
        budget: float = Field(description="预算（人民币）", ge=0)
        interests: List[str] = Field(
            description="兴趣爱好列表", default=["美食", "景点", "购物"]
        )
        has_children: bool = Field(description="是否有儿童", default=False)

    def create_travel_plan(
            destination: str,
            duration_days: int,
            budget: float,
            interests: List[str],
            has_children: bool,
    ) -> str:
        """生成个性化旅行计划"""
        # 计算每日预算
        daily_budget = budget / duration_days

        # 根据兴趣生成建议
        interest_str = "、".join(interests)
        family_note = "（适合家庭）" if has_children else ""

        plan = f"""
        【{destination} 旅行计划】{family_note}
        
        📅 行程时长: {duration_days} 天
        💰 总预算: ¥{budget:,} (每日约 ¥{daily_budget:,.0f})
        🎯 兴趣偏好: {interest_str}
        
        📝 建议行程:
        """

        # 根据天数生成行程
        for day in range(1, duration_days + 1):
            plan += f"\n第 {day} 天:\n"
            plan += f"  • 上午: 参观 {destination} 主要景点\n"
            plan += f"  • 下午: 体验当地特色活动\n"
            plan += f"  • 晚上: 品尝美食，休息\n"

        plan += f"\n💡 预算分配建议:\n"
        plan += f"  • 住宿: ¥{budget * 0.4:,.0f} (40%)\n"
        plan += f"  • 餐饮: ¥{budget * 0.3:,.0f} (30%)\n"
        plan += f"  • 交通: ¥{budget * 0.2:,.0f} (20%)\n"
        plan += f"  • 门票/购物: ¥{budget * 0.1:,.0f} (10%)\n"

        return plan

    # 创建 StructuredTool
    travel_tool = StructuredTool.from_function(
        func=create_travel_plan,
        name="旅行计划生成器",
        description="根据目的地、天数、预算等生成个性化旅行计划",
        args_schema=TravelPlanInput,
        return_direct=False,  # False 表示结果需要经过 LLM 处理
    )

    print(f"\n✅ 工具创建成功!")
    print(f"   - 名称: {travel_tool.name}")
    print(f"   - 描述: {travel_tool.description}")
    print(f"   - Pydantic Schema:")
    print(f"     {json.dumps(travel_tool.args_schema.model_json_schema(), indent=2)}")

    # 调用工具
    print(f"\n🧪 测试调用:")
    result = travel_tool.invoke(
        {
            "destination": "成都",
            "duration_days": 3,
            "budget": 5000,
            "interests": ["美食", "熊猫", "文化"],
            "has_children": True,
        }
    )

    print(f"   - 生成的计划:\n{result[:500]}...")


def demo_builtin_search_tools():
    """内置搜索工具"""
    print("\n" + "=" * 70)
    print("🌐 Demo 2: 内置搜索工具")
    print("=" * 70)

    # DuckDuckGo 搜索
    print(f"\n🔍 DuckDuckGo 搜索工具:")
    search = DuckDuckGoSearchRun()
    print(f"   - 工具名称: {search.name}")
    print(f"   - 描述: {search.description}")
    print(search.run("LangChain Python overview"))
    print()
    search = DuckDuckGoSearchRun(name="web_search", description="全网搜索实时信息")
    result = search.invoke("LangChain 1.2 新特性")
    print(result[:200] + "...")
    print()

    # 测试搜索
    query = "2024年奥运会举办城市"
    print(f"\n   📝 搜索: '{query}'")
    result = search.invoke(query)
    print(f"   📄 结果:\n{result[:300]}...")

    # Wikipedia 搜索
    print(f"\n📚 Wikipedia 搜索工具:")
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    )
    print(f"   - 工具名称: {wiki.name}")
    print(f"   - 描述: {wiki.description}")
    print(wiki.run("LangChain"))
    print()
    # 测试 Wikipedia
    query = "人工智能"
    print(f"\n   📝 查询: '{query}'")
    result = wiki.invoke(query)
    print(f"   📄 摘要:\n{result[:300]}...")


def demo_toolkit_pattern():
    """工具包模式"""
    print("\n" + "=" * 70)
    print("📦 Demo 3: 工具包模式（Toolkit）")
    print("=" * 70)

    # 创建多个相关工具
    @StructuredTool.from_function
    def create_file(filename: str, content: str) -> str:
        """创建新文件"""
        # 模拟文件创建
        return f"✅ 文件 '{filename}' 创建成功，内容 {len(content)} 字符"

    @StructuredTool.from_function
    def read_file(filename: str) -> str:
        """读取文件内容"""
        # 模拟文件读取
        return f"📄 文件 '{filename}' 内容: 这是模拟内容..."

    @StructuredTool.from_function
    def list_files(directory: str = ".") -> str:
        """列出目录文件"""
        # 模拟文件列表
        files = ["document.txt", "data.csv", "image.png", "script.py"]
        return f"📁 目录 '{directory}' 包含 {len(files)} 个文件:\n" + "\n".join(
            f"  • {f}" for f in files
        )

    @StructuredTool.from_function
    def delete_file(filename: str) -> str:
        """删除文件"""
        # 模拟文件删除
        return f"🗑️  文件 '{filename}' 已删除"

    # 组织成工具包
    file_tools = [create_file, read_file, list_files, delete_file]

    print(f"\n🔧 文件操作工具包:")
    for i, tool in enumerate(file_tools, 1):
        print(f"   {i}. {tool.name}: {tool.description}")

    print(f"\n🧪 工具包使用示例:")
    # 模拟 Agent 选择工具的逻辑
    user_requests = [
        ("创建文件", {"filename": "notes.txt", "content": "这是我的笔记"}),
        ("列出文件", {"directory": "."}),
        ("读取文件", {"filename": "notes.txt"}),
    ]

    for request_name, kwargs in user_requests:
        print(f"\n   📝 请求: {request_name}")

        # 简单的工具选择逻辑（实际中由 LLM 决定）
        if "创建" in request_name:
            tool = create_file
        elif "读取" in request_name:
            tool = read_file
        elif "列出" in request_name:
            tool = list_files
        elif "删除" in request_name:
            tool = delete_file
        else:
            continue

        result = tool.invoke(kwargs)
        print(f"   🤖 执行: {tool.name}")
        print(f"   ✅ 结果: {result}")


def demo_complex_structured_tool():
    """复杂 StructuredTool 示例"""
    print("\n" + "=" * 70)
    print("📊 Demo 4: 复杂 StructuredTool（数据分析）")
    print("=" * 70)

    class DataAnalysisInput(BaseModel):
        """数据分析输入"""

        data_type: str = Field(
            description="数据类型", json_schema_extra={"enum": ["sales", "user", "traffic", "revenue"]}
        )
        time_range: str = Field(
            description="时间范围", json_schema_extra={"enum": ["daily", "weekly", "monthly", "yearly"]}
        )
        metrics: List[str] = Field(
            description="分析指标", default=["total", "average", "max", "min"]
        )
        filters: Optional[dict] = Field(description="过滤条件", default=None)

    def analyze_data(
            data_type: str,
            time_range: str,
            metrics: List[str],
            filters: Optional[dict] = None,
    ) -> dict:
        """模拟数据分析"""
        # 模拟分析结果
        result = {
            "data_type": data_type,
            "time_range": time_range,
            "analysis_date": "2024-01-15",
            "metrics": {},
        }

        # 生成模拟指标
        base_value = {
            "sales": 100000,
            "user": 5000,
            "traffic": 50000,
            "revenue": 80000,
        }.get(data_type, 0)

        if "total" in metrics:
            result["metrics"]["total"] = base_value
        if "average" in metrics:
            result["metrics"]["average"] = round(base_value / 30, 2)
        if "max" in metrics:
            result["metrics"]["max"] = round(base_value * 1.2, 2)
        if "min" in metrics:
            result["metrics"]["min"] = round(base_value * 0.8, 2)

        if filters:
            result["filters_applied"] = filters

        return result

    # 创建工具
    analysis_tool = StructuredTool.from_function(
        func=analyze_data,
        name="数据分析工具",
        description="对销售、用户、流量等数据进行多维度分析",
        args_schema=DataAnalysisInput,
    )

    print(f"\n✅ 工具创建:")
    print(f"   - 支持的数据类型: sales, user, traffic, revenue")
    print(f"   - 支持的时间范围: daily, weekly, monthly, yearly")
    print(f"   - 支持的指标: total, average, max, min")

    # 测试调用
    print(f"\n🧪 测试调用:")
    result = analysis_tool.invoke(
        {
            "data_type": "sales",
            "time_range": "monthly",
            "metrics": ["total", "average", "max"],
            "filters": {"region": "china", "channel": "online"},
        }
    )

    import json

    print(f"   📊 分析结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_structured_tool()
    demo_builtin_search_tools()
    demo_toolkit_pattern()
    demo_complex_structured_tool()

    print("\n" + "=" * 70)
    print("✅ tools 完成！")
    print("=" * 70)
    print("\n💡 关键知识点:")
    print("   1. StructuredTool - 复杂参数验证")
    print("   2. Pydantic BaseModel - 参数 Schema 定义")
    print("   3. 内置工具: DuckDuckGoSearchRun / WikipediaQueryRun")
    print("   4. 工具包模式 - 组织相关工具")
    print("   5. from_function - 从函数创建 StructuredTool")
