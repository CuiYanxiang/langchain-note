# ============================================================
#  工具的 name / description / args_schema（Pydantic）
# ============================================================
# 安装依赖:
#   uv add langchain python-dotenv pydantic
#
# 运行:
#   uv run 03_tools/tool_schema.py
# ============================================================
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain_core.tools import tool, StructuredTool


# ----------------------------------------------------------
# 1. 通过 @tool 参数自定义 name 和 description
# ----------------------------------------------------------
@tool(
    name_or_callable="weather_query",
    description="查询指定城市的当前天气信息，返回温度、湿度和天气状况。",
)
def get_weather(city: str, unit: str = "celsius") -> str:
    """这个 docstring 会被 @tool 的 description 覆盖"""
    # Mock 数据
    mock_data = {
        "北京": {"temp": 22, "humidity": 45, "condition": "晴"},
        "上海": {"temp": 26, "humidity": 78, "condition": "多云"},
        "深圳": {"temp": 30, "humidity": 85, "condition": "阵雨"},
    }
    data = mock_data.get(city, {"temp": 20, "humidity": 50, "condition": "未知"})
    return f"{city}: {data['temp']}°{'C' if unit == 'celsius' else 'F'}, 湿度 {data['humidity']}%, {data['condition']}"


print("=== 1. 自定义 name 和 description ===")
print(f"  name:        {get_weather.name}")
print(f"  description: {get_weather.description}")
print(f"  args:        {get_weather.args}")
print(f"  结果:        {get_weather.invoke({'city': '北京'})}\n")


# ----------------------------------------------------------
# 2. 用 Pydantic BaseModel 定义精确的参数 Schema
# ----------------------------------------------------------
class SearchInput(BaseModel):
    """搜索工具的输入参数"""

    query: str = Field(
        description="搜索关键词，不超过 100 个字符",
        max_length=100,
    )
    max_results: int = Field(
        default=5,
        description="最大返回结果数，范围 1-20",
        ge=1,
        le=20,
    )
    language: str = Field(
        default="zh",
        description="搜索语言，zh=中文，en=英文",
        pattern="^(zh|en)$",
    )


@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5, language: str = "zh") -> str:
    """在知识库中搜索相关信息"""
    return f"搜索 '{query}'（语言={language}，最多{max_results}条）: 找到 {max_results} 条结果"


print("=== 2. Pydantic args_schema ===")
print(f"  name:        {search.name}")
print(f"  args_schema: {search.args_schema.model_json_schema()}")
print(
    f"  结果:        {search.invoke({'query': 'LangChain 教程', 'max_results': 3})}\n"
)


# ----------------------------------------------------------
# 3. StructuredTool —— 更显式的工具创建方式
# ----------------------------------------------------------
class FileOperationInput(BaseModel):
    """文件操作的输入参数"""

    file_path: str = Field(description="文件的绝对路径")
    operation: Literal["read", "write", "delete"] = Field(description="操作类型")
    content: str = Field(
        default="",
        description="写入的内容（仅 operation=write 时需要）",
    )


def file_operation(file_path: str, operation: str, content: str = "") -> str:
    """执行文件操作（读取、写入或删除）"""
    # Mock 实现
    return f"执行 {operation} 操作: {file_path}" + (
        f"，写入 {len(content)} 字符" if operation == "write" else ""
    )


file_tool = StructuredTool.from_function(
    func=file_operation,
    name="file_operation",
    description="对文件执行读取、写入或删除操作。删除操作需要额外确认。",
    args_schema=FileOperationInput,
    return_direct=False,  # True = 工具结果直接返回给用户；False = 返回给 Agent 继续处理
)

print("=== 3. StructuredTool ===")
print(f"  name:        {file_tool.name}")
print(f"  description: {file_tool.description}")
print(f"  args_schema: {file_tool.args_schema.model_json_schema()}")
print(
    f"  结果:        {file_tool.invoke({'file_path': '/tmp/test.txt', 'operation': 'read'})}\n"
)

# ----------------------------------------------------------
# 4. 对比：无 Schema vs 有 Schema 的差异
# ----------------------------------------------------------
print("=== 4. Schema 对比 ===\n")


# 无 Schema —— 参数描述来自 docstring（较模糊）
@tool
def vague_tool(query: str) -> str:
    """搜索东西"""
    return f"搜索: {query}"


# 有 Schema —— 参数有精确约束
class PreciseSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="搜索关键词")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="最低相关度分数")


@tool(args_schema=PreciseSearchInput)
def precise_search(query: str, top_k: int = 10, min_score: float = 0.5) -> str:
    """在向量数据库中进行语义搜索"""
    return f"语义搜索 '{query}': top_k={top_k}, min_score={min_score}"


print("  无 Schema:")
print(f"    args: {vague_tool.args}")
print()
print("  有 Schema:")
print(f"    args: {precise_search.args}")
