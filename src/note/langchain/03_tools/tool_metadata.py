"""
显式控制工具元数据与 Pydantic 参数模式
运行: uv run 03_tools/tool_metadata.py
所需依赖:
# uv add langchain pydantic
"""

from pydantic import BaseModel, Field
from langchain.tools import tool

# ── 用 Pydantic 定义精确的参数模式 ─────────────────
class WeatherInput(BaseModel):
    """天气查询的输入参数"""

    city: str = Field(description="城市中文名或英文名，如 '北京' 或 'Tokyo'")
    unit: str = Field(
        default="celsius", description="温度单位: 'celsius' 或 'fahrenheit'"
    )


class CalculatorInput(BaseModel):
    """计算器输入参数"""

    expression: str = Field(description="合法的数学表达式，如 'sqrt(16) + 5'")
    precision: int = Field(default=2, ge=0, le=10, description="结果保留小数位数，0-10")


# ── 显式覆盖 name 和 description ──────────────────
@tool("weather_lookup", args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    查询全球城市的实时天气，支持摄氏/华氏切换。
    当用户询问天气、温度、穿衣建议时调用此工具。
    """
    temp = 25 if unit == "celsius" else 77
    unit_label = "°C" if unit == "celsius" else "°F"
    return f"{city} 当前气温: {temp}{unit_label}，晴朗。"


@tool("safe_calculator", args_schema=CalculatorInput)
def calculate(expression: str, precision: int = 2) -> str:
    """
    高精度数学计算器。支持复杂表达式和精度控制。
    当用户需要数学运算、统计分析、单位换算时调用。
    """
    try:
        import math

        allowed = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "pow": math.pow,
            "pi": math.pi,
        }
        result = eval(expression, allowed)
        return f"结果: {round(result, precision)}"
    except Exception as e:
        return f"计算失败: {e}"


# ── 查看精确控制的元数据 ───────────────────────────
print("=" * 50)
print("📐 Pydantic 参数模式详情")
print("=" * 50)

for t in [get_weather, calculate]:
    print(f"\n🔧 {t.name}")
    print(f"   描述: {t.description}")
    print(f"   参数 JSON Schema:")
    import json

    print(json.dumps(t.args, indent=4, ensure_ascii=False))

# ── 调用验证 ──────────────────────────────────────
print(f"\n{'=' * 50}")
print("🧪 调用验证")
print("=" * 50)

print(get_weather.invoke({"city": "东京", "unit": "fahrenheit"}))
print(calculate.invoke({"expression": "sqrt(256) + pi", "precision": 4}))
