"""
Static vs Dynamic 模型配置
运行: uv run python 01_models/static_vs_dynamic.py

所需依赖:
# uv add langchain langchain-openai python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableField

load_dotenv()

# ── Static 配置：参数写死，所有调用行为一致 ─────────
static_model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0,
    max_tokens=256
)
# ── Dynamic 配置：运行时通过 with_config 注入参数 ───
# 先创建一个基础模型，将 temperature 标记为可配置
dynamic_model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"),
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=512
).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="控制输出随机性，0=确定，1=创意",
    )
)

# ── 对比演示 ───────────────────────────────────────
prompt = "给'AI 编程助手'起一个创意产品名"

print("【Static 模式】temperature=0.0，结果固定：")
for i in range(2):
    resp = static_model.invoke(prompt)
    print(f"  第{i + 1}次: {resp.content.strip()}")

print("\n【Dynamic 模式】运行时切换 temperature：")
for temp in [0.0, 1.0]:
    resp = dynamic_model.with_config(configurable={"temperature": temp}).invoke(prompt)
    print(f"  temperature={temp}: {resp.content.strip()}")
