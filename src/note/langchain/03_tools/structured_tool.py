# src/langchain_study/03_tools/structured_tool.py
"""
3.3 StructuredTool 创建复杂工具
运行: uv run python src/langchain_study/03_tools/structured_tool.py

所需依赖:
# uv add langchain "langchain[anthropic]" python-dotenv pydantic
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

load_dotenv()


# ── 模拟一个需要复杂初始化的服务类 ─────────────────
class DatabaseService:
    """模拟数据库查询服务"""
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.mock_data = {
            "users": [
                {"id": 1, "name": "张三", "role": "admin"},
                {"id": 2, "name": "李四", "role": "user"},
                {"id": 3, "name": "王五", "role": "user"},
            ]
        }

    def query_user(self, user_id: int) -> dict:
        for u in self.mock_data["users"]:
            if u["id"] == user_id:
                return u
        return {"error": "用户不存在"}


# ── Pydantic 参数模式 ─────────────────────────────
class QueryUserInput(BaseModel):
    user_id: int = Field(description="用户唯一标识符（正整数）")


# ── 初始化服务实例 ────────────────────────────────
db = DatabaseService(connection_string="sqlite:///app.db")

# ── 用 StructuredTool 包装类方法 ───────────────────
query_user_tool = StructuredTool.from_function(
    func=db.query_user,           # 绑定实例方法
    name="query_user_by_id",
    description="根据用户 ID 查询数据库中的用户信息。当用户询问某人资料时调用。",
    args_schema=QueryUserInput,
    return_direct=False,          # 结果返回给 LLM 继续处理（非直接返回给用户）
)

# ── 也可以包装普通函数，但支持更复杂的配置 ─────────
def send_email(to: str, subject: str, body: str) -> str:
    """模拟发送邮件"""
    return f"✅ 邮件已发送至 {to}，主题: {subject}"


class EmailInput(BaseModel):
    to: str = Field(description="收件人邮箱地址")
    subject: str = Field(description="邮件主题，简洁明了")
    body: str = Field(description="邮件正文内容")


send_email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="发送电子邮件给指定收件人。仅在用户明确要求发邮件时调用。",
    args_schema=EmailInput,
)

# ── 查看工具信息 ──────────────────────────────────
print("=" * 50)
print("🏗️  StructuredTool 注册信息")
print("=" * 50)

for t in [query_user_tool, send_email_tool]:
    print(f"\n📛 {t.name}")
    print(f"   描述: {t.description}")
    print(f"   参数: {t.args}")

# ── 调用验证 ──────────────────────────────────────
print(f"\n{'='*50}")
print("🚀 调用验证")
print("=" * 50)

print(query_user_tool.invoke({"user_id": 2}))
print(send_email_tool.invoke({
    "to": "admin@example.com",
    "subject": "系统通知",
    "body": "您的账户已激活。"
}))





# 03_tools/structured_tool.py
from typing import Annotated
from pydantic import BaseModel, StringConstraints, Field
from langchain_core.tools import StructuredTool
from langchain_core.utils import print_text

# 1. 定义 Pydantic 模型（用于参数验证）
class SearchParams(BaseModel):
    query: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    limit: Annotated[int, Field(ge=1, le=10)] = 5
    language: Annotated[str, StringConstraints(min_length=2, max_length=10)] = "zh"

# 2. 使用 StructuredTool 创建工具
search_tool = StructuredTool.from_function(
    func=lambda params: f"搜索结果（{params['query']}，限制 {params['limit']} 条，语言：{params['language']}）：...",
    name="web_search",
    description="执行网络搜索，支持限制结果数量和语言筛选。",
    args_schema=SearchParams,
    coroutine=None,  # 同步工具
)

# 3. 测试工具
if __name__ == "__main__":
    print("=== StructuredTool 测试 ===")
    result = search_tool.invoke({
        "query": "LangChain 1.x 新功能",
        "limit": 3,
        "language": "zh",
    })
    print_text(result, color="green")