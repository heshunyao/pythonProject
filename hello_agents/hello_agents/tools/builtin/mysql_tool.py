# mysql_tool.py
import os
from typing import Optional, Dict, Any
from hello_agents.hello_agents.tools.base import Tool, ToolParameter


class MySQLTool(Tool):
    """MySQL数据库查询工具"""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 database: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None):
        super().__init__(
            name="mysql_query",
            description="MySQL数据库查询工具，可以执行SQL查询并返回结果"
        )
        self.host = host or os.getenv("MYSQL_HOST", "localhost")
        self.port = port or int(os.getenv("MYSQL_PORT", "3306"))
        self.database = database or os.getenv("MYSQL_DATABASE")
        self.user = user or os.getenv("MYSQL_USER")
        self.password = password or os.getenv("MYSQL_PASSWORD")
        self.connection = None

    def _get_connection(self):
        """获取数据库连接"""
        if self.connection is None:
            try:
                # 尝试导入mysql.connector
                try:
                    import mysql.connector
                except ImportError:
                    raise ImportError(
                        "MySQL连接模块未安装。请运行以下命令安装:\n"
                        "pip install mysql-connector-python"
                    )

                self.connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
                print(f"✅ 成功连接到MySQL数据库: {self.database}")
            except ImportError as e:
                raise Exception(f"模块缺失: {str(e)}")
            except Exception as e:
                raise Exception(f"数据库连接失败: {str(e)}")
        return self.connection

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行SQL查询"""
        query = parameters.get("query", "").strip()

        if not query:
            return "❌ SQL查询语句不能为空"

        # 安全检查：只允许SELECT查询
        if not query.upper().startswith("SELECT"):
            return "❌ 为了安全起见，只允许执行SELECT查询"

        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute(query)
            results = cursor.fetchall()

            cursor.close()

            if not results:
                return "✅ 查询执行成功，但没有返回任何结果"

            # 格式化结果
            output = f"✅ 查询成功，返回 {len(results)} 条记录:\n\n"
            for i, row in enumerate(results, 1):
                output += f"[{i}] {row}\n"

            return output

        except Exception as e:
            return f"❌ 查询执行失败: {str(e)}"

    def get_parameters(self):
        """获取参数定义"""
        return [
            ToolParameter(
                name="query",
                type="string",
                description="SQL查询语句，例如: SELECT * FROM users WHERE age > 18"
            )
        ]

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("✅ 数据库连接已关闭")


def create_mysql_tool(host: Optional[str] = None, port: Optional[int] = None,
                      database: Optional[str] = None, user: Optional[str] = None,
                      password: Optional[str] = None):
    """创建MySQL工具实例"""
    return MySQLTool(host, port, database, user, password)


# 使用示例
if __name__ == "__main__":
    # 方式1: 使用环境变量
    # mysql_tool = create_mysql_tool()

    # 方式2: 直接传入参数
    mysql_tool = create_mysql_tool(
        host="localhost",
        port=3306,
        database="test_db",
        user="root",
        password="password"
    )

    # 执行查询
    result = mysql_tool.run({"query": "SELECT * FROM users LIMIT 5"})
    print(result)

    # 关闭连接
    mysql_tool.close()