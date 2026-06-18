def get_function_details(tools, function_name):
    """
    从工具词典中检索基于函数名称的函数描述。

    Args:
    tools (dict): A dictionary of tools, where each key is a function name mapping to its details.
    function_name (str): The name of the function for which the description is required.

    Returns:
    str: The description of the function if found, otherwise returns 'Function not found.'
    """
    return tools.get(function_name)

# Example usage:
tools = [{
    "query_by_product_name": {
        "type": "function",
        "function": {
            "name": "query_by_product_name",
            "description": "查询产品服务以检索与指定产品名称匹配或包含指定产品名称的产品列表。此功能可用于帮助客户通过在线平台或客户支持界面按名称查找产品。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "要搜索的产品的名称。搜索不区分大小写，允许部分匹配。"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    "read_store_promotions": {
        "type": "function",
        "function": {
            "name": "read_store_promotions",
            "description": "阅读商店的促销文件，查找与提供的产品名称相关的具体促销活动。此功能扫描文本文档，查找包含产品名称的任何促销条目。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "要在促销文档中搜索的产品名称。如果找到，该函数将返回促销详细信息。"
                    }
                },
                "required": ["product_name"]
            }
        }
    }
}]

# details = get_function_details(tools, "query_by_product_name")
# if details:
#     print(details)