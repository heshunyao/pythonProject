from typing import List, Dict

class InterviewTools:
    """面试官工具定义类，封装所有面试相关的工具定义"""
    
    @staticmethod
    def get_tools() -> List[Dict]:
        """获取所有面试工具定义
        
        Returns:
            List[Dict]: 工具定义列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_question",
                    "description": "根据对话历史生成下一个面试问题",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_history": {
                                "type": "string",
                                "description": "之前的对话历史记录"
                            },
                            "position": {
                                "type": "string",
                                "description": "面试职位"
                            },
                            "level": {
                                "type": "string",
                                "description": "面试级别"
                            }
                        },
                        "required": ["chat_history", "position", "level"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_answer",
                    "description": "评估候选人的回答并给出评分",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "候选人的回答内容"
                            },
                            "question": {
                                "type": "string",
                                "description": "面试问题"
                            },
                            "reference_answer": {
                                "type": "string",
                                "description": "参考答案（可选）"
                            }
                        },
                        "required": ["answer", "question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "生成面试总结报告",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interview_data": {
                                "type": "object",
                                "description": "面试会话数据，包含问题、回答和评分"
                            }
                        },
                        "required": ["interview_data"]
                    }
                }
            }
        ]
    
    @staticmethod
    def get_tool_names() -> List[str]:
        """获取所有工具名称列表
        
        Returns:
            List[str]: 工具名称列表
        """
        return ["generate_question", "evaluate_answer", "generate_report"]
    
    @staticmethod
    def get_tool_descriptions() -> Dict[str, str]:
        """获取所有工具的描述信息
        
        Returns:
            Dict[str, str]: 工具名称到描述的映射
        """
        return {
            "generate_question": "根据对话历史生成下一个面试问题",
            "evaluate_answer": "评估候选人的回答并给出评分",
            "generate_report": "生成面试总结报告"
        }

def test_interview_tools():
    """测试面试工具定义"""
    print("\n=== 测试面试工具定义 ===")
    
    # 获取工具定义
    tools = InterviewTools.get_tools()
    print(f"\n工具数量: {len(tools)}")
    
    # 打印工具名称
    tool_names = InterviewTools.get_tool_names()
    print("\n工具名称列表:")
    for name in tool_names:
        print(f"- {name}")
    
    # 打印工具描述
    tool_descriptions = InterviewTools.get_tool_descriptions()
    print("\n工具描述:")
    for name, description in tool_descriptions.items():
        print(f"- {name}: {description}")

if __name__ == "__main__":
    test_interview_tools() 