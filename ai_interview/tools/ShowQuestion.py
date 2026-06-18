# 出题工具，根据questionbank 中的数据向量数据库存储的问题question，生成题目, 也可以自我生成相关的问题
from ai_interview.tools.QuestionBank import QuestionBank


class QuestionGenerator:
    def __init__(self, question_bank):
        self.question_bank = question_bank
        self.llm = question_bank.llm
        # 初始化不同环节的提示模板
        self.technical_interview_template = """
                作为面试专业的软件开发技术面试官，你需要考察候选人的技术能力。
                根据之前的对话记录：{chat_history}

                请提出一个与技术相关的问题，重点考察以下方面：
                1. 核心技术栈（如Java/JavaScript/Python/SQL/vue/html）
                   - 深入理解程度
                   - 实际项目经验
                   - 最佳实践应用
                2. 系统设计能力
                   - 架构设计思维
                   - 可扩展性考虑
                   - 性能优化意识
                3. 疑难问题解决
                   - 问题分析能力
                   - 解决方案设计
                   - 技术选型判断

                请生成一个具体、有深度的问题，并说明考察重点。
                问题："""

        self.behavioral_interview_template = """
                作为面试{position}的{level}级别综合面试官，请提出一个行为面试问题，考察：
                1. 团队合作经验
                   - 跨团队协作
                   - 知识分享
                   - 团队建设
                2. 冲突解决能力
                   - 技术分歧处理
                   - 团队矛盾化解
                   - 压力管理
                3. 项目领导经历
                   - 项目管理方法
                   - 团队激励
                   - 风险控制

                根据之前的回答：{chat_history}
                请生成一个具体的行为面试问题，并说明考察重点。
                问题："""

    def generate_question(self, chat_history: str = "", position: str = "软件开发工程师", level: str = "高级",
                          use_vector_db: bool = True) -> dict:
        """生成面试问题
        
        Args:
            chat_history: 之前的对话记录
            position: 面试职位
            level: 面试级别
            use_vector_db: 是否优先使用向量数据库中的问题
            
        Returns:
            dict: 包含问题和答案的字典
        """
        try:
            # 1. 如果启用向量数据库，先尝试从数据库获取相似问题
            if use_vector_db and chat_history:
                # 使用对话历史作为查询，判断问题类型
                question_type = self.question_bank.judge_question_type(chat_history)
                results = self.question_bank.search_questions(chat_history, question_type, k=1)

                if results and results[0]['similarity_score'] > 0.8:  # 相似度阈值
                    return {
                        'question': results[0]['question'],
                        'answer': results[0]['answer'],
                        'type': question_type,
                        'source': 'vector_db',
                        'similarity_score': results[0]['similarity_score']
                    }

            # 2. 如果向量数据库没有合适的问题，使用大模型生成新问题
            # 根据对话历史判断问题类型
            question_type = self.question_bank.judge_question_type(chat_history)

            # 选择对应的模板
            template = self.technical_interview_template if question_type == "tech" else self.behavioral_interview_template

            # 填充模板
            if question_type == "tech":
                prompt = template.format(chat_history=chat_history)
            else:
                prompt = template.format(
                    position=position,
                    level=level,
                    chat_history=chat_history
                )

            # 使用大模型生成问题
            response = self.llm.invoke(prompt)
            generated_question = response.content.strip()

            # 生成答案（这里可以后续扩展，使用大模型生成标准答案）
            answer = "这是一个开放性问题，需要根据候选人的回答进行具体评估。"

            return {
                'question': generated_question,
                'answer': answer,
                'type': question_type,
                'source': 'llm_generated'
            }

        except Exception as e:
            print(f"生成问题时出现错误: {str(e)}")
            return {
                'question': "生成问题时出现错误，请重试。",
                'answer': str(e),
                'type': 'error',
                'source': 'error'
            }


def test_question_generator():
    """测试问题生成器功能"""
    print("\n=== 开始测试问题生成器功能 ===")

    # 初始化问题库和生成器
    question_bank = QuestionBank()
    generator = QuestionGenerator(question_bank)

    # 测试用例
    test_cases = [
        {
            "chat_history": "我之前主要做Java后端开发，使用Spring Boot框架",
            "position": "Java开发工程师",
            "level": "高级",
            "use_vector_db": True
        },
        {
            "chat_history": "我在团队中经常需要协调不同部门的工作",
            "position": "技术主管",
            "level": "资深",
            "use_vector_db": True
        }
    ]

    for case in test_cases:
        print(f"\n测试用例: {case}")
        print("-" * 50)

        result = generator.generate_question(
            chat_history=case["chat_history"],
            position=case["position"],
            level=case["level"],
            use_vector_db=case["use_vector_db"]
        )

        print(f"生成的问题类型: {result['type']}")
        print(f"问题来源: {result['source']}")
        print(f"问题: {result['question']}")
        print(f"参考答案: {result['answer']}")
        if 'similarity_score' in result:
            print(f"相似度分数: {result['similarity_score']}")
        print("-" * 50)


if __name__ == "__main__":
    test_question_generator()
