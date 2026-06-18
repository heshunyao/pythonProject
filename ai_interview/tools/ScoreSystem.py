# 评分工具， 对问题进行评分，由三个方面按一定算法进行综合评分，
# 1.用户对问题的回答内容，2.从向量数据库中找到问题对应的答案，3. 模型自我生成的答案
from ai_interview.llm.InterviewLLM import InterviewLLM
from ai_interview.tools.QuestionBank import QuestionBank
from typing import Dict, List, Tuple
import numpy as np


class ScoreSystem:
    def __init__(self, question_bank):
        self.llm = InterviewLLM().get_llm()
        self.embedding = InterviewLLM().get_embedding()
        self.question_bank = question_bank

        # 评分权重配置
        self.weights = {
            'content_score': 0.4,  # 回答内容评分权重
            'vector_score': 0.3,  # 向量数据库答案相似度评分权重
            'model_score': 0.3  # 模型生成答案评分权重
        }

        # 评分提示模板
        self.scoring_prompt_template = """
        作为专业的面试官，请对候选人的回答进行评分。评分标准如下：
        
        1. 技术准确性（40分）
           - 技术概念是否准确
           - 解决方案是否合理
           - 是否包含最佳实践
        
        2. 回答完整性（30分）
           - 是否覆盖问题的关键点
           - 是否有足够的细节支持
           - 是否有实际案例说明
        
        3. 表达清晰度（30分）
           - 逻辑是否清晰
           - 表达是否专业
           - 是否易于理解
        
        候选人回答：{answer}
        标准答案：{reference_answer}
        
        请给出一个0-100的总体评分，并简要说明评分理由。
        评分格式：
        分数：XX
        理由：XXX
        """

    def calculate_content_score(self, answer: str, reference_answer: str) -> Tuple[float, str]:
        """计算回答内容评分
        
        Args:
            answer: 候选人的回答
            reference_answer: 参考答案
            
        Returns:
            Tuple[float, str]: (分数, 评分理由)
        """
        try:
            # 构建评分提示
            prompt = self.scoring_prompt_template.format(
                answer=answer,
                reference_answer=reference_answer
            )

            # 使用LLM进行评分
            response = self.llm.invoke(prompt)
            result = response.content.strip()

            # 解析评分结果
            score_line = [line for line in result.split('\n') if '分数：' in line][0]
            score = float(score_line.split('：')[1].strip())

            # 提取评分理由
            reason_line = [line for line in result.split('\n') if '理由：' in line][0]
            reason = reason_line.split('：')[1].strip()

            return score, reason

        except Exception as e:
            print(f"计算内容评分时出现错误: {str(e)}")
            return 0.0, f"评分过程出现错误: {str(e)}"

    def calculate_vector_score(self, answer: str, question: str) -> Tuple[float, str]:
        """计算与向量数据库答案的相似度评分
        
        Args:
            answer: 候选人的回答
            question: 面试问题
            
        Returns:
            Tuple[float, str]: (相似度分数, 说明)
        """
        try:
            # 获取向量数据库中的相似问题
            question_type = self.question_bank.judge_question_type(question)
            results = self.question_bank.search_questions(question, question_type, k=1)

            if not results:
                return 0.0, "未找到相似问题"

            # 计算答案的向量表示
            answer_vector = self.embedding.embed_query(answer)
            reference_vector = self.embedding.embed_query(results[0]['answer'])

            # 计算余弦相似度
            similarity = np.dot(answer_vector, reference_vector) / (
                    np.linalg.norm(answer_vector) * np.linalg.norm(reference_vector)
            )

            # 将相似度转换为0-100的分数
            score = float(similarity * 100)

            return score, f"与标准答案的相似度为: {similarity:.2f}"

        except Exception as e:
            print(f"计算向量相似度评分时出现错误: {str(e)}")
            return 0.0, f"计算相似度时出现错误: {str(e)}"

    def calculate_model_score(self, answer: str, question: str) -> Tuple[float, str]:
        """使用模型生成评分
        
        Args:
            answer: 候选人的回答
            question: 面试问题
            
        Returns:
            Tuple[float, str]: (模型评分, 评分理由)
        """
        try:
            # 构建模型评分提示
            prompt = f"""
            作为面试官，请根据以下问题评估候选人的回答质量。
            
            问题：{question}
            候选人回答：{answer}
            
            请从以下方面进行评估：
            1. 回答的相关性（是否切中问题要点）
            2. 技术深度（是否展示了专业知识和经验）
            3. 表达质量（是否清晰、专业、有条理）
            
            请给出0-100的评分，并简要说明理由。
            格式：
            分数：XX
            理由：XXX
            """

            # 使用LLM进行评分
            response = self.llm.invoke(prompt)
            result = response.content.strip()

            # 解析评分结果
            score_line = [line for line in result.split('\n') if '分数：' in line][0]
            score = float(score_line.split('：')[1].strip())

            # 提取评分理由
            reason_line = [line for line in result.split('\n') if '理由：' in line][0]
            reason = reason_line.split('：')[1].strip()

            return score, reason

        except Exception as e:
            print(f"计算模型评分时出现错误: {str(e)}")
            return 0.0, f"模型评分过程出现错误: {str(e)}"

    def calculate_final_score(self, answer: str, question: str, reference_answer: str = None) -> Dict:
        """计算最终的综合评分
        
        Args:
            answer: 候选人的回答
            question: 面试问题
            reference_answer: 参考答案（可选）
            
        Returns:
            Dict: 包含各项评分和最终得分的字典
        """
        try:
            # 1. 计算内容评分
            content_score, content_reason = self.calculate_content_score(
                answer,
                reference_answer if reference_answer else ""
            )

            # 2. 计算向量相似度评分
            vector_score, vector_reason = self.calculate_vector_score(answer, question)

            # 3. 计算模型评分
            model_score, model_reason = self.calculate_model_score(answer, question)

            # 4. 计算加权平均分
            final_score = (
                    content_score * self.weights['content_score'] +
                    vector_score * self.weights['vector_score'] +
                    model_score * self.weights['model_score']
            )

            return {
                'final_score': round(final_score, 2),
                'details': {
                    'content_score': {
                        'score': round(content_score, 2),
                        'reason': content_reason,
                        'weight': self.weights['content_score']
                    },
                    'vector_score': {
                        'score': round(vector_score, 2),
                        'reason': vector_reason,
                        'weight': self.weights['vector_score']
                    },
                    'model_score': {
                        'score': round(model_score, 2),
                        'reason': model_reason,
                        'weight': self.weights['model_score']
                    }
                }
            }

        except Exception as e:
            print(f"计算最终评分时出现错误: {str(e)}")
            return {
                'final_score': 0.0,
                'error': str(e),
                'details': {
                    'content_score': {'score': 0.0, 'reason': '评分出错', 'weight': self.weights['content_score']},
                    'vector_score': {'score': 0.0, 'reason': '评分出错', 'weight': self.weights['vector_score']},
                    'model_score': {'score': 0.0, 'reason': '评分出错', 'weight': self.weights['model_score']}
                }
            }


def test_score_system():
    """测试评分系统功能"""
    print("\n=== 开始测试评分系统功能 ===")

    # 初始化问题库和评分系统
    question_bank = QuestionBank()
    score_system = ScoreSystem(question_bank)

    # 测试用例
    test_cases = [
        {
            "question": "请详细说明MyBatis的工作原理和主要特点",
            "answer": """MyBatis是一个优秀的持久层框架，它的工作原理主要包括以下几个方面：
1. 配置映射：通过XML或注解方式配置SQL映射
2. 动态SQL：支持动态生成SQL语句
3. 缓存机制：提供一级缓存和二级缓存
4. 延迟加载：支持关联对象的延迟加载
主要特点包括：
- 灵活性强，可以编写原生SQL
- 与Spring集成方便
- 支持动态SQL
- 提供缓存机制
- 支持延迟加载""",
            "reference_answer": "MyBatis是一个支持普通SQL查询、存储过程和高级映射的持久层框架。MyBatis避免了几乎所有的JDBC代码和手动设置参数以及获取结果集。MyBatis可以使用XML或注解来配置和映射原生信息，将POJO映射成数据库中的记录。"
        },
        {
            "question": "请描述一次你解决过的技术难题",
            "answer": """在我们项目中遇到过一个性能问题，系统在高并发下响应变慢。
我通过以下步骤解决了这个问题：
1. 使用性能分析工具定位到数据库查询是瓶颈
2. 优化了SQL语句，添加了合适的索引
3. 引入了缓存机制
4. 对热点数据进行了预加载
最终系统性能提升了300%""",
            "reference_answer": None
        }
    ]

    for case in test_cases:
        print(f"\n测试用例: {case['question']}")
        print("-" * 50)

        result = score_system.calculate_final_score(
            answer=case['answer'],
            question=case['question'],
            reference_answer=case['reference_answer']
        )

        print(f"最终得分: {result['final_score']}")
        print("\n详细评分:")
        for score_type, details in result['details'].items():
            print(f"\n{score_type}:")
            print(f"  分数: {details['score']}")
            print(f"  权重: {details['weight']}")
            print(f"  理由: {details['reason']}")
        print("-" * 50)


if __name__ == "__main__":
    test_score_system()
