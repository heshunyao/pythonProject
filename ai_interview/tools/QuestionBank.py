from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
import os
from typing import List, Dict
import json

from ai_interview.llm.InterviewLLM import InterviewLLM

TECH_QUESTIONS = "../data/java.json"
BEHAVIOR_QUESTIONS = "../data/behavior.json"
CHROMA_DB_DIR = "./chroma_db"


class QuestionBank:
    def __init__(self):
        """初始化问题库工具类"""
        self.embedding = InterviewLLM().get_embedding()
        self.llm = InterviewLLM().get_llm()
        self._tech_questions = None
        self._behavior_questions = None
        self.tech_db = None
        self.behavior_db = None

        # 确保数据库目录存在
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

        # 初始化向量数据库
        self._init_chroma()

        # 加载默认问题集
        self._load_default_questions()

    # 初始化Chroma数据库
    def _init_chroma(self):
        """初始化Chroma数据库"""
        try:
            # 尝试加载已存在的数据库
            self.tech_db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embedding,
                collection_name="tech_questions"
            )
            self.behavior_db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embedding,
                collection_name="behavior_questions"
            )
            print("成功加载现有向量数据库")
        except Exception as e:
            print(f"加载现有数据库失败: {str(e)}")
            print("将创建新的向量数据库")

    # 加载默认的问题集
    def _load_default_questions(self):
        """加载默认的问题集"""
        if os.path.exists(TECH_QUESTIONS):
            self.load_file_json(TECH_QUESTIONS, "tech")
        if os.path.exists(BEHAVIOR_QUESTIONS):
            self.load_file_json(BEHAVIOR_QUESTIONS, "behavior")

# 验证JSON文件格式是否正确
    def _validate_json_format(self, file_path: str) -> bool:
        """验证JSON文件格式是否正确"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict) or 'questions' not in data:
                    return False
                for q in data['questions']:
                    if not isinstance(q, dict) or 'question' not in q or 'answer' not in q:
                        return False
                return True
        except Exception as e:
            print(f"JSON文件格式验证失败: {str(e)}")
            return False

# 加载并处理 JSON 文件中的问题
    def load_file_json(self, questions_path: str, question_type: str = "tech") -> str:
        """加载并处理 JSON 文件中的问题
        
        Args:
            questions_path: JSON文件路径
            question_type: 问题类型 ("tech" 或 "behavior")
            
        Returns:
            str: 处理结果信息
        """
        if not os.path.exists(questions_path):
            error_msg = f"文件不存在: {questions_path}"
            print(error_msg)
            return error_msg

        if not self._validate_json_format(questions_path):
            error_msg = f"JSON文件格式不正确，请确保包含 questions 数组，每个问题包含 question 和 answer 字段"
            print(error_msg)
            return error_msg

        try:
            # 1. 加载JSON文档并提取问题和答案
            loader = JSONLoader(
                file_path=questions_path,
                jq_schema=".questions[].question",
                text_content=True
            )
            questions = loader.load()

            answer_loader = JSONLoader(
                file_path=questions_path,
                jq_schema=".questions[].answer",
                text_content=True
            )
            answers = answer_loader.load()

            if not questions or not answers:
                return "文件中没有找到问题或答案"

            # 2. 提取问题和答案
            texts = [doc.page_content for doc in questions]
            answer_texts = [doc.page_content for doc in answers]

            if len(texts) != len(answer_texts):
                return "问题和答案数量不匹配"

            print(f"从文件中加载了 {len(texts)} 个问题和答案")

            # 3. 向量化并存储到Chroma
            collection_name = f"{question_type}_questions"
            db = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding,
                persist_directory=CHROMA_DB_DIR,
                metadatas=[{
                    "source": questions_path,
                    "type": question_type,
                    "index": i,
                    "answer": answer
                } for i, answer in enumerate(answer_texts)],
                collection_name=collection_name
            )

            # 更新对应的数据库引用
            if question_type == "tech":
                self.tech_db = db
            else:
                self.behavior_db = db

            success_msg = f"成功将{len(texts)}个{question_type}问题向量化并存储到数据库"
            print(success_msg)
            return success_msg

        except Exception as e:
            error_msg = f"处理文件时出现错误: {str(e)}"
            print(error_msg)
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            return error_msg
# 搜索相似问题
    def search_questions(self, query: str, question_type: str = "tech", k: int = 1) -> List[Dict]:
        """搜索相似问题
        
        Args:
            query: 搜索查询
            question_type: 问题类型 ("tech" 或 "behavior")
            k: 返回结果数量
            
        Returns:
            List[Dict]: 相似问题列表，包含问题和答案
        """
        try:
            db = self.tech_db if question_type == "tech" else self.behavior_db
            if db is None:
                print(f"未找到{question_type}类型的数据库")
                return []
            # 对提问的问题也要相似度处理
            # 直接使用文本进行相似度搜索，Chroma 内部会自动进行向量化
            results = db.similarity_search_with_score(query, k=k)
            print(f"---------------搜索到{len(results)}个相似问题")

            if not results:
                print("未找到相似问题")
                return []

            formatted_results = []
            for doc, score in results:
                result = {
                    "question": doc.page_content,
                    "answer": doc.metadata.get("answer", "暂无答案"),
                    "similarity_score": round(score, 4),
                    "source": doc.metadata.get("source", "未知来源"),
                    "type": doc.metadata.get("type", "未知类型")
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"搜索问题时出现错误: {str(e)}")
            return []

# 利用llm 对问题进行判断是tech 还是behavior
    def judge_question_type(self, query: str) -> str:
        """利用llm 对问题进行判断是tech 还是behavior
        
        Args:
            query: 用户输入的问题
            
        Returns:
            str: 问题类型 ("tech" 或 "behavior")
        """
        try:
            # 构建提示词
            prompt = f"""请分析以下问题，判断它是技术问题(tech)还是行为问题(behavior)。
问题类型定义：
- tech: 与技术相关的问题，如编程语言、框架、算法、数据库等技术性问题
- behavior: 与个人行为、团队协作、项目管理、沟通等软技能相关的问题

问题: {query}

请只回答 "tech" 或 "behavior"，不要包含其他内容。"""

            # 使用LLM进行判断
            response = self.llm.invoke(prompt)
            # 获取 AIMessage 的内容
            question_type = response.content.strip().lower()
            
            # 验证返回结果
            if question_type not in ["tech", "behavior"]:
                print(f"LLM返回了意外的结果: {question_type}，默认使用tech类型")
                return "tech"
                
            print(f"问题类型判断结果: {question_type}")
            return question_type
            
        except Exception as e:
            print(f"判断问题类型时出现错误: {str(e)}")
            print("默认使用tech类型")
            return "tech"

# 文本进行向量化处理
    def vectorize_query(self, query: str) -> List[float]:
        """将用户输入的问题转换为向量
        
        Args:
            query: 用户输入的问题文本
            
        Returns:
            List[float]: 问题的向量表示
        """
        try:
            # 使用 embedding 模型将问题转换为向量
            query_vector = self.embedding.embed_query(query)
            return query_vector
        except Exception as e:
            print(f"问题向量化处理时出现错误: {str(e)}")
            return []

def test_question_bank():
    """测试问题库功能"""
    print("\n=== 开始测试问题库功能 ===")
    
    # 初始化问题库
    question_bank = QuestionBank()
    question_bank.load_file_json(TECH_QUESTIONS)

    # 测试用例
    # test_queries = [
    #     "模糊查询like语句该怎么写?",
    #     "方法能重载吗",
    #     "请举例说明一个团队协作中出现过的冲突，并描述你是如何解决这个冲突的？",
    #     "如何处理数据库连接池？"
    # ]
    #
    # print("\n=== 开始搜索测试 ===")
    # for query in test_queries:
    #     print(f"\n测试查询: {query}")
    #     print("-" * 50)
    #
    #     # 利用llm 对问题进行判断是tech 还是behavior
    #     question_type = question_bank.judge_question_type(query)
    #     print(f"判断的问题类型: {question_type}")
    #
    #     # 根据判断的类型进行搜索
    #     results = question_bank.search_questions(query, question_type, k=1)
    #     if not results:
    #         print("未找到相关答案")
    #     for result in results:
    #         print(f"问题: {result['question']}")
    #         print(f"答案: {result['answer']}")
    #         print(f"相似度: {result['similarity_score']}")
    #         print(f"来源: {result['source']}")
    #         print(f"类型: {result['type']}")
    #
    #     print("-" * 50)


if __name__ == "__main__":
    test_question_bank()
