from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class InterviewState(Enum):
    """面试状态枚举"""
    INITIAL = "initial"  # 初始状态
    CHATTING = "chatting"  # 自由对话
    INTERVIEWING = "interviewing"  # 面试中
    EVALUATING = "evaluating"  # 评估中
    COMPLETED = "completed"  # 已完成


class InterviewSession:
    def __init__(self, position: str, level: str):
        """初始化面试会话
        
        Args:
            position: 面试职位
            level: 面试级别
        """
        self.position = position
        self.level = level
        self.state = InterviewState.INITIAL
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

        # 面试数据
        self.chat_history: List[Dict] = []
        self.questions: List[Dict] = []
        self.current_question: Optional[Dict] = None
        self.current_round = 0
        self.max_rounds = 5

        # 评分数据
        self.scores: List[Dict] = []

    def start_interview(self) -> bool:
        """开始面试
        
        Returns:
            bool: 是否成功开始面试
        """
        if self.state != InterviewState.CHATTING:
            return False

        self.state = InterviewState.INTERVIEWING
        self.current_round = 0
        return True

    def add_chat_message(self, role: str, content: str):
        """添加聊天消息
        
        Args:
            role: 角色（user/assistant）
            content: 消息内容
        """
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def add_question(self, question: Dict):
        """添加面试问题
        
        Args:
            question: 问题信息
        """
        self.current_round += 1
        question["round"] = self.current_round
        self.questions.append(question)
        self.current_question = question

    def add_score(self, score: Dict):
        """添加评分
        
        Args:
            score: 评分信息
        """
        self.scores.append(score)

    def complete_question(self):
        """完成当前问题"""
        if self.current_question:
            self.current_question["completed"] = True
            self.current_question = None

    def is_interview_completed(self) -> bool:
        """检查面试是否完成
        
        Returns:
            bool: 是否完成
        """
        return (self.state == InterviewState.INTERVIEWING and
                self.current_round >= self.max_rounds and
                not self.current_question)

    def complete_interview(self):
        """完成面试"""
        self.state = InterviewState.COMPLETED
        self.end_time = datetime.now()

    def get_session_data(self) -> Dict:
        """获取会话数据
        
        Returns:
            Dict: 会话数据
        """
        return {
            "position": self.position,
            "level": self.level,
            "state": self.state.value,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else None,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "chat_history": self.chat_history,
            "questions": self.questions,
            "scores": self.scores
        }

    def should_generate_question(self) -> bool:
        """判断是否应该生成新问题
        
        Returns:
            bool: 是否应该生成新问题
        """
        return (self.state == InterviewState.INTERVIEWING and
                self.current_round < self.max_rounds and
                not self.current_question)

    def should_evaluate_answer(self) -> bool:
        """判断是否应该评估答案
        
        Returns:
            bool: 是否应该评估答案
        """
        return (self.state == InterviewState.INTERVIEWING and
                self.current_question and
                not self.current_question.get("completed", False))


def test_interview_session():
    """测试面试会话管理"""
    print("\n=== 开始测试面试会话管理 ===")

    # 创建会话
    session = InterviewSession("Java开发工程师", "高级")

    # 测试状态转换
    print("\n1. 测试状态转换")
    session.state = InterviewState.CHATTING
    print(f"开始面试: {session.start_interview()}")
    print(f"当前状态: {session.state.value}")

    # 测试添加消息
    print("\n2. 测试添加消息")
    session.add_chat_message("user", "我准备好了")
    session.add_chat_message("assistant", "好的，让我们开始面试")
    print(f"聊天历史: {len(session.chat_history)} 条消息")

    # 测试添加问题
    print("\n3. 测试添加问题")
    question = {
        "question": "请介绍下Spring Boot的主要特点",
        "type": "tech",
        "source": "vector_db"
    }
    session.add_question(question)
    print(f"当前轮次: {session.current_round}")
    print(f"当前问题: {session.current_question}")

    # 测试添加评分
    print("\n4. 测试添加评分")
    score = {
        "round": session.current_round,
        "final_score": 85.0,
        "details": {
            "content_score": {"score": 85.0, "reason": "回答全面"}
        }
    }
    session.add_score(score)
    print(f"评分数量: {len(session.scores)}")

    # 测试完成问题
    print("\n5. 测试完成问题")
    session.complete_question()
    print(f"当前问题: {session.current_question}")

    # 测试会话数据
    print("\n6. 测试获取会话数据")
    session_data = session.get_session_data()
    print(f"会话数据: {session_data.keys()}")


if __name__ == "__main__":
    test_interview_session()
