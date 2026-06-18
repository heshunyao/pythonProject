# 面试官的智能体，可以多轮出题，对用户回答内容，并且存储对话历史记录，最终多维度的评分， 及报告
from ai_interview.tools.QuestionBank import QuestionBank
from ai_interview.tools.ScoreSystem import ScoreSystem
from ai_interview.tools.ShowQuestion import QuestionGenerator
from ai_interview.tools.interview_tools import InterviewTools
from ai_interview.tools.ReportVisualizer import ReportVisualizer
from ai_interview.agent.InterviewState import InterviewSession, InterviewState
from typing import List, Dict, Optional
import json
import os
import time
from datetime import datetime
from openai import OpenAI
from openai.types.chat import ChatCompletion
import random


class InterviewerAgent:
    def __init__(self, client: OpenAI, position: str = "软件开发工程师", level: str = "高级"):
        """初始化面试官智能体
        
        Args:
            client: OpenAI客户端实例
            position: 面试职位
            level: 面试级别A
        """
        self.client = client
        self.position = position
        self.level = level

        # 初始化工具类
        self.question_bank = QuestionBank()
        self.score_system = ScoreSystem(self.question_bank)
        self.question_generator = QuestionGenerator(self.question_bank)
        self.report_visualizer = ReportVisualizer()

        # 获取工具定义
        self.tools = InterviewTools.get_tools()

        # 初始化面试会话
        self.session = InterviewSession(position, level)
        self.session.state = InterviewState.CHATTING  # 初始状态为自由对话

        # API调用配置
        self.max_retries = 3
        self.retry_delay = 1  # 基础延迟时间（秒）
        self.max_delay = 10  # 最大延迟时间（秒）

        # 系统提示词
        self.system_prompt = f"""
        您是一位专业的软件开发技术总监，为企业进行筛选各方面的开发人员，并对候选人进行技术评估和行为评估。您会要求候选按面试流程进行面试。
        根据候选人的回答，生成合适的问题，并评估候选人的回答质量。
        您需要：
        1. 在自由对话阶段，回答候选人的问题，并适时引导进入面试环节，必须在面试之前结束自由对话，以面试者回答"准备好"为结束自由对话。
        2. 在面试者回答"准备好"，你必须提问第一个问题：我想了解一下您的背景和职位。您可以简要介绍一下自己吗？
        3. 在面试者回答完自我介绍后，才能正式进行面试技术或职业观问题的提问；并结合面试者的自我介绍内容，生成合适的问题。
        4. 在面试阶段，根据候选人的回答生成合适的后续问题
        5. 评估候选人的回答质量
        6. 在面试结束时生成详细的评估报告

        面试过程中，您需要遵循以下原则：
        1. 问题应该循序渐进，从基础到深入
        2. 技术问题和行为问题要合理搭配
        3. 评分要客观公正，考虑多个维度
        4. 及时给予反馈，引导候选人展示真实水平

        您可以使用以下工具：
        {self._format_tools_description()}

        在面试过程中，您需要按照以下格式进行思考和行动：

        示例对话：
        候选人: 你好，我想参加面试
        面试官: 你好！很高兴您有面试的机会。一切准备好了吗？

        候选人: 我准备好了，可以开始面试        
        Thought: 候选人已经准备好了，我需要开始面试。
        
        面试官: 我想了解一下您的背景和职位。您可以简要介绍一下自己吗？
        
        候选人: 我是一名Java开发工程师，有5年工作经验，主要做后端开发。
        
        Thought: 候选人介绍了自己的背景，我需要评估他的技术深度和经验。应该从Java核心技术开始提问。
        Action: generate_question: {{"chat_history": "候选人介绍自己是Java开发工程师", "position": "Java开发工程师", "level": "高级"}}
        
        Observation: {{"question": "请详细说明Java中的垃圾回收机制，以及不同垃圾回收器的特点", "type": "tech", "source": "vector_db"}}
        
        Thought: 问题已经生成，需要等待候选人的回答。
        Answer: 好的，我了解到您有5年的Java开发经验。让我们从Java的核心机制开始。请详细说明Java中的垃圾回收机制，以及不同垃圾回收器的特点。

        候选人: Java的垃圾回收主要分为新生代和老年代，新生代使用复制算法，老年代使用标记整理算法。常用的垃圾回收器有Serial、Parallel、CMS和G1等。

        Thought: 候选人的回答比较基础，需要进一步考察他对垃圾回收器的深入理解。
        Action: evaluate_answer: {{"answer": "Java的垃圾回收主要分为新生代和老年代...", "question": "请详细说明Java中的垃圾回收机制"}}
        
        Observation: {{"final_score": 75.0, "details": {{"content_score": {{"score": 75.0, "reason": "回答基本正确但不够深入"}}}}}}
        
        Thought: 根据评分结果，候选人需要补充更多细节。我应该引导他深入讨论G1收集器的特点。
        Action: generate_question: {{"chat_history": "候选人讨论了垃圾回收的基本机制", "position": "Java开发工程师", "level": "高级"}}
        
        Observation: {{"question": "请详细说明G1垃圾回收器的工作原理和优势", "type": "tech", "source": "llm_generated"}}
        
        Thought: 已经生成了更深入的问题，需要等待候选人的回答。
        Answer: 您对垃圾回收的基本机制有一定了解。让我们深入讨论一下G1垃圾回收器，请详细说明它的工作原理和优势。

        在面试过程中，您需要：
        1. 思考：分析当前情况，决定下一步行动
        2. 行动：调用适当的工具
        3. 观察：获取工具返回的结果
        4. 回答：根据观察结果给出回应

        请按照以下格式输出：
        Thought: [您的思考过程]
        Action: [工具名称]: [参数]
        Observation: [工具返回的结果]
        Answer: [给候选人的回应]
        """.strip()

        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _format_tools_description(self) -> str:
        """格式化工具描述，用于系统提示词
        
        Returns:
            str: 格式化后的工具描述
        """
        descriptions = InterviewTools.get_tool_descriptions()
        return "\n".join([f"{i + 1}. {name}: {desc}"
                          for i, (name, desc) in enumerate(descriptions.items())])

    def __call__(self, message: str) -> str:
        """处理候选人的输入消息
        
        Args:
            message: 候选人的输入消息
            
        Returns:
            str: 面试官的回应
        """
        try:
            # 添加用户消息到会话历史
            self.session.add_chat_message("user", message)
            self.messages.append({"role": "user", "content": message})

            # 检查是否需要开始面试
            if (self.session.state == InterviewState.CHATTING and
                    "准备好" in message.lower()):
                self.session.start_interview()         # 状态的改变

            # 执行对话
            response = self.execute()

            if not response:
                return "抱歉，系统暂时无法生成回应，请稍后重试。"

            # 添加助手消息到会话历史
            self.session.add_chat_message("assistant", response)
            self.messages.append({"role": "assistant", "content": response})

            # 检查面试是否完成
            if self.session.is_interview_completed():
                self._generate_final_report()

            return response

        except Exception as e:
            error_msg = f"处理消息时出现错误: {str(e)}"
            print(error_msg)
            return "抱歉，系统处理您的输入时出现错误，请稍后重试。"

    def execute(self) -> Optional[str]:
        """执行面试流程
        
        Returns:
            Optional[str]: 面试官的回应，如果出错则返回None
        """
        for attempt in range(self.max_retries):
            try:
                # 调用大模型进行对话
                chat_response = self.chat_completion_request(
                    self.messages,
                    tools=self.tools
                )

                if not chat_response or not chat_response.choices:
                    if attempt < self.max_retries - 1:
                        # 在重试之前等待
                        self._wait_before_retry(attempt)
                        continue
                    return None

                response = chat_response.choices[0].message

                # 处理工具调用
                if response.tool_calls:
                    # 添加assistant消息作为工具调用的前置消息
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": response.tool_calls
                    })
                    
                    for tool_call in response.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        # 执行工具调用
                        print(
                            f"==============执行工具调用: {function_name}({function_args})"
                        )
                        if function_name == "generate_question":
                            result = self._handle_generate_question(function_args)
                        elif function_name == "evaluate_answer":
                            result = self._handle_evaluate_answer(function_args)
                        elif function_name == "generate_report":
                            result = self._handle_generate_report(function_args)
                        else:
                            result = f"未知工具调用: {function_name}"

                        # 添加工具调用结果到对话历史
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(result, ensure_ascii=False)
                        })

                        # 继续对话
                        chat_response = self.chat_completion_request(self.messages, tools=self.tools)
                        if chat_response and chat_response.choices:
                            return chat_response.choices[0].message.content

                return response.content

            except Exception as e:
                print(f"执行过程中出现错误 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    self._wait_before_retry(attempt)
                    continue
                return None

    def _wait_before_retry(self, attempt: int):
        """在重试之前等待
        
        Args:
            attempt: 当前尝试次数
        """
        # 使用指数退避策略，并添加随机抖动
        delay = min(self.retry_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
        print(f"等待 {delay:.2f} 秒后重试...")
        time.sleep(delay)

    def chat_completion_request(self, messages, tools=None, tool_choice=None, model="gpt-3.5-turbo") -> Optional[
        ChatCompletion]:
        """调用OpenAI API进行对话
        
        Args:
            messages: 对话历史
            tools: 可用工具列表
            tool_choice: 工具选择策略
            model: 使用的模型名称
            
        Returns:
            Optional[ChatCompletion]: OpenAI API的响应，如果出错则返回None
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
            return response
        except Exception as e:
            print(f"调用OpenAI API时出现错误: {str(e)}")
            return None

    def _handle_generate_question(self, args: Dict) -> Dict:
        """处理生成问题的工具调用
        
        Args:
            args: 工具调用参数
            
        Returns:
            Dict: 生成的问题信息
        """
        try:
            if not self.session.should_generate_question():
                return {"error": "当前状态不需要生成问题"}

            # 构建对话历史
            chat_history = "\n".join([
                f"面试官: {msg['content']}\n候选人: {msg['content']}"
                for msg in self.session.chat_history[-4:]  # 只使用最近的对话
            ])

            # 生成新问题
            question_info = self.question_generator.generate_question(
                chat_history=chat_history,
                position=self.position,
                level=self.level
            )

            # 记录问题
            self.session.add_question(question_info)

            return question_info

        except Exception as e:
            return {"error": f"生成问题时出现错误: {str(e)}"}

    def _handle_evaluate_answer(self, args: Dict) -> Dict:
        """处理评估答案的工具调用
        
        Args:
            args: 工具调用参数
            
        Returns:
            Dict: 评分结果
        """
        try:
            if not self.session.should_evaluate_answer():
                return {"error": "当前状态不需要评估答案"}

            # 获取当前问题信息
            current_question = self.session.current_question

            # 计算评分
            score_result = self.score_system.calculate_final_score(
                answer=args["answer"],
                question=current_question["question"],
                reference_answer=current_question.get("answer")
            )

            # 记录评分
            self.session.add_score({
                "round": self.session.current_round,
                "question": current_question["question"],
                "type": current_question["type"],
                "score": score_result
            })

            # 完成当前问题
            self.session.complete_question()

            return score_result

        except Exception as e:
            return {"error": f"评估答案时出现错误: {str(e)}"}

    def _handle_generate_report(self, args: Dict) -> Dict:
        """处理生成报告的工具调用
        
        Args:
            args: 工具调用参数
            
        Returns:
            Dict: 面试报告
        """
        try:
            if not self.session.is_interview_completed():
                return {"error": "面试尚未完成"}

            # 获取会话数据
            session_data = self.session.get_session_data()

            # 将数据扔大数据进行  大综合评估

            # 计算维度得分
            final_scores = self.report_visualizer.calculate_dimension_scores(
                session_data["questions"]
            )
            session_data["final_scores"] = final_scores

            # 生成报告
            print("========正在生成面试报告=============")
            output_dir = "interview_reports"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_dir, f"interview_report_{timestamp}.html")

            self.report_visualizer.generate_html_report(session_data, report_path)

            # 完成面试
            self.session.complete_interview()

            return {
                "status": "success",
                "message": f"面试报告已生成: {report_path}",
                "report_path": report_path
            }

        except Exception as e:
            return {"error": f"生成报告时出现错误: {str(e)}"}

    def _generate_final_report(self):
        """生成最终面试报告"""
        try:
            # 调用生成报告工具
            result = self._handle_generate_report({"interview_data": self.session.get_session_data()})

            if "error" in result:
                print(f"生成报告失败: {result['error']}")
            else:
                print(f"面试报告已生成: {result['report_path']}")

        except Exception as e:
            print(f"生成最终报告时出现错误: {str(e)}")


def test_interviewer_agent():
    """测试面试官智能体功能"""
    print("\n=== 开始测试面试官智能体 ===")

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV",
        base_url="https://api.chatanywhere.tech/v1"
    )

    # 初始化面试官
    interviewer = InterviewerAgent(
        client=client,
        position="Java开发工程师",
        level="高级"
    )

    # 模拟面试对话
    #     test_messages = [
    #         "你好，我想参加面试",
    #         "我准备好了，可以开始面试",
    #         """MyBatis是一个优秀的持久层框架，它的工作原理主要包括：
    # 1. 配置映射：通过XML或注解方式配置SQL映射
    # 2. 动态SQL：支持动态生成SQL语句
    # 3. 缓存机制：提供一级缓存和二级缓存
    # 4. 延迟加载：支持关联对象的延迟加载""",
    #
    #         """在团队协作中，我注重以下几点：
    # 1. 及时沟通：定期同步项目进度和遇到的问题
    # 2. 知识分享：组织技术分享会，促进团队成长
    # 3. 代码审查：严格执行代码审查，保证代码质量
    # 4. 文档维护：及时更新技术文档，方便团队协作"""
    #     ]

    # 执行对话
    # for message in test_messages:
    #     print(f"\n候选人: {message}")
    #     try:
    #         response = interviewer(message)
    #         print(f"面试官: {response}")
    #     except Exception as e:
    #         print(f"处理消息时出现错误: {str(e)}")
    #         print("继续处理下一条消息...")

    while True:
        message = input("请输入：")
        print(f"\n候选人: {message}")
        try:
            response = interviewer(message)
            print(f"面试官: {response}")
        except Exception as e:
            print(f"处理消息时出现错误: {str(e)}")
            print("继续处理下一条消息...")
        # if message == "exit":
        #     break


if __name__ == "__main__":

    test_interviewer_agent()
