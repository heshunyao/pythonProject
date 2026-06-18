from openai import OpenAI


class InterviewAI:
    def __init__(self, api_key: str, base_url: str = "https://api.chatanywhere.tech/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.context_messages = []

    def prepare_interview_context(self, questions: list):
        """
        初始self.client = OpenAI(api_key=api_key, base_url=base_url)化上下文并模拟一次“训练” —— 让 AI 先分析题目，比如进行筛选、归类等
        """
        combined_questions = "\n".join([f"{i + 1}. {q['content']}" for i, q in enumerate(questions)])

        # 设定系统角色和用户提问
        self.context_messages = [
            {"role": "system", "content": "你是一位资深的Java面试官，以下是你的面试题：\n" + combined_questions},
            {"role": "system", "content": "除了我提供的数据，不能回答其它问题"},
            # {"role": "user", "content": "请你先分析这些题目，归类出不同类型（如基础语法、面向对象、多线程、JVM、集合等），并挑选出每类中最能代表核心基础的1~2题。"}
        ]

        # 调用模型进行“训练分析”
        print("AI（初始化分析中）：", end="", flush=True)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.context_messages,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        # 将分析结果也加入上下文
        self.context_messages.append({"role": "assistant", "content": full_response})
        print("\n初始化完成。")

    def ask_followup_question(self, user_question: str) -> str:
        """
        后续提问，基于已有“分析训练”上下文提问
        """
        if not self.context_messages:
            raise ValueError("请先调用 prepare_interview_context 初始化上下文！")

        messages = self.context_messages + [{"role": "user", "content": user_question}]

        print("AI：", end="", flush=True)

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        # 更新对话历史
        self.context_messages.append({"role": "user", "content": user_question})
        self.context_messages.append({"role": "assistant", "content": full_response})

        return full_response
