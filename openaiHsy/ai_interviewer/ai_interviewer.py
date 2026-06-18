from openai import OpenAI

from openaiHsy.ai_interviewer.InterviewAI import InterviewAI
from openaiHsy.ai_interviewer.json_file import QuestionBatchReader

def main():
    # 初始化文件读取器
    question_reader = QuestionBatchReader("knowledge_base/java.json", batch_size=20)
    print(question_reader.load_questions_in_batches())
    # 初始化
    interview_ai = InterviewAI(api_key="sk-ENs53O4eMJfvwIF3eUTyc3xGUgkGu6mo8nIw4iInjwFzJalV")

    # 加载所有问题（这里你可以按批量或一次性加载）
    questions = question_reader.load_questions_in_batches()

    # 模拟训练
    interview_ai.prepare_interview_context(questions)

    # 进入提问循环
    print("\n开始提问（输入 'exit' 退出）")
    while True:
        user_input = input("\n你：")
        if user_input.lower().strip() == "exit":
            print("结束对话。")
            break

        interview_ai.ask_followup_question(user_input)

if __name__ == "__main__":
    main()




