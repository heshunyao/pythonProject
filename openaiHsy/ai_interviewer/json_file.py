import json


class QuestionBatchReader:
    def __init__(self, file_path: str, batch_size: int = 20):
        """
        初始化面试题读取器。

        :param file_path: JSON 文件路径
        :param batch_size: 每次读取的批次大小，默认为20条
        """
        self.file_path = file_path
        self.batch_size = batch_size

    def load_questions_in_batches(self):
        """
        按批次读取面试题数据，返回一个生成器，每次返回一个批次的题目。
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    def get_question_count(self):
        """
        获取总的面试题数量
        """
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
