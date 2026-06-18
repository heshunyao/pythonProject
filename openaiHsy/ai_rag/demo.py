# import numpy as np
# a = np.random.randint(0, 11, size=(2, 3))
# print("a",a)
#
# b=np.arange(1, 7, 1).reshape(2, 3).T
# print("b",b)
#
#
# c=np.array([1,2,3,4,5])
# print("c",c)
#
#
# d=np.dot(a,b)
# print("d",d)

import jieba
import torch
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor



# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope.models import Model
# from modelscope.preprocessors import TokenClassificationTransformersPreprocessor
#
# model_id = 'D:/modelscope_cache/models/iic/nlp_lstmcrf_word-segmentation_chinese-ecommerce'
# model = Model.from_pretrained(model_id)
# tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
# pipeline_ins = pipeline(task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)
# result = pipeline_ins(input="收腰显瘦黑裙长裙")
# print(result['output'])

# {'output': '收腰 显瘦 黑裙 长裙'}

# import os
# os.environ['MODELSCOPE_CACHE'] = 'D:/modelscope_cache'

# from modelscope import AutoTokenizer, AutoModelForMaskedLM
#
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
#
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")


from modelscope import AutoTokenizer, AutoModelForMaskedLM

# 指定模型的本地路径
local_model_path = "D:/modelscope_cache/models/bert-base-chinese"

# 加载 tokenizer 和模型
# 明确指定 local_files_only=True 表示使用本地模型
#分词器对象
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
#加载模型
model = AutoModelForMaskedLM.from_pretrained(local_model_path, local_files_only=True)
# 输入文本，使用 [MASK] 作为掩码
input_text = [
    '苹果和[MASK]是很常见的水果。',
    '我喜欢吃[MASK]和百香果。',
    '苹果今年推出了新的[MASK]。',
    '华为的新[MASK]拍照性能非常好。'
]


# 使用 tokenizer 对文本进行编码
# 将输出格式指定为 PyTorch 的张量（tensor），也可以设置为 "tf"（TensorFlow）或 "np"（NumPy）。
# padding=True 表示：对多个输入句子进行 自动填充，将它们填充到相同的长度。
#  truncation=True  表示：当输入句子 过长 超出模型最大长度时，进行截断处理。
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

print(inputs)

import torch

# 获取模型输出  **把里面所有的值都传给模型
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits
logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
# print(logits,"logits")

# 对每一句找到[MASK]的位置，提取预测
# 遍历每一句输入（即每一条 input_text 中的句子）
for i, input_ids in enumerate(inputs['input_ids']):
    print(input_ids,'input_ids')
    # 找出当前句子中 [MASK] 的位置
    # mask_token_id 是 [MASK] 的 token 对应的 ID，== 比较后是布尔张量
    # .nonzero(as_tuple=True)[0] 返回所有为 True 的位置索引
    mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    print(mask_index,"mask_index")

    # 如果该句子中没有 [MASK]，跳过处理
    if len(mask_index) == 0:
        print(f"第{i+1}句中没有 [MASK]")
        continue

    # 从模型输出的 logits 中提取出 [MASK] 位置的预测结果
    # logits[i] 表示第 i 条句子的预测结果（形状为 [seq_len, vocab_size]）
    # logits[i, mask_index[0]] 表示该句中 [MASK] 位置对应的 vocab 预测概率
    # .argmax(dim=-1) 找出概率最大的 token id
    predicted_token_id = logits[i, mask_index[0]].argmax(dim=-1).item()

    # 将预测出的 token id 解码为具体的词/字
    predicted_token = tokenizer.decode([predicted_token_id])

    # 打印结果
    print(f"第{i+1}句预测: {predicted_token}")

