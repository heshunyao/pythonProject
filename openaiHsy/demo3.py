
import os
os.environ['MODELSCOPE_CACHE'] = 'D:/modelscope_cache'
from modelscope import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
