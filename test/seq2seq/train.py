from seq_tokenizer import *
from seq_model import *
import torch
import torch.optim as optim

vocab_size = 0  # 假设词汇表大小
vocab_dict = {}

# 假设的训练数据
datasets = [
    {"question":"你好吗？","answer":"我很好。"},
    {"question":"你叫什么名字？","answer":"我叫小王。"},
    {"question":"你喜欢什么书？","answer":"我喜欢《哈利波特》"},
    {"question":"中国的首都是什么？","answer":"北京"}
]

spec_token = ["<SOS>","<EOS>"]
sens = []
for dataset in datasets:
    question = dataset["question"]
    answer = dataset["answer"]
    #构建词汇表
    sens.append(question)
    sens.append(answer)

vocab_dict,vocab_size = make_vocab(sens,spec_tokens=spec_token)

print(vocab_dict)

SOS_token_id = convert_word_to_token_id("<SOS>",vocab_dict=vocab_dict)  # Start Of Sequence Token
EOS_token_id = convert_word_to_token_id("<EOS>",vocab_dict=vocab_dict)  # End Of Sequence Token
vocab_dict_reverse = {v: k for k, v in vocab_dict.items()}
print(vocab_dict_reverse)

hidden_size = 256
encoder = Encoder(vocab_size, hidden_size)
decoder = Decoder(hidden_size, vocab_size)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

criterion = nn.NLLLoss()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=16):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_tensor)
    target_length = len(target_tensor)
	#每一个时间步的hidden state
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0
    # print("---------------encoder---------------------------------")
    #一个token一个token送入encoder编码器

    encoder_outputs_all, encoder_hidden = encoder.forward_sequence(input_tensor)
    encoder_outputs_all = encoder_outputs_all.squeeze(1)  # [seq_len, hidden_size]

    # for ei in range(input_length):
    #     encoder_output = encoder_outputs_all[ei]  # [hidden_size]
    #     encoder_outputs[ei] = encoder_output  # 如果你预定义了 encoder_outputs 张量

    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0, 0]	# 去除batch_size,seq_length维度
    #     # print("encoder_outputs",encoder_outputs[ei].shape)

    decoder_input = torch.tensor([[SOS_token_id]])
    #最终隐藏状态=上下文向量，作为解码器的初始状态
    decoder_hidden = encoder_hidden

    # print("---------------decoder--######-------------------")
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # print("decoder_output",decoder_output.shape)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        # print("target_tensor",target_tensor[di])
        #交叉熵损失
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token_id:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

for epoch in range(500):
    for data in datasets:
        question = data["question"]
        answer = data["answer"]
        inputs = convert_sentence_to_token_ids(question,vocab_dict)
        inputs.append(EOS_token_id)
        input_tensor = torch.LongTensor(inputs)
        targets = convert_sentence_to_token_ids(answer,vocab_dict)
        targets.append(EOS_token_id)
        target_tensor = torch.LongTensor(targets)
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss {loss:.4f}')


def predict_qa(encoder, decoder, sentence, max_length=16):
    encoder.eval()
    decoder.eval()
    input = convert_sentence_to_token_ids(sentence,vocab_dict)
    input_tensor = torch.LongTensor(input)
    encoder_hidden = encoder.initHidden()
    for i in range(len(input_tensor)):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
    #经过编码器运算，得到上下文向量
    decoder_input = torch.tensor(SOS_token_id)
    decoder_hidden = encoder_hidden
    # print("SOS_token",SOS_token_id)
    # print("EOS_token",EOS_token_id)
    print("decoder_input",decoder_input)
    print("-------------------------------------------------")
    while True:
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        print(decoder_output)
        topv, topi = decoder_output.topk(1)
        print(topi)
        _,ind = torch.max(decoder_output,dim=1)
        print("next:",vocab_dict_reverse[ind.item()])
        decoder_input = topi.squeeze().detach()
        if decoder_input.item() == EOS_token_id:
            break
        
predict_qa(encoder,decoder,"你叫什么名字？")