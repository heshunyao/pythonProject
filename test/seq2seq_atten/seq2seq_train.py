import torch
import torch.optim as optim
import torch.nn as nn
import math
from seq2seq_dataset import train_data_loader ,test_data_loader,tokenizer_zh,tokenizer_en
from seq2seq_model import Seq2Seq,Encoder,Decoder,Attention


# 定义超参数

INPUT_DIM = len(tokenizer_zh)
OUTPUT_DIM = len(tokenizer_en)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


# 初始化模型
attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters()).to(device)
#tokenizer_en.token2idx["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 训练模型
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src = batch["src_ids"].to(device)
        trg = batch["trg_ids"].to(device)
        print("train src=",src.shape)
        print("train trg=",trg.shape)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        print("train output=",output.shape)
        if model.decoder.rnn.batch_first:
            output = output[:,1:].contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def inference(model, text):
    # 设置模型为评估模式，关闭dropout等训练特性
    model.eval()

    # 将输入的中文文本编码成id序列（如：[CLS], 你, 好, [SEP] → [101, 872, 1962, 102]）
    encoder_ids = tokenizer_zh.encode(text)

    # 转换成Tensor并增加一个维度，形成 batch_size=1 的输入
    encoder_ids = torch.tensor(encoder_ids).unsqueeze(0)

    # 编码器处理输入，输出 encoder_outputs（注意力上下文） 和最终隐藏状态 hidden
    encoder_ouputs, hidden = model.encoder(encoder_ids)

    # 初始化解码器的输入，使用英文的起始标记 <sos>
    input_ids = [tokenizer_en.token2idx["<sos>"]]

    # 转换成Tensor格式，用于送入解码器
    input_ids = torch.tensor(input_ids)

    # 设置最大生成长度为64，防止死循环
    max_len = 64

    # 初始化输出索引列表，并先放入<sos>作为起始标志
    output_indices = []
    output_indices.append(tokenizer_en.token2idx["<sos>"])

    # 关闭梯度计算，加快推理速度并节省显存
    with torch.no_grad():
        for i in range(max_len):
            # 解码器根据当前 input_ids、隐藏状态和编码器输出进行一步解码
            output, hidden = model.decoder(input_ids, hidden, encoder_ouputs)

            # 获取当前时间步输出的预测结果（概率最大的词的索引）
            next_token = output.squeeze(1).argmax(1)

            # 打印当前预测的token索引
            print("inference next_token=", next_token)

            # 将预测token添加到结果序列中
            output_indices.append(next_token)

            # 如果预测到终止符<eos>，则停止生成
            if next_token == tokenizer_en.token2idx["<eos>"]:
                break

            # 更新下一时刻的解码器输入，需要注意要用形状匹配的方式更新
            input_ids = input_ids + next_token  # ⚠️ 这里存在潜在错误，见下方说明

    # 将预测的token索引序列转回字符串并返回
    return tokenizer_en.decode(output_indices)
    
# 训练循环
N_EPOCHS = 100
CLIP = 1
task = "train"

best_valid_loss = float('inf')

if task == "train":
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_data_loader, optimizer, criterion, CLIP)

        print(f'----------------------------------Epoch: {epoch+1:02}---------------------------------------')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'seq2seq-{epoch+1}-{train_loss:.3f}.pt')
    tokenizer_zh.save(f'tokenizer_zh.pkl')
    tokenizer_en.save(f'tokenizer_en.pkl')
    torch.save(model.state_dict(), 'seq2seq-last.pt')
    
    response = inference(model,text="企业的主要任务和责任担当")
    print("total result=",response)
elif task == "predict":
    model.load_state_dict(torch.load('seq2seq.pt'))
    response = inference(model,text="企业的主要任务和责任担当")
    print(response)

