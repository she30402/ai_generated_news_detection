# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:14:56 2021

@author: GameToGo
"""

import torch
import argparse
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification



parser = argparse.ArgumentParser()
parser.add_argument('--acc_file', default='/Users/Student/Desktop/假新聞/model/best_acc.txt', type=str, required=False, help='best accuracy recorded')
parser.add_argument('--pretrained_model', default='bert-base-chinese', type=str, required=False, help='模型训练起点路径')
parser.add_argument('--epochs', default=1, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=1, type=int, required=False, help='训练batch size')
parser.add_argument('--num_labels', default=2, type=int, required=False, help='類別數量')
args = parser.parse_args()   
    
    
    
class NewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".csv")
        self.len = len(self.df)
        #self.label_map = {'finance': 0, 'fashion': 1}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1][:300]             
        label = self.df.iloc[idx, 2]
        label_tensor = torch.tensor(label)
       
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
     
    def __len__(self):
        return self.len



# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `NewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def get_batch(batch_items):
    tokens_tensors = [item[0] for item in batch_items]
    segments_tensors = [item[1] for item in batch_items]
    # 測試集有 labels
    if batch_items[0][2] is not None:
        label_ids = torch.stack([item[2] for item in batch_items])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids



def get_predictions(model, dataloader, compute_acc=True):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions
    




if __name__ == "__main__":
    print('START RUNNING')
    acc_file = args.acc_file
    BATCH_SIZE = args.batch_size
    PRETRAINED_MODEL = args.pretrained_model
    NUM_LABELS = args.num_labels
    EPOCHS = args.epochs
    # 取得此預訓練模型所使用的 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
  
    
    try:
        with open(acc_file,'r') as f:
            best_acc = float(f.read())
            print(f'best acc is {best_acc}')
    except:        
        best_acc = 0

    # 初始化Dataset，使用中文 BERT 斷詞
    trainset = NewsDataset("train", tokenizer=tokenizer)
    testset = NewsDataset("test", tokenizer=tokenizer)

    # 初始化一個每次回傳1個訓練樣本的 DataLoader
    # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                                collate_fn=get_batch, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                                collate_fn=get_batch, shuffle=True)
    

    if PRETRAINED_MODEL == 'bert-base-chinese':
        model = BertForSequenceClassification.from_pretrained(
                       PRETRAINED_MODEL, num_labels=NUM_LABELS)
    else:
        model = torch.load(PRETRAINED_MODEL)
    

    # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    
    for epoch in range(EPOCHS):
        
        model.train()
        running_loss = 0.0
        print(f'START EPOCH {epoch+1} TRAINING')
        num = 1
        for data in trainloader:
        
            tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]

            # 將參數梯度歸零
            optimizer.zero_grad()
        
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

            loss = outputs[0]  
            # backward
            loss.backward()
            optimizer.step()
            
            print(f'{num}processed')
            num+=1
               
            # 紀錄當前 batch loss
            running_loss += loss.item()
        
        # 計算分類準確率
        _, acc = get_predictions(model, trainloader, compute_acc=True)

        print(f'[train {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f}')
        
        
        
        
        
        model.eval()
        running_loss = 0.0
        print(f'START EPOCH {epoch+1} EVALUATION')
        num = 1
        for data in testloader:
        
            tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]

        
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

            loss = outputs[0]  
            
            
            print(f'{num}processed')
            num+=1

            # 紀錄當前 batch loss
            running_loss += loss.item()
        
        # 計算分類準確率
        _, acc = get_predictions(model, testloader, compute_acc=True)

        print(f'[eval {epoch + 1}] loss: {running_loss:.3f}, acc: {acc:.3f}')
        
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
           torch.save(model,'model.pth')
           with open(acc_file,'w') as f:
               f.write(f'{best_acc}')










    
