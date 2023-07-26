import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, DistilBertForSequenceClassification

from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


'''
从这个网站下载的：https://github.com/CLUEbenchmark/CLUEPretrainedModels

Epoch 1/5:
Train Loss: 0.3249, Train Accuracy: 0.8391
Test Loss: 0.2888, Test Accuracy: 0.8444

Epoch 2/5:
Train Loss: 0.2655, Train Accuracy: 0.8682
Test Loss: 0.2863, Test Accuracy: 0.8604

Epoch 3/5:
Train Loss: 0.2433, Train Accuracy: 0.8885
Test Loss: 0.2767, Test Accuracy: 0.8750

Epoch 4/5:
Train Loss: 0.2166, Train Accuracy: 0.9038
Test Loss: 0.3577, Test Accuracy: 0.8729

Epoch 5/5:
Train Loss: 0.1925, Train Accuracy: 0.9253
Test Loss: 0.4597, Test Accuracy: 0.8708

F1 Macro Score on Test Set: 0.6163


'''
# 加载数据集
data = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset-4movies.xls')
data['sentiment'] = data['rating'].apply(lambda x: 2 if x > 3 else (1 if x == 3 else 0))
X = data['comment'].fillna("NA")  # 处理缺失值
y = data['sentiment']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Bert分词器和模型（使用本地预训练模型）
model_path = r"F:\pycharm_project\7071\7150CEM\model3"
tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)

# 将模型转移到GPU上（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 对数据进行分词和编码
def tokenize(text, max_length=128):
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

encoded_train = [tokenize(text) for text in X_train]
encoded_test = [tokenize(text) for text in X_test]

# 创建TensorDataset和DataLoader
batch_size = 8
train_dataset = TensorDataset(torch.cat([inputs['input_ids'] for inputs in encoded_train]),
                              torch.cat([inputs['attention_mask'] for inputs in encoded_train]),
                              torch.tensor(y_train.values))
test_dataset = TensorDataset(torch.cat([inputs['input_ids'] for inputs in encoded_test]),
                             torch.cat([inputs['attention_mask'] for inputs in encoded_test]),
                             torch.tensor(y_test.values))

# 创建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 使用混合精度训练（如果GPU支持）
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

# 训练模型 (train model)
def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss, total_accuracy = 0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 使用autocast对前向和后向传递进行混合精度训练
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        total_accuracy += accuracy_score(label_ids, np.argmax(logits, axis=1))

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy

# 测试模型
def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss, total_accuracy = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            total_accuracy += accuracy_score(label_ids, np.argmax(logits, axis=1))

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy

# 训练和评估模型
epochs = 5

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_dataloader, loss_fn, optimizer)
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print()

# 在测试集上进行预测
def predict(model, dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits.detach().cpu().numpy()
            predictions.extend(np.argmax(logits, axis=1))

    return predictions

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train')
plt.plot(range(1, epochs+1), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='Train')
plt.plot(range(1, epochs+1), test_accuracies, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 在测试集上进行预测并计算F1分数
y_pred = predict(model, test_dataloader)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"F1 Macro Score on Test Set: {f1_macro:.4f}")
