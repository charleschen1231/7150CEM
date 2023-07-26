import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import random
import torchtext
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
'''
1.8.0+cu101的环境下 完成RNN
Epoch: 01, Train Loss: 0.5452, Train Acc: 0.8313, Train F1: 0.4296, Val. Loss: 0.4504, Val. Acc: 0.8354, Val. F1: 0.7802
Epoch: 02, Train Loss: 0.5227, Train Acc: 0.8411, Train F1: 0.4633, Val. Loss: 0.4223, Val. Acc: 0.8396, Val. F1: 0.7830
Epoch: 03, Train Loss: 0.5222, Train Acc: 0.8453, Train F1: 0.4762, Val. Loss: 0.4230, Val. Acc: 0.8396, Val. F1: 0.7810
Epoch: 04, Train Loss: 0.5195, Train Acc: 0.8470, Train F1: 0.4883, Val. Loss: 0.4160, Val. Acc: 0.8368, Val. F1: 0.7745
Epoch: 05, Train Loss: 0.5152, Train Acc: 0.8491, Train F1: 0.5009, Val. Loss: 0.4253, Val. Acc: 0.8389, Val. F1: 0.7848
Epoch: 06, Train Loss: 0.5207, Train Acc: 0.8467, Train F1: 0.4934, Val. Loss: 0.4095, Val. Acc: 0.8375, Val. F1: 0.7847
Epoch: 07, Train Loss: 0.5294, Train Acc: 0.8413, Train F1: 0.4738, Val. Loss: 0.4161, Val. Acc: 0.8410, Val. F1: 0.7765
Epoch: 08, Train Loss: 0.5258, Train Acc: 0.8391, Train F1: 0.4559, Val. Loss: 0.4258, Val. Acc: 0.8389, Val. F1: 0.7806
Epoch: 09, Train Loss: 0.5194, Train Acc: 0.8455, Train F1: 0.4855, Val. Loss: 0.4402, Val. Acc: 0.8424, Val. F1: 0.7831
Epoch: 10, Train Loss: 0.5161, Train Acc: 0.8497, Train F1: 0.4953, Val. Loss: 0.4301, Val. Acc: 0.8396, Val. F1: 0.7803
Test Loss: 0.4095, Test Acc: 0.8375, Test F1: 0.7847


'''
# 设置随机种子
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# 加载数据集
data = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset-4movies.xls')

# 将评分映射为情感标签
data['sentiment'] = data['rating'].apply(lambda x: 1 if x > 3 else (0 if x == 3 else -1))

X = data['comment']
y = data['sentiment']

# 分词函数
def tokenize(text):
    return list(jieba.cut(text))

# 定义Field和LabelField
comment_field = torchtext.legacy.data.Field(sequential=True, tokenize=tokenize, lower=True)
label_field = torchtext.legacy.data.LabelField(dtype=torch.float)

# 创建TabularDataset
fields = [('comment', comment_field), ('sentiment', label_field)]
examples = [torchtext.legacy.data.Example.fromlist([str(X[i]), y[i]], fields=fields) for i in range(data.shape[0])]

dataset = torchtext.legacy.data.Dataset(examples, fields)

# 划分训练集和测试集
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.getstate())

# 构建词汇表
comment_field.build_vocab(train_data, min_freq=2)
label_field.build_vocab(train_data)

# 创建迭代器
BATCH_SIZE = 16  # 减小批大小以减少显存使用量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = torchtext.legacy.data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=lambda x: len(x.comment),
    sort_within_batch=False
)

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# 设置模型参数
vocab_size = len(comment_field.vocab)
embedding_dim = 100
hidden_dim = 128  # 减小隐藏层维度以减少显存使用量
output_dim = 3  # 三分类任务

# 创建模型
model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    for batch in iterator:
        optimizer.zero_grad()
        text = batch.comment
        labels = batch.sentiment.long()  # 将标签转换为整数类型
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = accuracy_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy(), average='macro')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

# 测试模型
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    with torch.no_grad():
        for batch in iterator:
            text = batch.comment
            labels = batch.sentiment.long()  # 将标签转换为整数类型
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            acc = accuracy_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy(), average='macro')
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

# 训练模型
N_EPOCHS = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
train_f1s = []
valid_losses = []
valid_accs = []
valid_f1s = []

for epoch in range(N_EPOCHS):
    train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_f1 = evaluate(model, test_iterator, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    valid_f1s.append(valid_f1)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'rnn_model.pt')

    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val. Loss: {valid_loss:.4f}, Val. Acc: {valid_acc:.4f}, Val. F1: {valid_f1:.4f}')

# 加载最佳模型
model.load_state_dict(torch.load('rnn_model.pt'))

# 在测试集上评估模型
test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

# 绘制损失和准确率曲线
epochs = range(1, N_EPOCHS + 1)

plt.plot(epochs, train_losses, 'b', label='Train Loss')
plt.plot(epochs, valid_losses, 'g', label='Val. Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, train_accs, 'b', label='Train Acc')
plt.plot(epochs, valid_accs, 'g', label='Val. Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, train_f1s, 'b', label='Train F1')
plt.plot(epochs, valid_f1s, 'g', label='Val. F1')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
