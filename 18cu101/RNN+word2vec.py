import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import random
import torchtext.legacy.data as data
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# Set the random seed
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# Load data
data_df = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset-4movies.xls', index_col=0)

# 重新加载数据集，确保设置了index_col=0，并跳过出现问题的行
data_df = pd.read_excel(r'F:\master_degree\7150CEM\douban_dataset\dataset-4movies.xls', index_col=0, errors='coerce')

# 将评分映射为情感标签
data_df['sentiment'] = data_df['rating'].apply(lambda x: 1 if x > 3 else (0 if x == 3 else -1))

X = data_df['comment']
y = data_df['sentiment']

# Tokenization function
def tokenize(text):
    return list(jieba.cut(text))

# Define Field and LabelField
comment_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
label_field = data.LabelField(dtype=torch.long)

# Create TabularDataset
fields = [('comment', comment_field), ('sentiment', label_field)]
examples = [data.Example.fromlist([str(X[i]), y[i]], fields=fields) for i in range(data_df.shape[0])]

dataset = data.Dataset(examples, fields=fields)

# Split train and test datasets
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.getstate())

#

# Build vocabulary
comment_field.build_vocab(train_data, min_freq=2)
label_field.build_vocab(train_data)

# Load pre-trained word2vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format(r'F:\pycharm_project\7071\7150CEM\model\wiki_word2vec_50.bin', binary=True)

# Create an embedding matrix for the vocabulary
embedding_dim = 50
vocab_size = len(comment_field.vocab)
embedding_matrix = torch.zeros((vocab_size, embedding_dim))

for i, word in enumerate(comment_field.vocab.itos):
    if word in word2vec_model:
        embedding_matrix[i] = torch.tensor(word2vec_model[word])
    else:
        embedding_matrix[i] = torch.rand(embedding_dim)

# Create iterators
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=lambda x: len(x.comment),
    sort_within_batch=True,  # Set this to True
    repeat=False,
    shuffle=True
)

# Define RNN model with pre-trained embeddings
class RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[:, -1, :])

# Set model parameters
HIDDEN_DIM = 256
OUTPUT_DIM = 3

# Create the model
model = RNN(embedding_matrix, HIDDEN_DIM, OUTPUT_DIM)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    for batch in iterator:
        optimizer.zero_grad()
        text = batch.comment
        labels = batch.sentiment
        predictions = model(text)
        loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

        # 将labels转换成一维张量
        labels = labels.view(-1)

        acc = accuracy_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy(), average='macro')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


# Evaluate the model on the test set
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    with torch.no_grad():
        for batch in iterator:
            text = batch.comment
            labels = batch.sentiment
            predictions = model(text)
            loss = criterion(predictions.view(-1, predictions.shape[-1]),
                             labels.view(-1))

            # 将labels转换成一维张量
            labels = labels.view(-1)

            acc = accuracy_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), torch.argmax(predictions, dim=1).cpu().numpy(), average='macro')
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


# Train the model
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

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_f1:.4f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. F1: {valid_f1:.4f}')

    # 保存效果最好的模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')

# 绘制损失和准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_accs, label='Train Acc')
plt.plot(valid_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_f1s, label='Train F1')
plt.plot(valid_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Training and Validation F1 Score')
plt.show()
