import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def init_seed(fix_seed):
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)

fix_seed = 18
init_seed(fix_seed)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, dropout_p, activation):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation(inplace=True),
            nn.MaxPool1d(pool_size, pool_stride, padding=0),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)


class CNNmodel(nn.Module):
    def __init__(self, num_classes):
        super(CNNmodel, self).__init__()
        kernel_size = 8
        conv_stride_size = 1
        pool_stride_size = 4
        pool_size = 8
        length_after_extraction = 57

        self.CNNnet = nn.Sequential(
            ConvBlock(1, 32, kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ELU),
            ConvBlock(32, 64, kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU),
            ConvBlock(64, 128, kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU),
            ConvBlock(128, 256, kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU),
            nn.Flatten(),
            nn.Linear(256 * length_after_extraction, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.CNNnet(x)
        return x

# 设置参数
num_classes = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并移动到设备
model = CNNmodel(num_classes).to(device)
model.load_state_dict(torch.load('FMWF_Pretrainmodel_weights200epoch.pth.pth'))

# 加载并处理fine-tuning数据集
fine_tune_df = pd.read_csv('/tor-5tab-finetuning100way40shot.csv', header=None)
fine_tune_data = fine_tune_df.iloc[:, 106:].values
fine_tune_labels = fine_tune_df.iloc[:, 0:100].values  # multi-hot labels

fine_tune_data = torch.tensor(fine_tune_data, dtype=torch.float32).unsqueeze(1)
fine_tune_labels = torch.tensor(fine_tune_labels, dtype=torch.float32)

# 打印fine-tuning数据集的形状
print("fine_tune_data.shape:", fine_tune_data.shape)
print("fine_tune_labels.shape:", fine_tune_labels.shape)

fine_tune_dataset = TensorDataset(fine_tune_data, fine_tune_labels)
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)

# 加载并处理验证数据集
val_df = pd.read_csv('/tor5tab-dataset-val.csv', header=None)
val_data = val_df.iloc[:, 106:].values
val_labels = val_df.iloc[:, 0:100].values  # multi-hot labels

val_data = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
val_labels = torch.tensor(val_labels, dtype=torch.float32)

val_dataset = TensorDataset(val_data, val_labels)
validate_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型训练
model.train()
criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 300
validate_epochs = list(range(40, 301))

# 初始化保存验证结果的列表
validation_results = {
    "epoch": [],
    "Accuracy": [],
}

for epoch in range(num_epochs):
    total_loss = 0
    total = 0
    HamAcc = {label: 0 for label in range(1, 3)}
    for data, target in fine_tune_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Tuning Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(fine_tune_loader):.4f}")


    if (epoch + 1) in validate_epochs:
        model.eval()
        val_loss = 0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for data, target in validate_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())

        # 转换为numpy数组
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)

        binarized_outputs = np.zeros_like(all_outputs)
        topk_indices = np.argsort(all_outputs, axis=1)[:, -2:]
        for i, indices in enumerate(topk_indices):
            binarized_outputs[i, indices] = 1



        val_accuracy = accuracy_score(all_targets, binarized_outputs)

        print(
            f"Validation at Epoch {epoch + 1}: "
            f"Loss: {val_loss / len(validate_loader):.4f}, "
            f"Accuracy: {val_accuracy * 100:.2f}%, "
        )


        validation_results["epoch"].append(epoch + 1)
        validation_results["Accuracy"].append(val_accuracy)

        model.train()


df_results = pd.DataFrame(validation_results)
df_results.to_csv('100way-40shot-validation_accuracy-300epoch.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(validation_results["epoch"], validation_results["Accuracy"], label='Accuracy', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Metrics over Epochs')
plt.legend()
plt.grid()
plt.show()