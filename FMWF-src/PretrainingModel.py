import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def init_seed(fix_seed):
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
fix_seed = 18
init_seed(fix_seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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


transfer_df = pd.read_csv('/synthesis-tordataset.csv', header=None)
print(transfer_df.shape)
transfer_data = transfer_df.iloc[:, 100:].values
transfer_labels = transfer_df.iloc[:, 0:100].values  # multi-hot labels

transfer_data = torch.tensor(transfer_data, dtype=torch.float32).unsqueeze(1)
transfer_labels = torch.tensor(transfer_labels, dtype=torch.float32)
print("transfer_data.shape:", transfer_data.shape)
print("transfer_labels.shape:", transfer_labels.shape)


dataset = TensorDataset(transfer_data, transfer_labels)


train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = CNNmodel(num_classes=100).to(device)
criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model.train()
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_targets = []
    all_outputs = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets.append(target.detach().cpu().numpy())
        all_outputs.append(output.detach().cpu().numpy())


    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)


    binarized_outputs = (all_outputs > 0).astype(int)


    train_accuracy = accuracy_score(all_targets, binarized_outputs)
    train_precision = precision_score(all_targets, binarized_outputs, average='samples', zero_division=1)
    train_recall = recall_score(all_targets, binarized_outputs, average='samples', zero_division=1)
    train_f1 = f1_score(all_targets, binarized_outputs, average='samples', zero_division=1)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {total_loss / len(train_loader):.4f}, "
        f"Train Accuracy: {train_accuracy * 100:.2f}%, "
        f"Train Precision: {train_precision * 100:.2f}%, "
        f"Train Recall: {train_recall * 100:.2f}%, "
        f"Train F1 Score: {train_f1 * 100:.2f}%"
    )

torch.save(model.state_dict(), 'FMWF_Pretrainmodel_weights200epoch.pth')
# # torch.save(model, 'FMWF_Pretrainmodel200epoch.pth')
