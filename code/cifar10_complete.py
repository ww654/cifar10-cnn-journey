import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from datetime import datetime

# ================================================================
# 【TXT日志】初始化：创建日志文件
# ================================================================
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f'complete_run_{timestamp}.txt')

def log(msg):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log('=' * 60)
log(f'  CIFAR-10 CNN 完整训练与评估')
log(f'  开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('=' * 60)
# ================================================================

# -------- 全局配置 --------
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 0.01
CLASSES    = ('plane','car','bird','cat','deer',
              'dog','frog','horse','ship','truck')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# 【TXT日志】记录运行配置
# ================================================================
log(f'\n运行设备：{DEVICE}')
log(f'Epochs={EPOCHS}  BatchSize={BATCH_SIZE}  LR={LR}\n')
# ================================================================

# -------- 1. 数据准备 --------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

log(f'训练集大小：{len(trainset)} 张')
log(f'测试集大小：{len(testset)} 张\n')

# -------- 2. 模型定义 --------
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 Sequential 让结构更清晰
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),       # 卷积1：32x32 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 池化1：28x28 -> 14x14

            nn.Conv2d(6, 16, 5),      # 卷积2：14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # 池化2：10x10 -> 5x5

            nn.Flatten(),             # 展平：16*5*5 = 400

            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),        # 输出10类
        )

    def forward(self, x):
        return self.net(x)

model = CIFAR10_CNN().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
log(f'模型总参数量：{total_params:,}\n')

# -------- 3. 损失函数、优化器、学习率调度 --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# -------- 4. 评估函数 --------
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

# -------- 5. 训练循环 --------
log('开始训练...')
log('-' * 60)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss   = running_loss / len(trainloader)
    acc        = evaluate()
    current_lr = optimizer.param_groups[0]['lr']

    # ================================================================
    # 【TXT日志】每轮训练结果
    # ================================================================
    msg = (f'Epoch [{epoch+1:02d}/{EPOCHS}]  '
           f'Loss: {avg_loss:.4f}  '
           f'测试准确率: {acc:.2f}%  '
           f'学习率: {current_lr:.6f}')
    log(msg)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        log(f'  ★ 新最佳模型已保存！准确率：{best_acc:.2f}%')
    # ================================================================

    scheduler.step()

log('-' * 60)
log(f'训练结束！最佳准确率：{best_acc:.2f}%\n')

# -------- 6. 加载最佳模型进行完整评估 --------
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs   = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

# ================================================================
# 【TXT日志】分类报告（精确率、召回率、F1）
# ================================================================
report = classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=4)
log('\n分类报告：')
log(report)
# ================================================================

# -------- 7. 绘制并保存混淆矩阵 --------
cm = confusion_matrix(all_labels, all_preds)

# ================================================================
# 【TXT日志】混淆矩阵数字版
# ================================================================
log('混淆矩阵（行=真实，列=预测）：')
header = '        ' + ''.join(f'{c:>8}' for c in CLASSES)
log(header)
for i, row in enumerate(cm):
    log(f'{CLASSES[i]:8s}' + ''.join(f'{v:>8}' for v in row))
log('')
# ================================================================

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
    ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix - CIFAR-10', fontsize=14)
plt.tight_layout()
plt.savefig(f'./logs/confusion_matrix_{timestamp}.png', dpi=150)
log('混淆矩阵图已保存')
plt.show()

# -------- 8. 绘制并保存 ROC 曲线 --------
all_labels_bin = label_binarize(all_labels, classes=list(range(10)))
fig, ax = plt.subplots(figsize=(10, 8))

log('各类别 AUC 值：')
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
    # ================================================================
    # 【TXT日志】AUC 值
    # ================================================================
    log(f'  {cls:10s}：AUC = {roc_auc:.4f}')
    # ================================================================

ax.plot([0,1],[0,1],'k--', label='随机猜测 (AUC=0.50)')
ax.set_xlabel('假正率')
ax.set_ylabel('真正率')
ax.set_title('ROC Curve - CIFAR-10', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'./logs/roc_curve_{timestamp}.png', dpi=150)
log('\nROC 曲线图已保存')
plt.show()

# ================================================================
# 【TXT日志】收尾
# ================================================================
log('=' * 60)
log(f'全部完成！结束时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'日志文件：{log_path}')
log('=' * 60)
# ================================================================
