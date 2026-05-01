import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from datetime import datetime

# ================================================================
# 【TXT日志】第一部分：创建评估日志文件
# ================================================================
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f'eval_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

def log(msg):
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + '\n')

log('='*50)
log(f'评估开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('='*50)
# ================================================================


# -------- 1. 加载测试数据 --------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# -------- 2. 加载模型 --------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
net.load_state_dict(torch.load('best_model.pth', map_location=device))
net.eval()


# -------- 3. 收集所有预测结果 --------
all_labels   = []   # 真实标签
all_preds    = []   # 预测类别
all_probs    = []   # 每个类别的原始得分（用于ROC）

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        # softmax 把原始得分转换成概率（加起来等于1）
        probs = F.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)


# -------- 4. 计算并打印每类准确率 --------
log('\n各类别准确率：')
log('-'*30)
for i, cls in enumerate(classes):
    mask = (all_labels == i)
    acc  = 100 * np.sum(all_preds[mask] == all_labels[mask]) / np.sum(mask)
    log(f'  {cls:10s}：{acc:.1f}%')

overall_acc = 100 * np.sum(all_preds == all_labels) / len(all_labels)
# ================================================================
# 【TXT日志】第二部分：记录总体准确率
# ================================================================
log('-'*30)
log(f'  总体准确率：{overall_acc:.2f}%')
# ================================================================


# -------- 5. 绘制混淆矩阵 --------
cm = confusion_matrix(all_labels, all_preds)

# ================================================================
# 【TXT日志】第三部分：把混淆矩阵的数字也存进txt
# ================================================================
log('\n混淆矩阵（行=真实类别，列=预测类别）：')
header = '        ' + ''.join(f'{c:>8}' for c in classes)
log(header)
for i, row in enumerate(cm):
    row_str = f'{classes[i]:8s}' + ''.join(f'{v:>8}' for v in row)
    log(row_str)
# ================================================================

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix - CIFAR-10', fontsize=14)
plt.tight_layout()
plt.savefig('./logs/confusion_matrix.png', dpi=150)
log('\n混淆矩阵图已保存：./logs/confusion_matrix.png')
plt.show()


# -------- 6. 绘制 ROC 曲线 --------
# 把标签转成 one-hot 格式（sklearn 的 roc_curve 需要）
all_labels_bin = label_binarize(all_labels, classes=list(range(10)))

fig, ax = plt.subplots(figsize=(10, 8))

# ================================================================
# 【TXT日志】第四部分：记录每个类别的 AUC 值
# ================================================================
log('\n各类别 AUC 值：')
log('-'*30)
# ================================================================

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
    # ================================================================
    # 【TXT日志】第四部分（续）：逐类写入AUC
    # ================================================================
    log(f'  {cls:10s}：AUC = {roc_auc:.4f}')
    # ================================================================

ax.plot([0,1],[0,1],'k--', label='随机猜测 (AUC=0.50)')
ax.set_xlabel('假正率 (False Positive Rate)')
ax.set_ylabel('真正率 (True Positive Rate)')
ax.set_title('ROC Curve - CIFAR-10（每类 vs 其余）', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('./logs/roc_curve.png', dpi=150)
log('\nROC曲线图已保存：./logs/roc_curve.png')
plt.show()

# ================================================================
# 【TXT日志】第五部分：评估结束
# ================================================================
log('='*50)
log(f'评估结束时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'完整日志保存于：{log_path}')
log('='*50)
# ================================================================
