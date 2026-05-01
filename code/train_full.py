import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from datetime import datetime

# ================================================================
# 【TXT日志】第一部分：创建日志文件，写入训练开始信息
# ================================================================
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

def log(msg):
    """同时打印到终端，并写入txt文件"""
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + '\n')

log('='*50)
log(f'训练开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('='*50)
# ================================================================


# -------- 1. 数据准备（加入数据增强） --------
# 训练集：加入随机翻转和随机裁剪，增加数据多样性
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),        # 随机水平翻转
    transforms.RandomCrop(32, padding=4),     # 随机裁剪（先在四周各填充4像素）
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 测试集：不做增强，只做基础处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=False, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# -------- 2. 模型定义 --------
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

# ================================================================
# 【TXT日志】第二部分：记录模型和训练配置信息
# ================================================================
log(f'\n使用设备：{device}')
log(f'模型总参数量：{sum(p.numel() for p in net.parameters()):,}')
log('超参数配置：')
log('  Epoch      = 20')
log('  Batch Size = 64')
log('  初始学习率  = 0.01')
log('  Momentum   = 0.9')
log('  数据增强    = 随机翻转 + 随机裁剪\n')
# ================================================================


# -------- 3. 损失函数、优化器、学习率调度器 --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 学习率调度：每 8 个 Epoch，学习率乘以 0.5（减半）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)


# -------- 4. 定义测试函数（方便每个Epoch后评估） --------
def evaluate():
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return 100 * correct / total


# -------- 5. 训练循环 --------
best_acc = 0.0
EPOCHS = 20

log('开始训练...')
log('-'*50)

for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每个 Epoch 结束后：计算平均Loss和测试准确率
    avg_loss = running_loss / len(trainloader)
    acc = evaluate()
    current_lr = optimizer.param_groups[0]['lr']

    # ================================================================
    # 【TXT日志】第三部分：每个 Epoch 的训练结果存档
    # ================================================================
    msg = (f'Epoch [{epoch+1:02d}/{EPOCHS}] '
           f'Loss: {avg_loss:.4f}  '
           f'测试准确率: {acc:.2f}%  '
           f'学习率: {current_lr:.6f}')
    log(msg)

    # 如果当前是最佳模型，保存权重
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), 'best_model.pth')
        # ============================================================
        # 【TXT日志】第四部分：记录最佳模型的保存事件
        # ============================================================
        log(f'  ★ 新最佳模型已保存！准确率：{best_acc:.2f}%')
        # ============================================================

    # 更新学习率
    scheduler.step()

# ================================================================
# 【TXT日志】第五部分：训练结束总结
# ================================================================
log('-'*50)
log(f'训练结束！最佳测试准确率：{best_acc:.2f}%')
log(f'结束时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'日志已保存到：{log_path}')
# ================================================================
