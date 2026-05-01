import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime


# 【TXT日志】初始化
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f'transfer_run_{timestamp}.txt')

def log(msg):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log('=' * 60)
log('  CIFAR-10 迁移学习（ResNet18 预训练）')
log(f'  开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('=' * 60)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# -------- 1. 数据准备 --------
# 注意：迁移学习要用 ImageNet 的均值和标准差来归一化
# 因为预训练模型是用 ImageNet 训练的，输入分布必须匹配
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize(224),               # ResNet18 期望 224x224 的输入
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=28),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,  download=False, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True,  num_workers=0)
testloader  = torch.utils.data.DataLoader(
    testset,  batch_size=64, shuffle=False, num_workers=0)

log(f'\n运行设备：{DEVICE}')
log(f'训练集：{len(trainset)} 张，测试集：{len(testset)} 张\n')

# 2. 加载预训练模型
# 【关键步骤】从 torchvision 加载 ResNet18，weights='DEFAULT' 代表
# 加载在 ImageNet 上训练好的权重
model = models.resnet18(weights='DEFAULT')


# 【TXT日志】记录原始模型结构
log(f'原始 ResNet18 最后一层：{model.fc}')
# 输出类似：Linear(in_features=512, out_features=1000, bias=True)
# 1000 是 ImageNet 的类别数，我们要换成 10

# 3. 替换分类头
# 获取最后全连接层的输入维度（ResNet18 是 512）
in_features = model.fc.in_features
# 换成输出 10 类的新层（权重是随机初始化的）
model.fc = nn.Linear(in_features, 10)
model = model.to(DEVICE)

log(f'替换后分类头：{model.fc}')
log(f'模型总参数量：{sum(p.numel() for p in model.parameters()):,}\n')


# 4. 第一阶段：冻结卷积层，只训练分类头 
log('=' * 60)
log('第一阶段：冻结卷积层，只训练分类头')
log('=' * 60)


# 【冻结操作】把所有参数的 requires_grad 设为 False
# requires_grad=False 意味着这个参数不参与反向传播，不会被更新
for param in model.parameters():
    param.requires_grad = False

# 只把最后一层（分类头）解冻
for param in model.fc.parameters():
    param.requires_grad = True

# 只优化 requires_grad=True 的参数
optimizer_stage1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
criterion = nn.CrossEntropyLoss()

def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, predicted = torch.max(model(inputs), 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

# 第一阶段训练 5 个 epoch（分类头很快收敛）
STAGE1_EPOCHS = 5
best_acc = 0.0
for epoch in range(STAGE1_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_stage1.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer_stage1.step()
        running_loss += loss.item()

    acc = evaluate()
    msg = (f'[阶段1] Epoch [{epoch+1}/{STAGE1_EPOCHS}]  '
           f'Loss: {running_loss/len(trainloader):.4f}  '
           f'准确率: {acc:.2f}%')
    log(msg)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'transfer_best.pth')
        log(f'  ★ 阶段1最佳：{best_acc:.2f}%')

log(f'\n阶段1结束，最佳准确率：{best_acc:.2f}%\n')

# 5. 第二阶段：解冻全部层，用小学习率微调 
log('=' * 60)
log('第二阶段：解冻所有层，整体微调')
log('=' * 60)


# 【解冻操作】把所有参数重新设回 requires_grad=True
for param in model.parameters():
    param.requires_grad = True

# 用比第一阶段小 10 倍的学习率
optimizer_stage2 = optim.SGD(
    model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_stage2, T_max=10
)

STAGE2_EPOCHS = 10
for epoch in range(STAGE2_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer_stage2.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer_stage2.step()
        running_loss += loss.item()

    acc = evaluate()
    current_lr = optimizer_stage2.param_groups[0]['lr']
    msg = (f'[阶段2] Epoch [{epoch+1}/{STAGE2_EPOCHS}]  '
           f'Loss: {running_loss/len(trainloader):.4f}  '
           f'准确率: {acc:.2f}%  '
           f'学习率: {current_lr:.6f}')
    log(msg)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'transfer_best.pth')
        log(f'  ★ 新最佳：{best_acc:.2f}%')

    scheduler.step()

# 【TXT日志】最终结果
log('=' * 60)
log(f'迁移学习完成！最终最佳准确率：{best_acc:.2f}%')
log(f'结束时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'日志：{log_path}')
log('=' * 60)
