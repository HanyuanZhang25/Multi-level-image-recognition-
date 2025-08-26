import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import DatasetFolder
from torchvision.models import resnet18

#新添加的类，作用：将我们生成的经过筛选后所生成的旧数据列表old_samples由list形式变为dataset形式，
#并重写其len方法，并为该list添加samples属性，以保证拼接后的数据集有samples属性，保证create_episode方法正常运行
class MemoryDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: 列表，每个元素为 (image_tensor, label) 元组
        transform: 可选的数据增强
        """
        self.samples = []  # 存储 (路径占位符, 标签) 的列表
        self.data = []  # 存储图像张量
        self.labels = []  # 存储标签

        # 填充数据结构
        for idx, (img_tensor, label) in enumerate(samples):
            # 为每个样本创建虚拟路径（实际不会使用）
            virtual_path = f"memory_sample_{idx}.jpg"
            self.samples.append((virtual_path, label))
            self.data.append(img_tensor)
            self.labels.append(label)

        self.transform = transform
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        # 如果需要应用额外的转换
        if self.transform:
            # 将张量转回PIL图像进行转换（如果原始transform包含图像操作）
            if isinstance(self.transform, torch.nn.Module) or callable(self.transform):
                img = Image.fromarray(img.numpy().transpose(1, 2, 0)) if img.dim() == 3 else Image.fromarray(
                    img.numpy())
                img = self.transform(img)
        return img, label

#新添加的类，作用：拼接新老数据集，保证数据集拼接后仍然有samples属性。
class CustomConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 合并所有属性
        for i, dataset in enumerate(datasets):
            self.samples.extend(dataset.samples)

            # 处理类别映射（需注意不同数据集类别冲突）
            if i == 0:
                self.classes = dataset.classes.copy()
                self.class_to_idx = dataset.class_to_idx.copy()
            else:
                # 这里简单合并，实际应用需考虑类别重映射
                self.classes.extend(dataset.classes)
                self.class_to_idx.update(dataset.class_to_idx)


# 1. 特征提取器：简单的CNN
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层

    def forward(self, x):
        x = self.backbone(x)  # (B, 512, 1, 1)
        return x.view(x.size(0), -1)  # 输出 (B, 512)


# 2. 数据预处理
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])


# 3. 构造 few-shot episode
def create_episode(dataset, n_way=4, k_shot=5, q_query=3):
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for class_id in range(n_way):
        idx = [i for i, (_, label) in enumerate(dataset.samples) if label == class_id]
        np.random.shuffle(idx)
        selected = idx[:k_shot + q_query]
        for i in range(k_shot):
            support_images.append(dataset[selected[i]][0])
            support_labels.append(class_id)
        for i in range(k_shot, k_shot + q_query):
            query_images.append(dataset[selected[i]][0])
            query_labels.append(class_id)

    return (
        torch.stack(support_images), torch.tensor(support_labels),
        torch.stack(query_images), torch.tensor(query_labels)
    )


# 4. 推理函数
def prototypical_forward(encoder, support_images, support_labels, query_images):
    support_embeddings = encoder(support_images)
    query_embeddings = encoder(query_images)
    prototypes = []

    for cls in torch.unique(support_labels):
        cls_emb = support_embeddings[support_labels == cls]
        prototypes.append(cls_emb.mean(0))
    prototypes = torch.stack(prototypes)

    dists = torch.cdist(query_embeddings, prototypes)
    return -dists  # 越大越相似


# 5. 在线微调函数 - 核心增强部分
def online_fine_tune(encoder, old_data_path, new_data_path, device='cuda',
                     epochs=20, lr=1e-5, freeze_layers=True):
    """
    在线微调增强模型

    参数:
    encoder: 预训练好的编码器模型
    old_data_path: 原始数据路径 (用于保留旧知识)
    new_data_path: 新增数据路径
    device: 训练设备
    epochs: 微调轮数
    lr: 学习率 (建议使用较低的学习率)
    freeze_layers: 是否冻结底层网络
    """
    # 冻结底层网络
    if freeze_layers:
        for name, param in encoder.named_parameters():
            # 冻结除最后一层外的所有层
            if 'backbone.7' not in name:  # 调整这个索引根据实际网络结构
                param.requires_grad = False
        print(">> 冻结了底层网络参数")

    # 准备数据集
    transform = get_transform()

    # 加载旧数据（部分样本）
    old_dataset = datasets.ImageFolder(old_data_path, transform=transform)
    # 从每个类别中随机选择少量样本保留旧知识
    old_samples = []
    for class_id in range(len(old_dataset.classes)):
        idx = [i for i, (_, label) in enumerate(old_dataset.samples) if label == class_id]
        np.random.shuffle(idx)
        selected = idx[:10]  # 每个类别保留10个样本
        for i in selected:
            old_samples.append(old_dataset[i])

    old_dataset_improved = MemoryDataset(old_samples)

    # 加载新数据
    new_dataset = datasets.ImageFolder(new_data_path, transform=transform)

    # 合并数据集
    combined_dataset = CustomConcatDataset([old_dataset_improved, new_dataset])
    # 优化器 - 只优化未冻结的参数

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=lr
    )

    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        encoder.train()

        # 创建训练episode
        s_img, s_lbl, q_img, q_lbl = create_episode(
            combined_dataset,
            n_way=len(old_dataset.classes),  # 使用所有类别
            k_shot=5,
            q_query=3
        )
        s_img, s_lbl, q_img, q_lbl = s_img.to(device), s_lbl.to(device), q_img.to(device), q_lbl.to(device)

        # 前向传播
        logits = prototypical_forward(encoder, s_img, s_lbl, q_img)
        loss = F.cross_entropy(logits, q_lbl)
        acc = (logits.argmax(dim=1) == q_lbl).float().mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"微调 Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Acc: {acc.item():.4f}")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            torch.save(encoder.state_dict(), r"/class5_code\enhanced_model.pth")
            print(">> 保存增强模型")

    # 解冻所有层
    for param in encoder.parameters():
        param.requires_grad = True

    return encoder

# 6. 主程序入口
if __name__ == '__main__':
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载预训练模型
    encoder = Encoder().to(device)
    pretrained_path = r"/class5_code/best_class5.pth"
    encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(">> 加载预训练模型成功")

    # 定义数据集路径
    old_data_path = r"/class5_code/class5_total_data/train_data6"
    new_data_path = r"/class5_code\class5_total_data\test_data"  # 新数据路径

    # 在线微调增强模型
    print(">> 开始在线微调...")
    enhanced_encoder = online_fine_tune(
        encoder,
        old_data_path,
        new_data_path,
        device=device,
        epochs=30,  # 使用较少的epochs
        lr=1e-5,  # 使用较低的学习率
        freeze_layers=True  # 冻结底层网络
    )

    print(">> 模型增强完成! 增强模型已保存为 enhanced_model.pth")