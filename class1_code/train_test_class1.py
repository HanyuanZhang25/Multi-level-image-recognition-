import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18

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
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(r"/class1_code/train_data1", transform=transform)
class_names = dataset.classes  # 获取类名
print("Classes:", class_names)  # 应为6类

# 3. 构造 few-shot episode
def create_episode(dataset, n_way=6, k_shot=5, q_query=3):
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

# 5. 训练与保存最优模型
def train(encoder, dataset, device='cpu', n_way=6, k_shot=5, q_query=3):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    best_loss = float('inf')
    for epoch in range(200):
        encoder.train()
        s_img, s_lbl, q_img, q_lbl = create_episode(dataset, n_way, k_shot, q_query)
        s_img, s_lbl, q_img, q_lbl = s_img.to(device), s_lbl.to(device), q_img.to(device), q_lbl.to(device)


        logits = prototypical_forward(encoder, s_img, s_lbl, q_img)
        loss = F.cross_entropy(logits, q_lbl)
        acc = (logits.argmax(dim=1) == q_lbl).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f} - Acc: {acc.item():.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(encoder.state_dict(), r"/class1_code\best_class1.pth")
            print(">> Best model saved.")

# 6. 测试函数：输入PNG图片，输出分类
def predict_image(image_path, encoder_path=r"E:\实习工作项目2\oil\class1_code\best_class1.pth", class_names=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = Encoder().to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.eval()

    support_images, support_labels = [], []
    k_shot = 5
    for label_id, class_name in enumerate(class_names):
        class_dir = os.path.join(r'/class1_code\test数据', class_name)
        image_names = sorted(os.listdir(class_dir))[:k_shot]
        for img_name in image_names:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            support_images.append(img_tensor)
            support_labels.append(label_id)

    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)

    with torch.no_grad():
        query_embedding = model(image_tensor)
        support_embeddings = model(support_images)

        prototypes = []
        for cls in torch.unique(support_labels):
            cls_emb = support_embeddings[support_labels == cls]
            prototypes.append(cls_emb.mean(0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(query_embedding, prototypes)
        pred = dists.argmin().item()

    return class_names[pred]

# 7. 主程序入口
if __name__ == '__main__':

#若要训练请将该多行注释部分的注释的6个引号删除，并将测试演示部分用注释符号遮盖
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder().to(device)

    print(">> 开始训练...")
    train(encoder, dataset, device=device, n_way=13)

'''

    #测试演示
    correct_num = 0
    total_num = 0

    dir_path = r"/class1_code\test数据"
    file1 = os.listdir(dir_path)
    # 解释：两个for循环，第一个file1是存放所有内部类别的总file，第二个for是遍历每个内部类别的图片

    for file_inside in file1:
        class_real_name = os.path.basename(file_inside)
        file_inside_all = os.listdir(dir_path + "\\" + file_inside)
        # file_inside_all是一个list，里面有该文件的所有图片
        # file_inside则是遍历出来的文件名，我们需要他来打开每个内部类别文件里的图片
        file_inside_name = os.path.abspath(file_inside)
        print(file_inside_name)
        for image in file_inside_all:
            total_num += 1
            print(image)
            print(file_inside)
            test_image_path = dir_path + "\\" + file_inside + "\\" + image  # 打开内部类别文件

            print(test_image_path)
            # test_image_path = r"C:\\Users\lenovo\Desktop\实习工作项目\训练与测试数据\训练数据\QPW型万象接头\\0d4a250bdb501edb08a62423c39efa3.png"  # 替换为你自己的测试图片路径
            result = predict_image(test_image_path, class_names=class_names)
            print("预测类别：", result)
            str(result)
            if result == class_real_name:
                correct_num += 1

    acc = correct_num / total_num
    print(f"准确率是{acc}")
