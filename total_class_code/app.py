import os
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from torch import nn
from torchvision.models import resnet18
from typing import List
from train_test import predict_image
"""
后续只需添加模型权重和 class{i}_map.json 及对应支持集目录：
class0_encoder.pth
class0_map.json
data_api/class0/*
class1_encoder.pth
class1_map.json
data_api/class1/*
...
然后在代码里把 supported_classes = [1] 改为 supported_classes = [0,1,2,...]，一行搞定。
特别注意！！！python的索引是从0开始，所以第一类分类应该是class0而非class1，以此类推。
"""
# === 1. 模型定义 ===
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


# === 2. 设备与预处理 ===
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# === 3. 加载6分类模型 ===
# with open("class_map.json", "r", encoding="utf-8") as f:
#     class_map = json.load(f)
# class_names = list(class_map.values())
with open('class_map.json', 'r', encoding='utf-8') as f:
    class_map = json.load(f)

class_names = list(class_map.keys())  # ['class1', 'class2', 'class3', ...]
real_names = [class_map[c] for c in class_names]  # ['狗', '猫', '鸟', ...]


model_6class = Encoder().to(device)
model_6class.load_state_dict(torch.load("best_prototypical_encoder.pth", map_location=device))
model_6class.eval()

# === 4. 加载 class2 模型和支持集 ===
supported_classes = [1]  # 仅支持 class2 的细分类

subclass_models = {}
subclass_support_data = {}

for cls_id in supported_classes:
    # 加载模型
    model_path = f"best_class{cls_id}.pth"
    json_path = f"class_map{cls_id}.json"
    data_dir = f"data_api{cls_id}"

    sub_model = Encoder().to(device)
    sub_model.load_state_dict(torch.load(model_path, map_location=device))
    sub_model.eval()
    subclass_models[cls_id] = sub_model

    with open(json_path, "r", encoding="utf-8") as f:
        sub_class_map = json.load(f)
    sub_class_names = list(sub_class_map.values())

    support_images = []
    support_labels = []
    for label_id, sub_class in enumerate(sub_class_names):
        class_dir = os.path.join(data_dir, sub_class)
        img_files = sorted(os.listdir(class_dir))[:5]
        for img_file in img_files:
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            support_images.append(img_tensor)
            support_labels.append(label_id)

    subclass_support_data[cls_id] = (
        torch.stack(support_images).to(device),
        torch.tensor(support_labels).to(device),
        sub_class_names
    )

# === 5. FastAPI 实例 ===
app = FastAPI(title="油器件总分类API", description="6分类后再进入细分类（目前仅支持class2）", version="0.1")


@app.post("/predict/")
async def predict_all(image: UploadFile = File(...)) -> dict:
    # 将 UploadFile 保存为临时文件路径
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    # 使用主分类函数进行 6 分类预测
    pred_class = predict_image(
        temp_path,
        encoder_path='best_prototypical_encoder.pth',
        class_names=class_names
    )
    pred_class_id = class_names.index(pred_class)
    pred_class_name = class_map[pred_class]

    # 将上传图片转为张量
    img = Image.open(temp_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 不支持细分类的类直接返回主分类
    if pred_class_id not in subclass_models:
        return {
            "major_class": pred_class_name,
            "subclass_top3": [],
            "message": f"该类（{pred_class_name}）暂未支持细分类"
        }

    # --- 细分类处理 ---
    sub_model = subclass_models[pred_class_id]
    support_images, support_labels, sub_class_names = subclass_support_data[pred_class_id]

    with torch.no_grad():
        query_emb = sub_model(img_tensor)
        support_emb = sub_model(support_images)

        prototypes = []
        for cls in torch.unique(support_labels):
            cls_emb = support_emb[support_labels == cls]
            prototypes.append(cls_emb.mean(0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(query_emb, prototypes)
        probs = torch.softmax(-dists, dim=1).squeeze(0)
        top3_probs, top3_indices = torch.topk(probs, 3)

        result = []
        for i in range(3):
            result.append({
                "subclass": sub_class_names[top3_indices[i].item()],
                "probability": round(top3_probs[i].item(), 4)
            })

    return {
        "major_class": pred_class_name,
        "subclass_top3": result
    }



# === 6. 启动入口 ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
