import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from train_test_class6 import train
from train_test_class6 import Encoder
from train_test_class6 import transform
import torch
from torchvision import datasets

app = FastAPI()

@app.post("/incremental-train1/")
async def incremental_train():
    root_folder = "train_data6"
    added_image_count = 0

    # Step 1: 调用外部接口获取JSON数据
    try:
        resp = requests.get("http://10.204.20.100/img/recognition/pdToday")  # ← 替换成实际HOST
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败：{e}")

    # Step 2: 处理下载逻辑
    for item in data:
        name = item.get("name")
        urls = item.get("url", "").split(",")

        folder_path = os.path.join(root_folder, name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"创建目录：{folder_path}")

        for single_url in urls:
            single_url = single_url.strip()
            if not single_url:
                continue

            try:
                response = requests.get(single_url)
                if response.status_code == 200:
                    filename = f"{uuid4().hex}.jpg"
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"保存图片到: {file_path}")
                    added_image_count += 1
                else:
                    print(f"下载失败（{single_url}），状态码：{response.status_code}")
            except Exception as e:
                print(f"下载图片（{single_url}）时出错：{e}")

    # Step 3: 若有新增图片，重新训练模型
    if added_image_count > 0:
        try:
            model_path = r"D:\oil\best_class6.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"已删除旧模型：{model_path}")
            else:
                print(f"模型文件不存在：{model_path}")

            print("开始重新训练模型...")
            dataset = datasets.ImageFolder(r"D:/oil/class5_code/class5_total_data/train_data6", transform=transform)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            train(Encoder, dataset, device=device, n_way=6)
            return {"message": "训练完成", "added_images": added_image_count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"训练失败：{e}")
    else:
        return {"message": "未添加图片，不需要训练", "added_images": 0}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)