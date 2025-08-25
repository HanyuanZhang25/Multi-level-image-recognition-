import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 原始数据路径：每个子文件夹是一个类
input_root = r"E:\实习工作项目2\oil\total_data\data_class5" # 例如 'data/'
output_root = r"E:\实习工作项目2\oil\total_data\data_class5\after_augmentation"  # 增强后保存路径

# 定义增强操作（可以自行添加更多）
augmentations = {
    'hflip': transforms.RandomHorizontalFlip(p=1.0),
    'vflip': transforms.RandomVerticalFlip(p=1.0),
    'rotate': transforms.RandomRotation(degrees=15),
    'left_rotate': transforms.RandomRotation(degrees=90),
    'right_rotate':transforms.RandomRotation(degrees=270)

}

# 遍历每个类的文件夹
for class_name in os.listdir(input_root):
    input_folder = os.path.join(input_root, class_name)
    output_folder = os.path.join(output_root, class_name)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc=f'Processing {class_name}'):
        if not img_name.endswith('.png'):
            continue
        img_path = os.path.join(input_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # 保存原图
        image.save(os.path.join(output_folder, img_name))

        # 应用每种增强方式
        for key, aug in augmentations.items():
            aug_img = aug(image)
            new_name = f"{os.path.splitext(img_name)[0]}_{key}.png"
            aug_img.save(os.path.join(output_folder, new_name))
