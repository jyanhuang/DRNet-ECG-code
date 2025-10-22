import json
import os
import pandas as pd

# 定义类别映射
label_mapping = {
    "N": 0,
    "V": 1,
    "R": 2,
    "L": 3,
    "_": 4,
    "A": 5,
    "!": 6,
    "E": 7
}

# 定义数据集的根目录
root_dir = "D:\\python\\python项目\\ECG\\ecg-classification-master\\data\\1D"

# 初始化结果列表
results = []

# 遍历100到234的文件夹
for folder in range(100, 235):
    folder_path = os.path.join(root_dir, str(folder))
    if os.path.exists(folder_path):
        #print(f"检查文件夹：{folder_path}")
        # 遍历MLII文件夹
        mlii_path = os.path.join(folder_path, 'MLII')
        if os.path.exists(mlii_path):
            #print(f"找到MLII文件夹：{mlii_path}")
            # 遍历类别文件夹
            for label in label_mapping.keys():
                label_path = os.path.join(mlii_path, label)
                if os.path.exists(label_path):
                    #print(f"找到类别文件夹：{label_path}")
                    # 读取文件名
                    for filename in os.listdir(label_path):
                        file_path = os.path.join(label_path, filename)
                        # 确保是文件
                        if os.path.isfile(file_path):
                            #print(f"找到文件：{file_path}")
                            # 构建结果字典
                            result = {
                                "name": str(folder),
                                "lead": 'MLII',  # 假设MLII文件夹下的文件都属于MLII导联
                                "label": label,
                                "filename": os.path.splitext(filename)[0],  # 只保存文件名，不包括扩展名
                                "path": file_path
                            }
                            results.append(result)
                        #else:
                            ##print(f"{file_path} 不是文件")
                #else:
                    #print(f"未找到类别文件夹：{label_path}")
        #else:
            #print(f"未找到MLII文件夹：{mlii_path}")
    #else:
       # print(f"未找到文件夹：{folder_path}")

# 将结果列表转换为DataFrame
data = pd.DataFrame(results)

# 设置随机种子
random_state = 7

# 对数据进行随机抽样
data = data.sample(frac=1, random_state=random_state)

# 定义验证集大小
val_size = 0.1

# 划分验证集
val_ids = []
for cl in label_mapping.keys():
    val_ids.extend(
        data[data["label"] == cl]
        .sample(frac=val_size, random_state=random_state)
        .index,
    )

val = data.loc[val_ids, :]
train = data[~data.index.isin(val.index)]

# 保存训练集和验证集为JSON文件
train.to_json(os.path.join(root_dir, "train.json"), orient="records")
val.to_json(os.path.join(root_dir, "val.json"), orient="records")

# 创建类别映射器
class_mapper = {}
for label in train.label.unique():
    class_mapper[label] = len(class_mapper)

# 保存类别映射器为JSON文件
with open(os.path.join(root_dir, "class-mapper.json"), "w") as file:
    file.write(json.dumps(class_mapper, indent=1))

#print("数据已成功保存到JSON文件中。")