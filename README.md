## 项目介绍
https://tianchi.aliyun.com/competition/entrance/532073

## 软硬件环境

PyTorch  1.11.0

Python  3.8(ubuntu20.04)

Cuda  11.3

RTX 3090(24GB) * 1

## 部署与运行

### 部署
```
unzip AREA.zip

cd AREA

pip install -r requirements.txt

mkdir -p ./src/raw_data/ecom-social/

mkdir -p ./src/data/ecom-social/

mkdir -p ./src/save/ecom-social/
```

将初赛和复赛的文件都放到 ./src/raw_data/ecom-social/

这些文件如下：
```
event_info.json
source_event_preliminary_train_info.json
target_event_final_test_info.json
target_event_final_train_info.json
target_event_preliminary_test_info.json
target_event_preliminary_train_info.json
user_info.json
```

### 运行代码：
```
cd src

python process_data2.py

bash run.sh
# 程序大概会运行 2.5 个小时...

cd ..
```

### 程序运行结束
结果文件会保存在 ./submit/, 文件名为submit.json

## 关于结果的随机性
结果会在一定范围里波动，请一直执行run.sh，直到复现最优成绩即可。
