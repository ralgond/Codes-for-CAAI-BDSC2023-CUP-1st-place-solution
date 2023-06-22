# LineaRE

## 软硬件环境

PyTorch  1.11.0

Python  3.8(ubuntu20.04)

Cuda  11.3

RTX 3090(24GB) * 1

## 安装与运行
```bash
unzip LineaRE-final.zip

cd LineaRE-final

pip install -r requirements.txt

mkdir -p ./raw_data/ecom-social/

mkdir -p ./data/ecom-social/

mkdir -p ./save/ecom-social/
```

Put raw data into ./raw_data/ecom-social/

raw data:
```
event_info.json
source_event_preliminary_train_info.json
target_event_final_test_info.json
target_event_final_train_info.json
target_event_preliminary_test_info.json
target_event_preliminary_train_info.json
user_info.json
```

```
python process_data2.py

run.sh

# wait about 2.5 hours...
```

The result file is located in ./save/ecom-social/, it's name is submit.json

## 关于结果的随机性
结果会在一定范围里波动，请一直执行run.sh，直到复现最优成绩即可。
