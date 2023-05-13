# LineaRE

## 软硬件环境

PyTorch  1.11.0

Python  3.8(ubuntu20.04)

Cuda  11.3

RTX A5000(24GB) * 1

## 安装与运行
```bash
pip install -r requirements.txt

mkdir -p ./raw_data/ecom-social/

mkdir -p ./data/ecom-social/

mkdir -p ./save/ecom-social/
```

put raw data into ./raw_data/ecom-social/

```
python process_data2.py

bash run.sh

# wait about 7 hours...

python merge_all.py
```

the result file is local at ./save/ecom-social/, it's name is submit.json
