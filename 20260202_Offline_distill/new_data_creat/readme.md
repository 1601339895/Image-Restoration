**此文件夹旨在说明识别添加新数据的步骤**

**数据上传**
```
cd new_data_creat
python img2nori.py
```
**提前准备数据及文件夹**
```
---- helmet_train
├── diku/
│   ├── 1-male/（id名称）
│   │   ├── 1-male_001.jpg          # 
│   │   ├── 1-male_helmet_002.jpg   # 
│   │   ├── 1-male_mask_003.jpg     # 
│   │   └── xxx.jpg 
│   ├── 12-female/
│   │   ├── 2-female_001.jpg
│   │   ├── 2-female_helmet_002.jpg
│   │   ├── 2-female_mask_003.jpg
│   │   └── xxxx.jpg
│   └── ...（其他ID）
├── jiesuo/
│   ├── 1-male/
│   │   ├── 1-male_001.jpg
│   │   ├── 1-male_helmet_002.jpg
│   │   ├── 1-male_mask_003.jpg
│   │   └── xxx.jpg
│   ├── 2-female/
│   │   ├── 2-female_001.jpg
│   │   ├── 2-female_helmet_002.jpg
│   │   ├── 2-female_mask_003.jpg
│   │   └── xxx.jpg
│   └── ...（其他ID）

```
**验证数据**
```
python eval_data.py
```

**数据生成**

最后将生在s3上的yaml文件替换或者加入自己本地的config文件夹里