# 农作物生长情况识别 Identification-of-crop-growth
讯飞2021AI开发者大赛-农作物生长情况识别赛道 第四名
最佳单模型，*Test Accuracy:0.90427*

## ATTENTION ： 
仓库只包含最佳单模型，队伍最佳分数由多个不同训练策略的swin融合所得，精度提升1%。
CV刚入门，所以方法比较简单

# 训练策略
数据：训练集标签清洗（提升5%）测试集去除黑边（提升0.1%）

模型：Swin Transformer Base 384

数据增强：一些常规的翻转，加噪，亮度增强等；训练集size x 1.15 + randomcrop，验证集/测试集size x 1.35 + centercrop；mixup + cutmix

### 由于举办方要求，仓库不提供数据集下载，[官方网站](http://challenge.xfyun.cn/topic/info?type=crop)

## 运行
```bash
python main.py --path (your data root)
