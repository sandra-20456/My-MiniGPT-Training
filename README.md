本项目基于PyTorch从零手搭了极简的GPT架构，并使用自建的27000+字中文语料进行了本地训练。

数据验收
验证集推理误差(Val Loss):3.8510
推理吞吐能力(Tokens/sec):619.13 tokens/秒

运行环境
Python 3.12.8
PyTorch(CPU 版本)
机械革命 RTX 5070 笔记本

文件说明
'model.py':手写的Transformer架构
'train.py':数据处理、模型训练、及吞吐量压测脚本
'my_data.txt':专属训练语料