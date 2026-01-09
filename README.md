#目录  
1.项目标题  
2.模型概述  
3.环境搭建  
4.项目结构  
5.模型训练  
6.可调参数  
7.监控学习过程  
8.输出文件说明  
9.故障排除  



# 模型标题
HEGN 事件抽取模型  

一个基于多尺度特征融合和图神经网络的金融事件抽取模型，支持10种金融事件类型的论元关系识别。

# 模型概述
本模型采用多模块架构设计，主要特点包括：  
轻量级多尺度特征融合：使用1x1和3x3卷积核提取不同尺度的特征  
Transformer编码器：捕捉长距离依赖关系  
图解码器：建模论元间的复杂关系  
自适应序列处理：支持长文本序列的分割和池化处理  
支持的事件类型：质押、股份股权转让、投资、减持、起诉、收购、判决、签署合同、担保、中标  

# 环境搭建
pip install -r requirements.txt

主要依赖：  
torch==2.9.1+cu130  
transformers==4.57.3  
ltp==4.2.14 (中文语言处理工具)  
gensim==4.4.0  
tensorboardX==2.6.4  
其中torch可根据及其自身调整版本，建议采用cuda版本，利用GPU加速训练推理过程

# 项目结构
.
├── cache/                # 缓存目录（含嵌入/训练示例缓存）  
│   ├── embedding/        # 嵌入相关缓存文件目录  
│   └── train_example.pkl # 训练示例缓存文件  
├── data/                 # 数据集&数据处理目录  
│   ├── 1train.json       # 训练数据  
│   ├── 1dev.json         # 验证数据  
│   ├── company.txt       # 公司词典  
│   ├── stopwords.txt     # 停用词词典  
│   ├── __init__.py       # 数据模块初始化文件  
│   ├── datasets.py       # 数据集加载/数据处理核心逻辑  
│   └── data_process.py   # 数据预处理脚本（清洗/格式转换等）  
├── exp/                  # 实验输出目录  
│   ├── ab_graph_16qw0/   # 实验结果子目录1（16qw0版本）  
│   └── ab_graph_160/     # 实验结果子目录2（160版本）  
├── model/                # 模型模块目录  
│   ├── __init__.py       # 模型模块初始化文件  
│   └── models.py         # 模型结构定义（网络层/损失函数等）  
├── train/                # 训练模块目录  
│   ├── __init__.py       # 训练模块初始化文件  
│   └── trainer.py        # 训练逻辑（优化器/训练循环/验证等）  
├── run.py                # 主程序入口（替代原main.py）  
├── README.md             # 项目说明文档（使用说明/实验说明等）  
└── requirements.txt      # 项目依赖列表（python包版本）  

自建文件夹（cache、exp等）均为项目运行必需的辅助目录，其中：  
cache用于存储预处理中间结果，避免重复计算，提升运行效率；  
exp用于按实验名称隔离不同训练任务的输出，方便对比不同参数 / 模型结构的效果；  
代码包（data、model、train）通过模块化设计拆分功能，便于维护和扩展。  

# 模型训练

    # 基础训练（使用默认参数）
    python main.py

    # 指定实验名称
    python main.py --exp_name my_experiment

    # 禁用特定模块进行消融实验
    python main.py --exp_name no_graph --use_graph_decoder False（消融图解码模块）
    python main.py --exp_name no_multi_scale --use_multi_scale_fusion False（消融多尺度特征融合模块）

# 可调参数

    # 数据路径
    --dataset_path ./data
    --cache_dir ./cache

    # 模型结构开关
    --use_multi_scale_fusion True    # 多尺度融合
    --use_transformer_encoder True   # Transformer编码器  
    --use_graph_decoder True         # 图解码器

    # 训练参数
    --per_gpu_train_batch_size 1     # 批大小
    --gradient_accumulation_steps 6   # 梯度累积
    --learning_rate 1e-4            # 学习率
    --num_train_epochs 160          # 训练轮数

# 监控学习过程
    # 启动tensorboard监控
    tensorboard --logdir exp/your_exp_name/tensorboard

# 输出文件说明
训练完成后会在 exp/实验名称/下生成：  
output/result.txt：详细训练日志和评估结果  
tensorboard/：可视化数据  
best_model/：最佳模型参数  
config.json：实验配置备份  

# 故障排除
常见问题  
1.CUDA内存不足  
2.LTP模型下载失败  
3.BERT模型加载慢  
解决方法：  
1.# 减小批大小或增加梯度累积步数  
python main.py --per_gpu_train_batch_size 1 --gradient_accumulation_steps 8  
2.参考https://github.com/HIT-SCIR/ltp的readme  

    # 方法 1： 使用清华源安装 LTP  
    # 1. 安装 PyTorch 和 Transformers 依赖  
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers  
    # 2. 安装 LTP  
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ltp ltp-core ltp-extension  

    # 方法 2： 先全局换源，再安装 LTP
    # 1. 全局换 TUNA 源
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    # 2. 安装 PyTorch 和 Transformers 依赖
    pip install torch transformers
    # 3. 安装 LTP
    pip install ltp ltp-core ltp-extension  
3.首次运行会自动下载BERT模型，请打开加速器保持网络连接