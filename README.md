# 1. 创建并激活 Python 3.8 虚拟环境
python3.8 -m venv jt-env
source jt-env/bin/activate

# 2. 升级 pip 工具链
pip install --upgrade pip setuptools wheel

# 3. 安装 Jittor（需根据实际 CUDA 版本匹配）
# 可替换为其他版本：https://cg.cs.tsinghua.edu.cn/jittor/downloads/
pip install jittor==1.3.8.5

# 4. 安装 JTorch（Jittor 的 PyTorch API 兼容层）
pip install jtorch==0.1.7


# Step 1: 准备数据
cd data_prepare
bash create_datasets.sh


# Step 2: 下载预训练模型
cd ..
bash download_pretrain_checkpoints.sh

# Step 3: 启动微调任务
bash finetune.sh

# Step 4: 执行推理（生成样本）
bash run_inference.sh

# Step 5: 下载官方评估脚本（BLEU / METEOR / NIST 等）
cd eval
bash download_evalscript.sh
cd ..

# Step 6: 运行指标计算脚本
bash run_metric.sh

