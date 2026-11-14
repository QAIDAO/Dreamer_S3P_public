# 条件分子生成训练

代码位于`src/s3p_repro/training.py`，CLI为`train-generator`。它从公开GPT-2基座开始监督微调，并保存模型与分词器供`generation.py`加载。不是仅提供伪代码或远端任务配置。

## 数据和参数

`data/generation/chebi20/train.txt`用于训练。`validation.txt`与`test.txt`作为独立保留集，本训练脚本不将它们并入训练。上游来源与SHA-256见`upstream.json`，原始CID和分子描述是公开数据字段。

| 参数 | 正式配置 |
|---|---:|
| 基座 | GPT-2 |
| 轮数 | 10 |
| 每设备批大小 | 16 |
| 梯度累积 | 2 |
| 学习率 | 5×10⁻⁵ |
| 最大序列长度 | 256 |
| 随机种子 | 42 |
| 权重衰减 | 0.01 |
| 梯度裁剪 | 1.0 |

CUDA使用fp16，CPU使用float32。优化器为AdamW，学习率采用余弦衰减与warmup。正式命令首次下载GPT-2公开基座，不需要本项目的商业模型账号；网络或算力不可用时命令会报错，不回退到示例生成。

## 序列与损失

输入顺序为`分子描述 + "\nSMILES:" + " " + SMILES + EOS`。描述、分隔符与批次填充的标签为`-100`；只有目标SMILES和EOS参与因果语言模型损失。优先保留完整目标及分隔符，过长描述从左侧截断；目标本身超长的样本计入排除统计，不静默生成全掩码样本。

输出的`training_summary.json`记录源文件校验值、选中/实际使用的样本数、排除数、训练步数、配置、平均损失与设备类型，不保存用户目录或连接信息。

## 小型检查与正式训练

```bash
python scripts/s3p.py train-generator --tiny --steps 2 --limit 32 --device cpu --output outputs/tiny-generator
python scripts/s3p.py train-generator --output outputs/conditional-gpt2
```

`--tiny`使用随机初始化的2层、2注意力头、64维小型GPT-2及本地训练分词器，验证掩码、反向传播、保存与加载；其结果不代表论文模型的化学生成能力。`--steps`或`--limit`会把产物标记为部分训练。已有模型输出目录不会被覆盖。

正式训练可在有GPU的计算节点上执行同一命令，再将整个模型目录传回研究环境。代码不附带Kaggle账号、自动上传或商业平台连接实现。

## 推理

```bash
python scripts/s3p.py generate --model outputs/conditional-gpt2 --output outputs/generated
```

提示偏好为偶氮苯骨架和对位硝基，以`SMILES:`结束；20条、温度0.8、top-p 0.96、总长度64。这里的`max_length`包含提示tokens，与`max_new_tokens`不同。模型输出先作为字符串保存，随后由RDKit检查；不会自动把已知候选补入不足的输出。达到最大批次数或连续无新输出时记录提前停止原因。
