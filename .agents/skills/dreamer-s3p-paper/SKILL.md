---
name: dreamer-s3p-paper
description: Run the FDC2026-0145 molecular research workflow, including recorded-case checks, conditional GPT-2 training, molecular screening, three-qubit features and SVM evaluation. Use for this paper's reproduction tasks rather than general chemistry questions.
---

# 光响应分子科研流程

先读仓库根目录的`agent.md`。从本技能目录向上三级即仓库根目录；以下命令均在仓库根目录执行。具体参数见`configs/paper.json`，论文对应关系见`docs/paper_mapping.md`。

根据用户目的选择入口：

- 核对论文表4与案例规模：`python scripts/s3p.py replay-case --output outputs/replay`。该命令重算有效性、QED和SA，模型得分使用明确标记的论文快照。
- 执行新的CPU模拟流程：`python scripts/s3p.py inspect-data`，再运行`python scripts/s3p.py pipeline --output outputs/simulator`。
- 训练并重新生成：阅读`docs/training.md`。先用两步小模型检查代码；正式训练按10轮配置，明确算力和耗时授权。训练后执行`generate`，再把实际生成CSV传给`pipeline --candidates`。
- 真机执行：先运行`plan`并阅读`docs/hardware.md`，在用户明确授权后使用独立设备worker。默认不访问设备。

执行时由程序完成结构解析、QED/SA、指纹、PCA、量子特征与分类；智能体负责计划、工具组织、状态核对和解释。分隔符错误只修正并重跑生成阶段。输出检查失败时停止后续依赖；设备状态不明时保留现场并核对队列，不能自动重复提交。

交付`summary.json`、`metrics.json`、`test_predictions.csv`、`ranking.csv`、`screening.json`和执行记录。注明输入来自原案例快照还是新生成，特征来自精确模拟、有限采样模拟还是真机。表4快照、三次补充评估记录和新的本地运行分别标记，不调整结果以匹配论文数字。

格式与来源边界：QED、SA和模型得分面向论文显示2位小数；得分不是校准概率。保持648/162固定数据划分。不得把20条生成字符串当成分类器测试集，也不得将`shots=128`解释为128次完整流程运行。
