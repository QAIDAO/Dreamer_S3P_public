# 数据与来源

## 分类数据

`classification/train_648.csv`与`test_162.csv`保留提供的固定划分，仅添加公开仓库中的稳定分子编号和来源字段。训练正负各324，测试各81。标签1为Photoswitch来源，标签0为ZINC15来源；来源标签不等同于逐例验证的光响应/无光响应标签。

可运行`inspect-data`核对Bemis-Murcko骨架及规范SMILES交集。当前固定划分两项交集均为0，训练集芳基偶氮结构正类322/324、负类1/324，测试集正类81/81、负类0/81。脚本将空骨架视为同一分组，避免把无环分子的空值当作自动无交叉。

源数据集合见[Photoswitch](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset)与[ZINC15](https://zinc15.docking.org/)。本仓库保留已提供的选择和划分，不从上游重新抽样来改变现有810个结构。

## 生成案例

`case/generated_20.csv`保留包含无效结构在内的20条原始输出，按顺序赋予G01至G20编号。无效SMILES是实际流程输出的一部分，不能从快照中删去后声称有效率为100%。`case/candidates_5.csv`关联G编号与表4的M1至M5编号，QED、SA及模型得分按论文保留2位小数。

## 生成器训练数据

ChEBI-20文件取自[MolT5公开仓库](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data)。本仓库保存文件内容与SHA-256，`scripts/fetch_training_data.py`只在文件缺失时下载并验证校验值；上游内容变更时会报错，不静默替换。第三方许可文本保留在`licenses/`。

## 结果记录

`reference/paper_results.json`为论文原报告值；`reference/simulation_three_runs.json`为seed 6、7、8三次模拟补充评估记录，包含混淆矩阵和ROC-AUC。数据校验脚本检查每个矩阵合计162例且正负各81，并核对报告准确率。新运行的逐样本预测和指标仅写入`outputs/`。

`reference/case_execution.json`保留原案例执行事件中的阶段、状态、计数和实际后端来源，移除宿主会话、设备标识、请求/任务名和本机路径。它是脱敏历史记录，不是本仓库新提交的量子实验。

`data/manifest.json`列出发布数据文件的校验值、规模和来源。数据文件不包含商业应用会话、工作区路径、远端个人账号、设备凭据或私人运行标识。
