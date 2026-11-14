# 第三方来源与许可

本仓库独立参考实现的MIT许可证不替代第三方数据、模型或依赖的条款。

- ChEBI-20文本/SMILES数据来自MolT5项目的`ChEBI-20_data`。上游仓库为BSD-3-Clause，原许可保存在`licenses/MolT5-BSD-3-Clause.txt`；分子基础数据仍应保留ChEBI来源归属。论文：Edwards等，Translation between Molecules and Natural Language，EMNLP 2022，375–413。
- Photoswitch子集来自The Photoswitch Dataset，上游MIT许可保存在`licenses/Photoswitch-MIT.txt`。论文：Griffiths等，Data-driven discovery of molecular photoswitches with multioutput Gaussian processes，Chemical Science 2022，13，13541–13551。
- ZINC15子集来自论文使用的固定分类数据，保留ZINC15来源，不将本仓库代码许可证解释为对ZINC数据的重新许可。来源：https://zinc15.docking.org/ 。论文：Sterling和Irwin，ZINC 15–Ligand Discovery for Everyone，JCIM 2015，55，2324–2337。
- GPT-2基座在使用者首次正式训练时从模型库获取；本仓库不重新分发其权重。请保留该模型和后续分发检查点适用的模型许可。
- RDKit、scikit-learn、NumPy、PyTorch、Transformers、Tokenizers及SpinQit由各自项目提供，安装版本记录在运行产物中，其软件许可由相应发行包保留。SA计算使用RDKit随包提供的`Contrib.SA_Score`实现，没有复制其源码或数据文件。

第三方项目名称用于来源说明，不表示其对本仓库、晓梦或论文结论的认可。
