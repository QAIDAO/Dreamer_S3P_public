# 可选NMR设备执行

CPU模拟器足以运行默认最小示例。真实设备访问需要使用者自己的设备权限、SDK及连接信息。本仓库没有预设IP、账号、密码或设备标识，也不在安装和测试阶段探测设备。

先运行：

```bash
python scripts/s3p.py plan
```

若从头为训练、测试及5个候选采集特征，最多需要815条未缓存线路；每条线路`shots=128`表示采样128次。缓存可能减少提交数量。论文生成案例中的5条候选线路与从头训练分类器所需线路数量分别计算。

## SDK环境

`adapters/spinq_worker.py`使用SpinQit 0.2.4的`Circuit`、`NMRConfig`、原生编译器与`get_nmr().execute`接口。按该版本要求准备独立Python 3.10环境，安装`numpy>=1.24,<2`和`spinqit==0.2.4`。此SDK适配器根据接口编写，本仓库的本地验证不连接真实设备。

只在本地进程环境中设置`SPINQ_NMR_IP`、`SPINQ_NMR_PORT`、`SPINQ_NMR_USERNAME`和`SPINQ_NMR_PASSWORD`。不要将真实值填入仓库文件。

在已获得执行授权后，指定实际设备worker解释器运行：

```bash
python scripts/s3p.py pipeline --backend spinq-nmr --worker-python YOUR_ISOLATED_PYTHON --allow-hardware --output outputs/hardware
```

`YOUR_ISOLATED_PYTHON`是使用者本地独立环境的解释器路径，不是仓库内的固定地址。

## 数据接口与状态

每个请求包含3个角度、单层配置、采样数、任务名及请求校验值；不包含连接信息。设备返回的计算基概率按000至111排列，接口约定为`q2 q1 q0`。接入其他后端时，应按平台位序适配并核对，不改变分子编号。

父进程在提交前写入`submitting`状态。超时、worker失败或返回数据不完整后，状态设为`unknown`并停止。先核对设备队列，再在确有授权时使用`--retry-unknown`；已完成结果直接复用缓存。任何失败都不会自动转成模拟器结果。

输出只保留阶段、请求校验值、任务名、概率及实际来源，不持久化SDK异常中的原始连接详情。模拟器与真机缓存按不同请求校验值保存，不能混用。
