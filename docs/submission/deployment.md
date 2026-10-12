# 部署说明

下列命令从项目根目录执行，路径均为相对路径。此前单机 12 节点配置、650 参数与 7,850 参数的一轮运行均已完成真实验证；2026-10-06 新默认为 13 个角色，旧性能记录只代表其原配置。本文的三机步骤描述物理部署方法，其性能应引用对应机器的实测记录。最终验收结果以设计报告和实测记录为准。

当前新建加密实验统一使用 LegoGroth16（`lego_norm_v1`），须选择与模型维度及 8 位量化匹配的 CRS；完整部署自动补齐 MNIST 650 维、CIFAR-10 1,930 维的默认开发参数。`plain` 明文实验无需证明和 CRS。旧逐坐标、5A/5B 仅作为历史记录和研究材料保留，旧性能数字不代表当前证明方案的耗时。

## 1. 环境与安装

- 当前锁定部署要求 Python 3.12+，推荐 CPython 3.12；其他受支持版本先核验全部依赖的 wheel 兼容性。底层依赖为 `py-arkworks-bls12381==0.5.0`，需要匹配平台和解释器的 wheel，或者具备对应源码编译环境。
- 完整部署自动构建前端，需要 Node.js 22.12+ 或 24 与 npm。已有完整、匹配的 `web/dist` 后，日常运行服务不需要 Node.js。
- 首次完整部署安装运行依赖并准备 MNIST、CIFAR-10。部署完成后使用离线启动，只校验本地环境、数据及前端，不安装依赖或下载数据。
- 本版本默认 CPU 训练；完整部署同时安装 CPU 版 PyTorch 与 torchvision，训练实现使用 CPU/float64。密码 CUDA 后端独立使用 NVIDIA NVRTC，不需要安装 CUDA 版 PyTorch。
- Windows 从源码构建原生扩展需要 MSVC x64 C++ 工具及 Windows SDK；Linux 需要 `cc`、`ar` 等编译链接工具。缺少 Rust 时，部署脚本自动下载官方最小工具链，校验 SHA-256，安装到项目 `tmp/native-toolchain`，不修改系统 PATH。

推荐先完成完整部署，再打开服务。Windows：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -SetupOnly
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Offline
```

Linux：

```bash
bash scripts/start_demo.sh --setup-only
bash scripts/start_demo.sh --offline
```

`SetupOnly` 不启动角色或控制服务。它安装基础锁定依赖、CPU Torch/torchvision 和包含 Lego、聚合验证及批量算术的当前原生扩展，然后在目标运行目录核验或补齐默认 CRS；检测到 NVIDIA GPU 时安装 NVRTC 并执行密码算术自检；随后准备两套数据、自动安装并构建前端，最后执行 `pip check`。普通一键脚本也会先完成同样的准备再启动应用。首次下载、本地编译或 CRS 生成可能耗时较长，后续复用经过检查的环境、缓存和完整参数。

依赖下载由 `scripts/deployment_downloads.py` 统一处理：普通 Python 包默认依次使用清华、华为云、PyPI，CPU Torch 使用上海交大镜像与 PyTorch 官方源，npm 使用 npmmirror、华为云与 npm 官方源。各次尝试保留超时和有限重试，不修改用户或系统配置。`DGFL_PIP_INDEX_URL`、`DGFL_TORCH_INDEX_URL`、`DGFL_NPM_REGISTRY` 可固定对应源；已有显式 pip/npm 配置也会保留。

完整部署记录位于 `.venv/dgflow-deployment.json`，其中 `proof_parameters` 记录默认参数核验结果。原生扩展另用 `tmp/native-toolchain/install.json` 绑定源码和实际二进制摘要；相同的 0.2.0 版本号不代表同一构建。离线重启时保留该记录，或准备匹配平台的本地 wheel 及其 `.whl.source.json`。部署阶段通过 `scripts/setup_lego_parameters.py --runtime TARGET --defaults` 补齐两个默认形状；其他模型维度仍需显式建立或安装匹配参数。

NVIDIA 驱动由操作系统提供，pip 不安装驱动。无 NVIDIA 显卡时完整部署采用 CPU；有显卡但驱动或密码自检失败时会说明原因，CPU 仍可用，只有通过精确算术自检后才开放 GPU。当前按实际显卡的 `sm_` 架构由 NVRTC 生成 CUBIN，并使用独立二进制缓存，避免较新 NVRTC 生成的 PTX 被较旧驱动拒绝加载；原有 `.ptx` 不作为新二进制读取。此方式遵循 CUDA 同一大版本的小版本兼容条件，仍须核验实际驱动、硬件和精确运算自检，不能仅以“支持 CUDA 12”判断。参见 [NVIDIA 官方兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)和 [NVRTC CUBIN 接口](https://docs.nvidia.com/cuda/nvrtc/)。

已自行创建环境时，Windows 可通过 `-PythonPath` 指定实际解释器；Linux 使用 `DGFL_PYTHON_PATH`。同一解释器完成部署后再离线启动，不跨操作系统复制整个虚拟环境。

仅需基础 NumPy 演示环境时，也可按锁定依赖手动安装：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
```

基础锁文件只覆盖 NumPy 演示环境，不等同于上述完整部署。手动安装 CPU 训练依赖使用 `pip install -r requirements-torch.txt`；执行测试另装 `pip install -e ".[test]"`。这里的 `pip` 均应使用选定环境的 `python -m pip`。只装基础依赖后运行最小演示应直接使用第 2 节的 `cli init/start/serve`，并在页面选择明文基线；新加密实验另需当前原生扩展和匹配 CRS。

一键部署自动构建前端；需要单独重建时执行：

监控页的 TensorBoard 已加入基础锁定依赖。旧环境更新后须重启控制服务并重建前端；旧离线包也需重新生成，以包含新增依赖。使用和采样口径见[监控与 TensorBoard](monitoring.md)。

```powershell
Set-Location web
npm.cmd ci
npm.cmd run build
Set-Location ..
```

Linux 手动环境使用 `python3 -m venv .venv`，将下文的 Python 路径改为 `.venv/bin/python`；前端用 `npm`。不同操作系统不得直接复制整个虚拟环境，应重新安装匹配平台的依赖。

### MNIST 下载超时与离线复用

`prepare-data` 默认优先从[飞桨官方实现指定的北京 BOS 镜像](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/datasets/mnist.py)下载四个原始 gzip 文件；传输失败后依次尝试 PyTorch 使用的 OSSCI S3 地址与 [CVDF 发布的 HTTPS 镜像](https://github.com/cvdfoundation/mnist)。每轮最多尝试三个地址，默认最多三轮，轮间短暂退避。有效缓存会复用，只补缺失文件；摘要错误、超出大小上限或降级到 HTTP 会立即拒绝。国内源四个文件的完整下载、原始 MD5 与 IDX/CRC 已通过本机核验。默认 30 秒为每次阻塞网络操作的超时，不是整个数据准备的总时限。网络较慢时可单独先准备数据：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --mnist-timeout 120 --mnist-retries 2
```

`--mnist-timeout` 支持 1–300 秒，`--mnist-retries` 支持 0–5 个额外轮次。可用 `--mnist-source https://dataset.bj.bcebos.com/mnist/` 显式固定国内 HTTPS 基址；指定后只尝试该地址。镜像同样受原始文件摘要校验约束，不能保证它在所有网络中都可访问。Windows 完整部署可用 `-MnistSource`，Linux 用 `DGFL_MNIST_SOURCE` 传入该基址。

最可靠的离线方法是在能联网的机器上准备一次，将 `data/mnist/raw` 下这四个**原始压缩文件**复制到目标项目的相同位置；不用解压，不用复制虚拟环境：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

在目标项目根目录执行校验并生成元数据：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --offline
```

此命令不联网，缺少文件会明确列出；损坏的缓存不会自动覆盖。仅完成 MNIST 校验时可直接运行最小 CLI 演示；完整一键离线启动还要求两套数据、Torch、当前原生扩展、所需 NVRTC 和前端均已准备。公开数据可单独保存为部署材料，源码无需包含数据集。

### CIFAR-10 下载超时与原始包离线复用

默认先尝试 [MindSpore 官方教程发布的镜像](https://www.mindspore.cn/tutorials/zh-CN/master/dataset/sampler.html)，再尝试 [Toronto 原始发布地址](https://cave.cs.toronto.edu/kriz/cifar.html)。两者都必须通过同一原始二进制包摘要校验。默认每次阻塞网络操作超时 60 秒，额外重试两轮；这不是整个下载的总时限。慢网络可提前单独准备：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --dataset cifar10 --cifar-timeout 120 --cifar-retries 2
```

`--cifar-timeout` 支持 1–300 秒，`--cifar-retries` 支持 0–5 个额外轮次。用 `--cifar-source` 指定 HTTPS 基址后只尝试该地址，例如：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --dataset cifar10 --cifar-source https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/
```

Windows 完整部署对应 `-CifarSource`，Linux 对应 `DGFL_CIFAR_SOURCE`。下载失败保留既有缓存，不接受降级 HTTP、摘要错误或超出大小上限的内容。

离线复用时，把约 162 MiB 的原始 `cifar-10-binary.tar.gz` 放到 `data/cifar10/raw/`，不需要手工解压，也不要用 Python pickle 版本替代。随后执行：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --dataset cifar10 --offline
```

该命令完全使用本地原始包，校验摘要、解压和检查二进制批次，再生成元数据；缺包或缓存损坏会明确报错。训练与数据加载器不会自行下载。

### 加密实验的 LegoGroth16 参数

纯源码目录不包含运行目录、生成的 CRS 或身份。完整部署在安装原生后端后，调用 `setup_lego_parameters.py --runtime TARGET --defaults` 准备 MNIST 默认 650 维和 CIFAR-10 默认 1,930 维的 8 位参数。已有完整参数核验后复用，保留原指纹，不覆盖或重新随机生成；只对缺少的形状执行本地单方开发设置。离线部署同样可以本地生成，不下载参数。角色及实验启动本身不会生成 CRS。

已完成依赖部署的现有运行目录，可独立执行以下补齐命令，无需重新部署其他依赖：

```powershell
.\.venv\Scripts\python.exe scripts/setup_lego_parameters.py --runtime runtime --defaults --workers 4
# 非默认网格：例如 MNIST grid=4，对应 170 维
.\.venv\Scripts\python.exe scripts/setup_lego_parameters.py --runtime runtime --dimension 170 --bits 8 --workers 4
```

`--defaults` 只处理两个默认形状，改用其他模型网格后需按实际维度显式传入 `--dimension`。命令只保存公共 PK/VK 和清单，不保存设置陷门；这是单方开发设置，不能替代正式多方可信设置仪式。参数存放在目标运行目录的 `proof-parameters/<crs_hash>/`。仅新增参数无需重启服务，在网页点“刷新已安装参数”后手动选择匹配指纹。明文基线无需 CRS。三机需共享同组公共参数，部署顺序见第 3.3 节。

控制 API 的加密请求使用 `proof_suite=lego_norm_v1` 和匹配的完整 `proof_crs_hash`，安装清单由 `GET /api/proof-parameters` 返回。新任务不接受旧证明方案；历史结果仍按原配置查阅。

## 2. 单机部署

一次性准备并初始化：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data
.\.venv\Scripts\python.exe -m dgfl.cli init
```

`init` 创建身份、证书、公开注册表和 `runtime/cluster.json`。已有配置或密钥时会拒绝覆盖。以后重启应复用此目录，不要每次重新初始化；需要全新身份时选择新的运行目录，而不是覆盖旧密钥。

新部署默认 n=6 个客户端、w=3 个边缘授权节点、v=4 个云聚合节点。客户端范围 2–100，边缘与云范围分别为 2–32；边缘门限 s 为 2–w，云门限 e 为 2–v。对应 CLI 参数为 `--client-count`、`--authority-count`、`--aggregator-count`、`--authority-threshold`、`--aggregator-threshold`，`init`、`start`、`demo` 均支持。两个门限初始默认均为 2；当前密码实现不支持小于 2 的门限。

每个客户端恰好归属一个边缘，默认 `clientN → authority((N−1) mod w+1)`，这是工程分配规则。原始证明与提交只到归属边缘，所有边缘共享函数钥份额；归属边缘执行证明核验与 VerDec，再共享签名的余弦、范数、摘要和密文核心，供所有边缘共同筛选与授权。V/VerDec 是边缘内部操作。详细对照见[分层拓扑说明](../research/hierarchical-topology-20261006.md)。

新建 n=20、w=3、v=4、s=2、e=3 的独立本机集群：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli init --runtime runtime-paper --client-count 20 --authority-count 3 --aggregator-count 4 --authority-threshold 2 --aggregator-threshold 3
.\.venv\Scripts\python.exe scripts/setup_lego_parameters.py --runtime runtime-paper --defaults
.\.venv\Scripts\python.exe -m dgfl.cli start --runtime runtime-paper
.\.venv\Scripts\python.exe -m dgfl.cli serve --runtime runtime-paper
```

已有本机集群空闲时可通过部署页显式应用，或停掉控制服务后执行带参数的 `start`。未提供的边缘/云数量与门限沿用既有值；不带拓扑参数的 `start`/`demo` 沿用全部既有设置。旧 `cluster.json` 缺字段时按连续节点清单推断数量，门限默认 2，历史 6/3/3 不会在普通重启时扩成 6/3/4。

默认分组策略 `regroup` 接纳任意至少两名合格成员，奇数人数可正常聚合。`fixed` 只接纳完整二人组；声明人数为奇数时末位客户端不参与聚合，伙伴被拒绝的成员也会退出该组。页面等待全部部署节点在线后允许创建实验；运行中客户端掉线仍保留声明成员名单，并按合格人数和所选分组策略决定能否聚合。

拓扑调整保留签名身份、密钥交换身份、实验结果、数据缓存和证明参数。新单机初始化预留最多 100 个客户端、32 个边缘及 32 个云的身份，预留身份不会成为活动节点或任务成员。历史部署缺少新增身份时沿安全事务路径扩容并轮换本机 TLS 证书；CA 签名私钥仍不落盘。角色数量或门限变化会在校验进程归属后停止并重启受管活动角色，失败时恢复原配置和凭据；缩容不删除退役身份，退役节点不能参加新任务。运行中的实验禁止改变拓扑；三机部署需要显式列出全部主机并重新分发，不通过本机页面自动重配。

启动节点，再启动本机控制服务：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli start
.\.venv\Scripts\python.exe -m dgfl.cli serve
```

也可用 `python -m dgfl.cli demo` 组合“缺失时初始化、启动节点、运行控制服务”，其中 `python` 应为已安装本项目的解释器。一键脚本还会在服务启动前处理完整环境、双数据集和前端构建；已完成第 1 节的完整部署后使用离线参数启动。

在另一终端检查：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli doctor
```

打开 [本机控制页](http://127.0.0.1:8765)。控制界面仅面向本机浏览器；当前 Host/Origin 检查不支持把 8765 任意暴露到公网或直接作为远程管理服务。

浏览器控制请求必须与接收请求的页面来源完全一致，包括协议、主机名和端口；`localhost` 与 `127.0.0.1` 也属于不同来源。没有 `Origin` 头的本机命令行请求仍可使用。所有控制请求的实际正文上限为 32 KiB，分块传输同样受限，超限时在执行操作前返回 413。开发用 Vite 代理保留浏览器访问的 `Host`；若自行添加本机反向代理，应同样保留 `Host`，并仅通过受信代理配置传递 HTTPS 协议信息。

| 角色 | 身份与单机端口 |
| --- | --- |
| authority | `authority1`–`authorityW`，9101–`9100+W`；W 为 2–32，默认 3 |
| aggregator | `aggregator1`–`aggregatorV`，9201–`9200+V`；V 为 2–32，默认 4 |
| client | `client1`–`clientN`，9301–`9300+N`；N 为 2–100，默认 6 |
| coordinator / 控制服务 | 本机 8765；另有用于节点 RPC 的独立身份 |

节点采用 HTTPS/mTLS，控制页面与节点 RPC 不是同一个端口。单机节点监听回环地址。节点日志位于 `runtime/logs`，受管进程记录在 `runtime/pids`，实验结果在 `runtime/results`。

关闭控制服务使用 `Ctrl+C`；随后停止受管节点：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli stop
```

停止工具会核对 PID、创建时间、命令和路径再结束进程。若返回 `refused`，应人工排查身份不匹配，不要直接杀死该 PID。

## 3. 三机局域网部署

### 3.1 拓扑与边界

先准备 A、B、C 三台机器的稳定内网 IPv4 地址。示例地址只作配置示意，需要替换成实际地址。

| 主机 | 运行角色 | 示例地址 | 需要允许的节点端口 |
| --- | --- | --- | --- |
| A | authority1、aggregator1、aggregator4、client1、client4、coordinator | 192.168.10.11 | 9101、9201、9204、9301、9304 |
| B | authority2、aggregator2、client2、client5 | 192.168.10.12 | 9102、9202、9302、9305 |
| C | authority3、aggregator3、client3、client6 | 192.168.10.13 | 9103、9203、9303、9306 |

防火墙仅向这组实验机器开放需要的节点端口。A 本机浏览器访问 8765；B、C 不需要取得 coordinator 私钥，也不需要启动控制服务。

三机部署仍在每台机器上托管多个角色。它改善进程和主机分布，但不是完整的独立机构治理或安全隔离证明。整台 B/C 故障会同时失去 authority、aggregator 和两个客户端，不能等同于只失去一个聚合器。

### 3.2 一次性配置和按主机分包

在受控的初始化环境中复制并编辑配置：

```powershell
Copy-Item -LiteralPath .\configs\hosts.example.yaml -Destination .\configs\hosts.lan.yaml
```

将文件中各 `host` 改为真实内网 IP，保留示例全部 13 个活动角色身份及 A/B/C 归属。配置只接受主机地址，不填写 `https://` 或端口；角色端口由程序确定。coordinator 必须属于 A。

此示例使用 n=6、w=3、v=4、s=2、e=2，与新默认一致。其他规模应显式列出连续的 `client1`–`clientN`、`authority1`–`authorityW`、`aggregator1`–`aggregatorV`，初始化时传相同的数量与门限。保留旧三云 YAML 时必须传 `--aggregator-count 3`；不要遗漏节点后依靠自动发现。

使用一个全新的专用目录初始化，再分包：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli init --runtime provisioning-runtime --hosts configs/hosts.lan.yaml
.\.venv\Scripts\python.exe -m dgfl.cli bundle --runtime provisioning-runtime --output machine-bundles
```

输出 `machine-bundles/A/runtime`、`machine-bundles/B/runtime`、`machine-bundles/C/runtime`。每包包含公共 CA、公开身份注册表、共同集群配置、主机标记，以及**本机角色的私钥/身份文件**。只有 A 包包含 coordinator 的私钥。

此初始化工具是集中式的身份/证书准备工具；它与每轮模型加密用的 DKG 不同。初始化目录暂时持有整套传输身份，因此必须保留在受控离线准备环境，不能作为常规三机运行目录到处分发。**不要把完整 `provisioning-runtime/keys` 或全部 machine-bundles 一起复制给每台机器。**

### 3.3 分别安装并启动

每台机器使用同一版本程序，但使用全新的目标运行目录。把对应主机包内的 `runtime` 放到该机器项目根目录的 `runtime`；不要覆盖旧部署目录或合并另一主机的密钥。示例在 A 的新项目目录中：

```powershell
Copy-Item -LiteralPath .\machine-bundles\A\runtime -Destination .\runtime -Recurse
```

B/C 仅接收各自的身份包并按同样布局放置。先在 A 执行第 1 节的 `-SetupOnly`（Linux 为 `--setup-only`）完整部署，再将 A 的同组 `runtime/proof-parameters/<crs_hash>/` 公共 PK/VK 和清单复制到 B/C 对应运行目录；随后在 B/C 执行完整部署，让默认参数检查复用这些指纹，避免各机分别随机生成。此部署选项不启动控制服务，适用于 B/C；只分发公开参数，不复制 A 的身份私钥。每台机器都提前准备公开 MNIST 缓存，单独校验命令为：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data
```

此处下载/复制的是公开数据。完整缓存位于各项目的 `data/mnist/raw`，训练时按规则选择本地分区；不要称为三个现实机构各自收集的数据。加载器不会在训练期间自动联网下载。

运行 CIFAR-10 时，每台承载客户端的机器以及主控 A 也必须提前准备对应缓存：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --dataset cifar10
```

缓存位于各项目的 `data/cifar10/raw`。控制台的数据准备只处理主控本机缓存，不会替 B/C 下载；首次准备约下载 162 MB 官方二进制包。基础 `prepare_offline.py` 只自动纳入 MNIST；第 5.1 节的完整离线材料包含两种数据，也可单独复制 CIFAR-10 缓存并在目标机器显式执行准备命令复核摘要。实验选择 `dataset=cifar10`，默认 RGB 8×8 池化、1,930 参数线性模型；当前入口支持网格 2–25，尚未接通论文完整 CNN 安全训练。

加密实验须在三台机器选择同一组 `proof-parameters/<crs_hash>/` 公共参数，保持 PK、VK、清单和指纹完全一致。非默认维度同样只在一处生成再复制，不要在 A/B/C 分别随机建立三组 CRS；若各机还保留其他参数，实验应显式固定共同指纹。明文实验无需参数。

分别执行：

```powershell
# A
.\.venv\Scripts\python.exe -m dgfl.cli start --machine A
# B（在 B 的终端）
.\.venv\Scripts\python.exe -m dgfl.cli start --machine B
# C（在 C 的终端）
.\.venv\Scripts\python.exe -m dgfl.cli start --machine C
```

仅在 A 启动控制服务并执行全群健康检查：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli serve
.\.venv\Scripts\python.exe -m dgfl.cli doctor
```

`serve` 常驻当前终端，因此 `doctor` 应在 A 的另一个终端执行。B/C 包故意不含 coordinator 身份，当前 `doctor` 的全群 RPC 检查在 B/C 会报告缺少 coordinator；不要为消除这个提示而把 A 私钥复制过去。

若改用自定义 `--runtime`，当前控制/训练代码从该目录的父目录寻找 `data/mnist`。把运行目录放在项目根目录的直接子目录下最容易保持数据路径一致；不要仅把 `--runtime` 指向嵌套分包目录，却仍假设使用默认数据路径。

关闭时在每台机器分别执行匹配的 `stop --machine A/B/C`。它会停止该机所有受管角色，不适合用来模拟“只停止一个聚合器”。

### 3.4 mTLS 检查

证书的地址在初始化时与配置绑定。修改 IP 后只编辑 `cluster.json` 可能导致证书名称不匹配，应使用新目录重新准备配置和身份。不要把 `verify=False`、关闭证书校验、开放任意来源当作部署解决方案。

全群验收至少保留 A/B/C 上各自角色清单、节点健康记录、一次完整真实任务、结果哈希与实际网络配置。没有这些证据时，将状态写为“三机部署说明已提供，物理环境实测待完成”。

## 4. 可用性实验的正确解释

界面的“本轮排除聚合节点”范围是 0–v，仅改变聚合调用的参与名单；剩余 v−排除数 必须至少达到 e，否则拒绝启动。新默认 e=2、v=4 时排除两云仍满足门限，排除三云不足；旧 e=2、v=3 则排除两云不足。此设置不会关闭任何服务，健康页仍可能显示被排除的云在线。

真实进程崩溃测试需要另外记录具体聚合进程、故障时点、剩余健康状态和是否发布模型。当前编排会预检所需节点，并不承诺自动识别任意时点的崩溃后重新选择参与者。已得到足够合法份额时的门限恢复、主动逻辑排除和自动故障切换是三件不同的事。

新任务 DKG 仍依赖全部 w 个边缘完成准备，不由 s/w 恢复门限推导出 DKG 可容错。主控承担工程 RPC 转发与编排，仍是单点；它退出后需重启服务，新加载的未结束记录会被标为中止，不能宣称无缝恢复正在执行的密码轮次。固定批次也会放大客户端退出的可用性损失。此拓扑调整不构成完整论文安全、性能或 PFLlib CNN 复现。

## 5. 测试、离线复验和提交包

已安装测试依赖后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

MNIST 实际训练检查脚本位于 `tests/training/run_mnist_smoke.py`；它的参数应先用 `--help` 核对。公开训练缓存不是测试成功的替代证据。

第 1 节的完整部署已经完成时，Windows 使用 `start_demo.ps1 -Offline`，Linux 使用 `start_demo.sh --offline`。这会检查 Torch、当前原生扩展、所需 NVRTC、双数据集与前端；只有基础 NumPy/MNIST 环境时应使用下述最小 CLI 路径。缓存校验失败会报错，不会静默改写损坏文件。

### 5.1 准备完整离线部署材料

在与目标相同 Python 实现、次版本、操作系统和架构的准备机器上，先完成第 1 节完整部署，其中已核验或补齐默认 650/1,930 维、8 位 Lego CRS。随后可将这些公共参数随包分发，以便目标复用相同指纹：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --parameters-runtime runtime
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --verify-only
.\.venv\Scripts\python.exe scripts/package_submission.py --output dist/submission-full --full-offline-path full-offline
```

完整材料包含 CPU Torch/torchvision 及当前环境所需的完整 wheel 闭包、受支持平台的 NVRTC、当前原生密码 wheel 与源码凭据、MNIST 四个原始 gzip、CIFAR-10 原始二进制归档和已检查的批次，以及带源码指纹的 `web/dist`。清单逐项记录大小与 SHA-256，另检查 wheel 的依赖闭包、平台标签、原生和前端源码绑定、官方数据摘要与结构。输出目录须尚不存在；已有目录用 `--verify-only` 复查。`--wheelhouse` 可复用精确的完整本地 wheel 集合，`--native-wheelhouse` 可选带源码凭据的本地原生 wheel。

`--parameters-runtime` 可选，默认不包含 CRS。上例只携带 `runtime` 已安装的公共 PK/VK 与参数清单，原 `setup_kind` 保留；默认生成的是单方开发参数，不代表正式多方可信设置。携带参数时，目标恢复并复用原指纹；省略时，目标完整部署可离线本地补齐默认 CRS，不下载参数。三机部署须携带并共享同组公共参数，或只在 A 生成再分发，不能各自随机生成。身份私钥、设置秘密、实验历史均不进入完整材料。`plain` 不使用证明或 CRS。

在目标机器解压 `source.zip`，系统需已有清单所对应的 Python；GPU 计算还需兼容的 NVIDIA 驱动。预构建完整包不要求目标安装 Node.js、Rust 或 C++ 编译器。先用系统 Python 执行仅依赖标准库的文件及平台校验：

```powershell
py -3.12 full-offline/VERIFY.py
```

通过后按 `full-offline/INSTALL.txt` 从本地 wheel 安装。Windows x64 / CPython 3.12 的典型流程如下；其他目标按清单调整解释器，不跨平台复制 `.venv`：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links full-offline/wheelhouse -r full-offline/requirements-full-lock.txt
$nativeWheel = (Get-ChildItem -LiteralPath .\full-offline\native-wheels -Filter *.whl).FullName
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps $nativeWheel
.\.venv\Scripts\python.exe -m pip install --no-index --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --verify-only
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --restore-assets
.\.venv\Scripts\python.exe scripts/setup_environment.py --offline
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Offline
```

`--restore-assets` 只补公共数据、前端、原生 wheel/来源凭据和可选公共 CRS；参数默认放到 `runtime/proof-parameters`，可用 `--runtime PATH` 指定运行目录。已有文件与包内摘要冲突时拒绝，身份不会初始化或覆盖。随后的 `setup_environment.py --offline` 核验已有默认 CRS，未随包携带且本地缺少时执行本地开发设置，不联网。Linux 将解释器换成 `.venv/bin/python`，按 `INSTALL.txt` 安装相同平台的 wheel，再执行 `bash scripts/start_demo.sh --offline`。整个目标安装均使用 `--no-index`，完成深度部署校验后，启动和训练不再下载依赖或数据。

纯源码包不携带这些大体积材料；完整材料可单独保管，或用上例 `--full-offline-path` 在核验后随提交 ZIP 分发。`validate_release.py` 仍针对下节的基础包，不是此完整部署的验收工具。

### 5.2 准备基础 NumPy/MNIST 离线材料

在与目标环境匹配、能够联网的准备机器上，先完成基础依赖安装和 MNIST 校验，再执行：

```powershell
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data
.\.venv\Scripts\python.exe scripts/prepare_offline.py --output offline
.\.venv\Scripts\python.exe scripts/prepare_offline.py --output offline --verify-only
```

`prepare_offline.py` 保留原来的基础包范围：下载基础锁文件对应 wheel，复制经过检查的四个 MNIST 原始压缩文件，生成 `offline/manifest.json` 与 `offline/INSTALL.txt`，可另纳入兼容的本地原生 wheel。它不包含 Torch/torchvision、NVRTC、CIFAR-10、Node.js、系统驱动或 Python 解释器，因此该包本身不能满足新的完整一键离线启动。清单记录文件大小、SHA-256、Python 版本和平台，且不分发节点身份。输出目录已存在时拒绝覆盖；使用 `--verify-only` 复查原目录，或者选择新目录重新准备。已有完整匹配 wheel 缓存时，可追加 `--wheelhouse cached-wheels` 避免再次下载。

目标机器需事先安装与清单匹配的 Python 解释器与操作系统架构。在新的项目目录中，按包内 `offline/INSTALL.txt` 操作。例如，清单确为 Windows x64 / CPython 3.12 时：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links offline/wheelhouse -r offline/requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-index --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -c "import shutil; shutil.copytree('offline/data/mnist', 'data/mnist', dirs_exist_ok=True)"
.\.venv\Scripts\python.exe scripts/prepare_offline.py --output offline --verify-only
.\.venv\Scripts\python.exe -m dgfl.cli prepare-data --offline
.\.venv\Scripts\python.exe -m dgfl.cli init
.\.venv\Scripts\python.exe -m dgfl.cli start
.\.venv\Scripts\python.exe -m dgfl.cli serve
```

此最小示例用于新的本机运行目录，使用已准备的 MNIST、NumPy 和 CPU，启动后选择明文基线，无需密码证明或 CRS；不代表 Torch、Lego、CIFAR-10 或 GPU 全功能部署，提交包还需已有 `web/dist`。新加密实验必须另行安装当前原生扩展与匹配参数。不要复制覆盖需保留的旧数据，已有身份时不要重复 `init`。三机部署先放好各自 runtime 包，再执行第 3 节 `cli start --machine A/B/C`，仅 A 执行 `cli serve`；不在 B/C 使用会打开控制服务的一键入口。离线 wheel 包只适用于记录的平台与解释器。测试依赖不在基础锁内，离线执行 pytest 需要单独准备。

完整功能的离线运行使用第 5.1 节完整材料，或在目标机器提前完成第 1 节完整部署并保留环境、原生来源记录、两套数据和前端。两个默认 CRS 由完整部署核验或补齐，非默认模型维度仍需显式准备。基础离线包和 `validate_release.py` 的成功结果不能代替完整环境及模型参数检查。

### 5.3 运行实验矩阵与打包

当前 `configs/experiments.yaml` 和 `configs/full-data-experiment.yaml` 的加密 case 使用 LegoGroth16。运行前先安装模型维度、8 位量化对应的 CRS。加密 case 未指定 `proof_crs_hash` 或为 `null` 时，脚本通过 `GET /api/proof-parameters` 查找对应数据集和网格的参数；仅有一个匹配项时自动选择，没有匹配或匹配多组时明确拒绝，需要先安装参数或在配置中显式填写指纹。显式提供的指纹不会被替换，明文 case 无需参数。

选定指纹会写入输出目录的 `config.json` 并纳入矩阵摘要，恢复运行沿用已冻结的指纹，不会因新安装 CRS 自动换用另一组。历史研究 JSON 和旧输出仍按原方案解释；不要直接提交旧 5A/5B 配置，也不要把切换到 Lego 的新矩阵写入旧输出目录。

在节点与控制服务已启动、没有其他活动任务且参数准备完成时，将复验记录写入新的本地目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --config configs/experiments.yaml --output runtime/reproductions/formal-lego
```

包内 `docs/submission/evidence` 保存原始交付证据，供查阅与核对，不作为新机器的任务输出目录。矩阵脚本恢复运行时会复用输出目录中的任务标识，因此只有保留原 `runtime` 及其中任务记录、配置也未变化时，才能恢复原输出。更换机器、重新初始化运行目录或修改配置后，都应选择新的输出目录。失败或中止应保留并解释；不能删除失败记录后声称全部成功。具体矩阵配置与性能结论以设计报告为准。

上述矩阵结束后，完整 MNIST 复验使用另一配置和输出目录，同样解析或显式指定匹配 CRS：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --config configs/full-data-experiment.yaml --output runtime/reproductions/full-data-lego
```

真实进程故障检查应在所有训练任务结束、实际单机全部活动节点恢复在线后单独执行。脚本按当前 v/e 停止足够的云，分别验证恰好满足和低于门限，再检查固定分组下的客户端缺席；所有进程先核对归属，随后恢复本次停止的角色。不要与训练矩阵同时运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_fault_checks.py --output runtime/reproductions/faults
```

故障检查不恢复旧输出，重复复验需另选尚不存在的输出目录；保留原故障记录及脚本的恢复状态。

具备 README、源码和设计报告后，可生成可核验提交包：

```powershell
.\.venv\Scripts\python.exe scripts/package_submission.py --output dist/submission
```

当前打包器输出 `source.zip` 与 `manifest.json`，记录每个文件的大小和 SHA-256，并检查实际包加清单小于 4 GiB。若主办方对“4G”采用不同口径，以其最终要求为准，提交前人工核对实际大小。

打包采用白名单，默认包含后端源码、测试、配置、完整前端源码及其构建结果、前端锁文件和提交文档；排除 `node_modules`、运行时密钥、完整数据缓存、虚拟环境和开发计划。默认源码 ZIP 不包含可重建的离线依赖包。

如已生成并验证项目根目录中的 `offline`，使用以下命令把经过验证的离线材料一并交付：

```powershell
.\.venv\Scripts\python.exe scripts/package_submission.py --output dist/submission --include-offline
```

打包器重新校验离线清单后才纳入 wheel 与公开数据。核对最终 `manifest.json`，确认实际包含需要的源码、构建结果、文档和离线材料；发布记录应分别列出基础离线重装与完整部署的验证范围。

上述 `--include-offline` 保持 NumPy/MNIST 基础包范围。完整功能材料使用第 5.1 节的 `--full-offline-path PATH`，打包时按完整校验器核对当前源码，再以 `full-offline/` 路径收录，二者不可混称。

对包含 `offline` 的包执行独立重装校验：

```powershell
.\.venv\Scripts\python.exe scripts/validate_release.py --package dist/submission --output tmp/release-checks
```

校验器会在指定输出目录下新建唯一工作目录，验证 ZIP 与清单，创建全新虚拟环境，从包内 wheel 离线安装，再检查命令入口、真实小向量密码流程、MNIST 读取及前端构建产物。它不启动节点，也不使用现有 `runtime`。默认保留 `release-check-*/report.json` 并清理本次环境；追加 `--keep` 可保留本次解包目录与新环境。当前发布包的 wheel 针对 Windows x64 / CPython 3.12，需在匹配环境下执行。

静态扫描会拒绝敏感密钥文件、明显绝对开发路径和自有材料中的身份字段，但不等于完整匿名性证明。发布前人工检查截图、异常日志、文件属性及文档内容；上游公开版权归属须保留，不应为了匿名而删除第三方许可。
