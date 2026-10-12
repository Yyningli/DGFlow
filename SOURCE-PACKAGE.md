# DGFlow 源码项目说明

当前源码版本包含双数据集镜像下载、完整部署准备、GPU 自动识别和此前 P1/P2 修复。源码、测试、配置和必要文档保留；运行材料由部署脚本生成，并由 Git 忽略。

## 本次源码版本

本次修复源码发布标识：`dgflow-review-improvements-20261011`，变更与实测范围见[审查记录](docs/research/review-improvements-20261011.md)。此前 `dgflow-20261008-complete-deployment-source` 及其清单只描述当时冻结的文件；后续源码或文档修改后，须完成相应验证并生成新的发布标识、清单与校验文件，不能沿用旧清单声称当前文件已经验收。

MNIST 与 CIFAR-10 均支持 HTTPS 镜像、有限重试、自定义下载源及 `prepare-data --offline`。MNIST 默认优先使用飞桨官方的北京 BOS 镜像，再回退 OSSCI、CVDF；CIFAR-10 优先使用 MindSpore 官方国内镜像，回退原始站点。所有源固定校验原始 MD5，两种国内源均已实测完整下载及结构校验。部署助手在打开服务之前完成训练依赖、完整原生密码扩展、GPU 编译依赖、两种数据集和前端准备。

GPU 硬件识别不依赖 NVIDIA Python 包；部署时自动安装所需 NVRTC，控制服务启动后自动对主控与边缘节点执行精确密码运算自检。默认计算设备仍为 CPU，自检通过后可在页面选择 GPU。

当前新建加密实验只使用 LegoGroth16（`lego_norm_v1`），必须安装并选择当前模型维度、8 位量化对应的 CRS。`plain` 明文基线不生成密码证明，无需 CRS。旧逐坐标、5A/5B 方案及原性能证据保留用于历史记录与研究核对，不再是当前新实验的选项。

2026-10-08 曾完成本机全新环境部署、禁网离线检查和 CIFAR-10 / Torch / GPU 真实流程验证，详见[部署验收记录](docs/research/evidence/deployment-ready-20261008.json)。历史验收材料保留，运行部署和交付产物另行归档，不属于当前纯源码目录。

纯源码包包含前端源码和依赖清单，不包含已安装的环境、完整数据缓存、前端构建产物、运行目录、生成的 CRS 或身份。完整部署脚本准备环境和数据，并在目标运行目录自动补齐默认开发参数。需要发布时，按本文末尾命令冻结当前源码。

## 保留内容

- `src/`：Python 服务、训练、密码协议与 CUDA 源码。
- `native/dgfl-native/`：Rust 源码、Cargo.toml、Cargo.lock 和构建配置。
- `web/`：Vue 源码、测试、测试样本、依赖清单及锁文件。
- `tests/`、`scripts/`、`configs/`、`.github/`：回归测试、工具、部署示例与 CI。
- `docs/`：协议说明、研究说明、参考来源和报告资源；原 DGFlow 论文不进入交付包。部分历史证据由报告生成器和回归测试直接使用，作为必要项目资源保留，不能作为当前源码重跑所得结果。
- 根目录 README、依赖清单、pyproject.toml、第三方声明与忽略规则。

完整 CNN 仍是独立本地训练模块，尚未接入安全联邦服务。单归属验证、门限可用性和开发 CRS 的边界仍以 README 与协议说明为准。

## 生成物与既有部署

全新安装不依赖旧部署归档。以下内容由部署或实验生成，不应加入源码包：

- `runtime/`：原部署、身份密钥、CRS、实验结果、节点状态和日志。
- `data/`：已校验的 MNIST 与 CIFAR-10 公开数据缓存。
- `dist/`：上次生成的源码 ZIP、交付清单和校验文件。
- `.venv/`、`web/node_modules/`：原依赖环境。
- `web/dist/`：前端构建产物。
- Python 缓存、egg-info，以及根目录的 `SOURCE-MANIFEST.json`。
- `tmp/native-toolchain/`：项目私有 Rust 工具链、Cargo 缓存、匹配 wheel 与构建指纹。

若自行保留过旧部署归档，可以恢复对应 `runtime/` 和公开数据后继续使用原身份、拓扑、CRS 与历史记录；不要先创建新的同名部署再覆盖。旧虚拟环境不要跨机器或跨操作系统复制。

## 从源码启动

锁定部署要求 Python 3.12+，推荐 Python 3.12。安装 Node.js 22.12+ 与 npm，并具备系统 C/C++ 构建工具。Windows 原生构建需要 Visual Studio C++ 工具和 Windows SDK，Linux 需要 C/C++ 编译器。NVIDIA GPU 需要系统驱动；其余项目依赖由脚本准备。在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -SetupOnly
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Offline
```

第一条联网完成部署，不启动任何节点或控制服务；第二条仅使用已部署环境和缓存启动。直接省略 `-SetupOnly` 也可完成部署后立即启动。Linux 对应 `bash scripts/start_demo.sh --setup-only` 和 `bash scripts/start_demo.sh --offline`。

部署成功后 `.venv/dgflow-deployment.json` 记录依赖版本、原生源码与二进制摘要、`proof_parameters` 默认 CRS 核验结果、GPU 自检、数据及前端状态。失败时不会发布就绪标记，也不会启动服务。源码变化后会重新核对原生与前端构建指纹，旧版本号相同的原生二进制不能冒充当前构建。

下载源可在部署时指定：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -SetupOnly -CifarSource https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/
```

Linux 使用 `DGFL_MNIST_SOURCE` / `DGFL_CIFAR_SOURCE` 环境变量。镜像和离线缓存校验详见[部署说明](docs/submission/deployment.md)。运行中的训练不会下载数据或安装依赖，GPU 内核只在本地编译。完整部署在目标运行目录自动准备 MNIST 650 维和 CIFAR-10 1,930 维的 8 位 CRS：核验并复用已有完整参数，不覆盖或重新随机生成；缺少时使用本地单方开发设置补齐，离线也不下载参数。这不是正式多方可信设置仪式，角色及实验启动本身不生成 CRS。纯源码仍不携带生成的参数或身份。仅增加参数后，在网页刷新已安装参数并选择匹配指纹即可，无需重启服务。

已有环境可独立补齐默认参数；非默认网格使用显式维度，命令见 [README](README.md)：

```powershell
.\.venv\Scripts\python.exe scripts/setup_lego_parameters.py --runtime runtime --defaults
```

三机应先在一台机器完成准备，再将同组公共 PK/VK、清单和指纹分发到各自运行目录，让其余部署复用；不要分别随机生成三组参数。

Python、CPU Torch 与 npm 下载策略位于 `scripts/deployment_downloads.py`：默认优先国内镜像，失败后逐个回退。`DGFL_PIP_INDEX_URL`、`DGFL_TORCH_INDEX_URL`、`DGFL_NPM_REGISTRY` 可覆盖对应源，不改写用户或系统配置。数据集显式下载源仍只尝试所选地址。

## 完整离线部署材料

在与目标相同 Python 实现、次版本、操作系统和架构的已部署机器上执行。完整部署已补齐该机器目标 `runtime` 的默认参数，下例将其公共 CRS 随包分发；不携带参数时省略 `--parameters-runtime runtime`：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --parameters-runtime runtime
.\.venv\Scripts\python.exe scripts/prepare_full_offline.py --output full-offline --verify-only
.\.venv\Scripts\python.exe scripts/package_submission.py --output dist/submission-full --full-offline-path full-offline
```

`prepare_full_offline.py` 冻结当前环境的完整依赖闭包，准备 CPU Torch/torchvision、受支持平台的 NVRTC、当前原生 wheel 及源码凭据、两套经过校验的数据和匹配的 `web/dist`。`--wheelhouse` 可指定完整、精确的本地 wheel 集合以免再次下载；可用 `--native-wheelhouse` 选择一个带当前源码凭据的原生 wheel。输出目录必须尚不存在，已有目录使用 `--verify-only` 复查。

公开 Lego 参数默认不随包分发。上例使用 `--parameters-runtime runtime` 纳入现有 650/1,930 维、8 位参数的 PK/VK 与清单；保留开发用单方设置来源，不包含设置秘密或节点身份。携带时，目标恢复并复用相同指纹；不携带时，目标完整部署可离线本地补齐默认 CRS。非默认维度仍需显式准备，三机同一实验须共享同组公共参数。明文基线始终无需 CRS。

目标先用系统 Python 运行包内 `full-offline/VERIFY.py` 核对文件大小、SHA-256 和目标平台，再按 `full-offline/INSTALL.txt` 创建环境、从本地 wheel 安装并执行深度核验和 `--restore-assets`，最后离线启动。恢复只补公共数据、前端、原生材料及所选公共参数，遇到冲突会拒绝，不覆盖既有身份。系统 Python 及系统运行库由目标机器提供；GPU 另需兼容的 NVIDIA 驱动。使用预构建材料不要求目标安装 Node.js、Rust 或 C++ 构建工具。

源码包和完整材料分开保存：`freeze_source.py --source-only` 不纳入完整离线包、大数据、环境或构建结果。`package_submission.py --full-offline-path` 才会重新检查完整材料的依赖闭包、数据、源码绑定及精确清单，并统一放入 ZIP 的 `full-offline/`。原 `--include-offline` 保持基础包语义。

`scripts/prepare_offline.py` 仍是基础 NumPy + MNIST 材料工具，不包含完整启动所需的 Torch、GPU、当前原生扩展和 CIFAR-10，也不包含模型专用 CRS。只安装这个旧式基础包可用于明文最小演示，不足以通过完整部署检查或运行新加密实验。

## 验证与发布

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
Push-Location web
npm.cmd test
npm.cmd run build
Pop-Location
```

测试依赖需开发者另行安装，例如 `python -m pip install -e ".[test]"`；应用运行不依赖 pytest 或 Ruff。历史 P1/P2 验收保留在 `docs/research/evidence/p1p2-fixes-20261008.json`。

设计报告第 7 节和答辩稿中的 12 角色、旧证明、固定批次及 3.04 倍轮时比属于冻结历史配置。`build_report.py --freeze-evidence` 仅重建该历史章节，并拒绝不同运行源码指纹；它不会启动新实验，也不能把当前 Lego、四云及并行默认套上旧硬件和旧测试结论。当前版本性能应另行完成匹配实测并形成新报告。

生成纯源码包时，先完成相关测试，然后运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/freeze_source.py --source-only --release-id dgflow-review-improvements-20261011 --archive dist/source-release/dgflow-review-improvements-20261011/DGFlow-review-improvements-20261011.zip
.\.venv\Scripts\python.exe -B scripts/freeze_source.py --verify
```

`--source-only` 排除 `web/dist`、离线材料、历史 Git bundle、数据集、运行目录、环境与构建缓存。需要包含前端构建结果的原有交付模式时，先构建前端并省略此选项。冻结工具会生成新的 SOURCE-MANIFEST.json；运行时数据、依赖环境和构建缓存不应加入源码版本控制。
