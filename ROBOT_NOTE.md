# Cosmos3-Nano DROID 后训练 + (后续) RoboTwin 评测 笔记

本文记录在本机用 Cosmos3-Nano 复现 DROID action policy 后训练的全过程，以及为后续 RoboTwin 闭环评测做的环境规划。路径以当前机器为准。

> **当前进度**：正在复现 Cosmos3-Nano 的 **DROID 训练**。RoboTwin 评测是**后续目标，尚未开始**。

---

## 0. 总目标与路线

最终想做：用 Cosmos 作基模 → 用机器人数据训一个 policy → 用 RoboTwin 闭环评测。

拆成两段，**当前在第 1 段**：

1. **【进行中】DROID action policy 后训练**：基于 `Cosmos3-Nano`，在 `Cosmos3-DROID` 数据上后训练出动作策略模型。
2. **【后续】RoboTwin 评测**：把训好的 policy 接进 RoboTwin 做闭环。

---

## 1. 环境与硬件

```text
repo:        /pfs/pfs-7jnepv/shukaigong/code/cosmos-framework
venv:        /pfs/pfs-7jnepv/shukaigong/code/cosmos-framework/.venv  (uv sync --all-extras --group=cu130-train)
激活:        source .venv/bin/activate && export LD_LIBRARY_PATH=
```

- Python 3.13 / torch 2.10.0+cu130 / transformers 4.57.6 / hf_hub 0.36.2 / numpy<2.3
- 硬件：8× NVIDIA B20Z（Blackwell，~183GB/卡）；2.8TB 内存；180 核
- 系统 CUDA：`/usr/local/cuda-12.8` 和 `/usr/local/cuda-13.0` 都在

---

## 2. RoboTwin 环境规划（后续段，重要结论先记下）

**结论：不要把 RoboTwin 装进 cosmos 的 `.venv`，既不可行也没必要。用进程分离（server/client）。**

- **不可行**：cosmos venv 是 py3.13；RoboTwin 核心二进制依赖 `sapien==3.0.0b1 / mplib==0.2.1 / open3d==0.18.0` 在 PyPI 上**最高只有 cp312 wheel**，py3.13 装不上。且 RoboTwin pin 的 `scipy 1.10.1 / huggingface_hub 0.25.0` 会和 cosmos 栈冲突。
- **没必要**：cosmos 本身就为进程分离设计，自带 OpenPI 协议的 WebSocket policy server：
  - `cosmos_framework/scripts/action_policy_server_robolab.py`（cosmos 在此进程加载，websocket 吐 action）
  - 文档 `docs/action_policy_droid_server.md`；依赖组 `pyproject.toml` 的 `policy-server`（`openpi-server`）
  - RoboTwin 侧也有 TCP 的 `RoboTwin/script/policy_model_server.py` + `eval_policy_client.py`
- **推荐架构**：cosmos venv 跑 policy server；**另建一个 py3.10 conda env** 按官方流程装 RoboTwin（参考 FastWAM `NOTE.md` 第 4 节：Vulkan、注释掉 requirements 的 torch、预编译 pytorch3d、curobo 用 CUDA 12.8 编译、SAPIEN/MPLib patch、OIDN 升级、assets/task_config 链接），两者用 socket 通信。
- 缺的胶水：`RoboTwin/policy/` 下要新建一个 `cosmos_policy/`，实现 RoboTwin 的 `get_model/eval/reset_model`，内部用 websocket client 连 cosmos server，并对齐 obs/action 格式（参考 `RoboTwin/policy/pi0`）。

---

## 3. DROID 数据集性质

格式：**LeRobotDataset v3.0**，已下载到：

```text
/pfs/pfs-7jnepv/shukaigong/data/cosmos3-droid/        (749G)
├── success/        ← 训练用这个 split (DROID_ROOT)
│   ├── meta/info.json        robot_type=panda, fps=15, total_episodes=57639
│   ├── data/chunk-000/*.parquet     (62 个 parquet，每个聚合上千 episode)
│   └── videos/<camera>/chunk-*/*.mp4 (3063 段，360x640 av1)
├── failure/
└── keep_ranges_1_0_1.json    ← 窗口过滤器 (见 §6)
```

一个 parquet 不是一帧一行；如 `file-000.parquet` 有 313318 行、1216 个 episode。视频帧不在 parquet，单独按时间戳对齐。

### 三路相机
- `observation.image.wrist_image_left`（腕部）
- `observation.image.exterior_image_1_left`（左肩）
- `observation.image.exterior_image_2_left`（右肩）

recipe 用 `concat_view`：腕部相机放上半幅（480p），左右肩各缩半拼下半幅。

### parquet 列含义（panda = 7 关节 + 1 夹爪；笛卡尔 = 3 平移 + 3 旋转）

| observation.state.* | 维度 | 含义 |
|---|---|---|
| joint_positions | 7 | 当前关节角(rad) |
| joint_velocities | 7 | 关节角速度 |
| joint_torques_computed | 7 | 计算关节力矩 |
| motor_torques_measured | 7 | 实测电机力矩 |
| cartesian_position | 6 | 末端位姿(xyz+旋转) |
| gripper_position | 1 | 夹爪开合(0张开…1闭合) |

| action.* | 维度 | 含义 |
|---|---|---|
| joint_position | 7 | 目标关节角(绝对) |
| joint_velocity | 7 | 目标关节速度 |
| cartesian_position | 6 | 目标末端位姿 |
| cartesian_velocity | 6 | 目标末端速度 |
| gripper_position | 1 | 目标夹爪开合 |
| gripper_velocity | 1 | 夹爪速度 |

元信息：`timestamp/frame_index/episode_index/index/task_index`（task_index → `meta/tasks.parquet` 的语言指令）。

### 本次训练实际用到的列（recipe: `action_space="joint_pos"` + `use_state=True`）

只取 **4 列**，其余忽略（velocity/torque/cartesian 都没用）：
- **监督目标(label)**：`action.joint_position`(7) + `action.gripper_position`(1) → 8 维绝对关节动作；夹爪翻转 `1-g`；预测长度 **32 的 action chunk**。
- **状态条件(输入)**：`observation.state.joint_positions`(7) + `observation.state.gripper_position`(1) → 8 维初始状态，prepend 到动作序列前（→ 33 帧）。
- **视觉条件**：三路相机 concat 成 480p。
- **语言条件**：`task_index` 对应文本。
- joint_pos 模式**不归一化**（`action_normalization=None`）；动作 pad 到 `max_action_dim=64`（多 embodiment 共用头）。

---

## 4. 训练所需的三个外部输入（已全部就位）

| 输入 | 路径 | 说明 |
|---|---|---|
| Cosmos3-Nano 基座 | `/pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano` | 33G，HF 原始权重 |
| Cosmos3-Nano DCP | `/pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano-dcp` | 29G，转换产物，训练实际加载这个 |
| Wan2.2 VAE | `/pfs/pfs-7jnepv/shukaigong/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2.8G，视频 latent 编码器 |
| 窗口过滤器 | `/pfs/pfs-7jnepv/shukaigong/data/cosmos3-droid/keep_ranges_1_0_1.json` | 22M，可选，见 §6 |

下载命令（HF_HOME 放 pfs，断点续传）：
```bash
export HF_HOME=/pfs/pfs-7jnepv/shukaigong/hf_home
hf download nvidia/Cosmos3-Nano
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir <dir>   # 只下这一个文件，别下整个 34G repo
hf download KarlP/droid keep_ranges_1_0_1.json --local-dir <dir>
```

---

## 5. 训练流程（已跑通）

### 配置是两层

- **运行级 TOML**：`examples/toml/sft_config/action_policy_droid_repro.toml` —— 只放 run-level 标量（iter 数、存盘频率、并行、wandb、VAE 路径）。**batch/lr/dataset 参数不在这里。**
- **注册 experiment**：`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py` —— recipe 真正 knobs：
  - `optimizer.lr=2e-4`（为 global batch 8192 调的）；只训 `moe_gen/time_embedder/vae2llm/llm2vae/action2llm/llm2action/action_modality_embed`；动作头有 `lr_multipliers=5.0`
  - `dataloader_train.max_samples_per_batch=128`（per rank）
  - dataset 全部参数（chunk_length=32, action_space=joint_pos, use_state, concat_view, 480p, iterable_shuffle...）
  - 动作头 fresh init：`checkpoint.keys_to_skip_loading` 跳过 `action2llm/llm2action/action_modality_embed/action_pos_embed` 和 `net_ema.`
- TOML 标量 + 命令行 `--` 之后的 tail override 都叠在 experiment 上；**tail override 最后生效、覆盖 TOML**（`scripts/train.py:281`）。

### batch 的关键认知

- 没有传统 `batch_size`；样本是变长 token 序列打包，用 `max_samples_per_batch`（每 rank 每步打包多少 sample）表达。
- **全局 batch = max_samples_per_batch × world_size × grad_accum_iter**
  - 官方参考 64 卡：128 × 64 × 1 = **8192**
  - 本机单机 8 卡、batch 32：32 × 8 × 1 = **256**（是参考的 1/32）
- override 路径：`dataloader_train.max_samples_per_batch=<N>`
- **lr 要配 batch**：参考 lr=2e-4 对应 8192；本机 256 时按 sqrt 缩放 ≈ `2e-4 × sqrt(256/8192) ≈ 3.5e-5`，故加 `optimizer.lr=3.5e-5`。

### 复现步骤

```bash
# Step 1: 准备 Cosmos3-DROID success split (已下)
# Step 2: 转 DCP 基座 (本地目录可直接吃，不联网重下)
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path /pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano \
  -o /pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano-dcp
# Step 3: (可选) 下 keep_ranges 过滤器
# Step 4: 设环境变量 + 启动 (见下方脚本)
```

启动脚本 `scripts/train_droid.sh`（**必须在 repo 根目录跑**，因为脚本内有相对路径）：
```bash
cd /pfs/pfs-7jnepv/shukaigong/code/cosmos-framework
bash scripts/train_droid.sh
```
关键环境变量：
```bash
DATASET_PATH=/pfs/.../cosmos3-droid/success   # launcher 桥接成 DROID_ROOT
DROID_ROOT=$DATASET_PATH
BASE_CHECKPOINT_PATH=/pfs/.../weights/Cosmos3-Nano-dcp
WAN_VAE_PATH=/pfs/.../weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
NPROC_PER_NODE=8
EXTRA_TAIL_OVERRIDES="<keep_ranges 开关> dataloader_train.max_samples_per_batch=32 optimizer.lr=3.5e-5"
```

### smoke 验证（先跑这个确认链路）
临时加 tail override，几分钟跑完 10 步：
```bash
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32"
bash examples/launch_sft_action_policy_droid.sh
```
- smoke 想更快可加 `model.config.compile.enabled=False`（跳过编译）；**正式跑保留 compile=True**（首步编译 10–20 分钟，分摊到上万步划算）。
- `HF_HUB_OFFLINE=1` 可避免本机到 huggingface.co 不稳（SSL EOF）导致的网络卡顿（tokenizer 已缓存后用）。

### 实测结果（已跑通）
- smoke 10 步 exit 0，checkpoint 存盘成功。
- **首步 ~1105s（torch.compile 编译），之后每步 ~52–77s @ batch 32/rank。**
- ⚠️ 单机 8 卡跑满 10000 步 ≈ 9 天，且 batch 是参考 1/32、只看 ~1/32 数据，**不可能复现官方 released 质量**。按目标决定 `trainer.max_iter`：只要能用的 policy → 调小到 2000–3000 先出一版。

### 诊断技巧
- "卡住"先看：`nvidia-smi`（util）、`ps -o %cpu`（每 rank 100% 单核 = CPU-bound = 多半在 compile）、日志 tail。
- 抓栈：`sudo env "PATH=$PATH" .venv/bin/py-spy dump --pid <rank0_pid>`（本机 sudo 要密码）。栈里有 `_dynamo/_inductor/fx` = 正在编译，等着即可。

---

## 6. keep_ranges_1_0_1.json 是什么

**训练窗口白名单**：告诉 dataloader 每条轨迹哪些帧段值得训，丢掉发呆/无效帧。

- 结构：dict，95658 条轨迹；key = DROID 原始 `gs://...trajectory.h5` 路径，value = 一组 `[start,end]` 帧区间。例：`[[0,19],[38,159],[177,274]]`。
- 用法（`droid_lerobot_dataset.py:174-199, 228-241`）：`use_filter_dict=True` 时只在保留区间内开窗；dict 里没有的 episode 整条丢弃 → 训练精选 ≈74% 窗口（对齐官方）。
- smoke 可不用；对齐官方质量就开（`scripts/train_droid.sh` 已开）。

---

## 7. 模型输入输出

**核心认知**：Cosmos3-Nano-Policy 不是普通 action 回归模型，而是**"世界模型+动作"联合生成模型**：`vision_gen=True` 且 `action_gen=True`，用 **rectified flow / flow matching** 同时建模未来视频和未来动作。所以 loss 有 vision 项（`loss_scale=10`）和 action 项（`action_loss_weight=10`）两块——绝对值 5–25 是被 ×10 撑上去的，不是异常。

### 训练时
一个样本 dict（`base_dataset.py:204` `_build_result`）：
| 字段 | 来源 | 形状 |
|---|---|---|
| `video` | 三路相机 concat_view | `[3, 33, H, W]` uint8 |
| `action` | joint_position(7)+gripper(1)，夹爪翻转 | `[33, 8]`：第0行=初始状态，后32行=动作chunk |
| `ai_caption` | task_index→文本 | str |
| `conditioning_fps` | 15 | 标量 |
| `domain_id` | embodiment 域 | 标量 |
| `mode` | "policy" | |

流程：`video` 经 Wan2.2 VAE 编码成 latent，`action` 经 `action2llm`(8→hidden) 投进 token 空间，文本进 Qwen3-VL；对 video latent 和 action 各自加噪，MoT(`moe_gen`) 预测速度场；loss = vision FM×10 + action FM×10；backbone 冻结，只训生成头+动作头。

### 推理/部署时（后续 RoboTwin 评测用，`action_policy_server_robolab.py:490-568`）
**输入**（机器人/仿真每步发的 obs）：
- `prompt`：任务指令
- 当前相机 RGB → `video[:,0]`，其余 32 帧置零
- `observation/joint_position`(7)+`observation/gripper_position`(1) → `action[0]` 初始状态
- 组装成与训练同构的 sample

**输出**（`generate_samples_from_batch`）：
- `action`：`[32, 8]` —— 未来 32 步绝对关节动作（7 关节 + 1 夹爪）；可整段执行或 receding-horizon。
- 可选 `video`：模型顺带生成的未来帧（闭环可不用）。

**一句话**：给「当前图 + 当前关节/夹爪 + 语言指令」→ 生成「未来 32 步怎么动关节和夹爪」。

⚠️ 对接 RoboTwin 时：训练用的是 DROID 的 **8 维 panda joint_pos 动作空间**（绝对角度，夹爪 `1-g`）。RoboTwin 机器人若非 panda 7-DoF joint、或夹爪定义不同，需在适配层做映射。

---

## 8. RoboTwin 数据准备（merge → LeRobot v2.1 → v3.0）

为后续在 **RoboTwin（aloha-agilex 双臂）数据**上做训练/对齐，把原始 RoboTwin HDF5 数据整理成 cosmos 能吃的 **LeRobot v3.0** 格式。任务样例：`place_a2b_left`（把 A 放到 B 左边）。三步流水线，脚本都放在 `RoboTwin/script/` 下可复用。

最终目录（`/pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/`）：
```text
aloha-agilex_clean_50/              原始 HDF5，50 条（干净场景）
aloha-agilex_randomized_500/        原始 HDF5，500 条（随机化场景）
aloha-agilex_combined_550/          ① merge 后的 HDF5，550 条
aloha-agilex_combined_550_lerobot_v2.1/   ② 转 LeRobot v2.1
aloha-agilex_combined_550_lerobot_v3.0/   ③ 转 LeRobot v3.0（训练用这个）
```

### ① 合并 clean + randomized → 550 条 HDF5

脚本：`RoboTwin/script/merge_datasets.py`（自写，通用可复用）。

```bash
cd /pfs/pfs-7jnepv/shukaigong/code/RoboTwin
python script/merge_datasets.py     # 默认即下面的 50+500 -> 550
# 通用：python script/merge_datasets.py --sources A B C --output OUT --mode hardlink|copy|symlink
```

做了什么：
- 按顺序拼接：`clean_50` → episode **0–49**，`randomized_500` → episode **50–549**，连续重编号无空缺。
- 自动发现 `episode<N>.<ext>` 子文件夹（`data/.hdf5`、`instructions/.json`、`_traj_data/.pkl`、`video/.mp4`）并整体重编号；同步重写 `scene_info.json`（键 `episode_<N>`)与 `seed.txt`（按 episode 位置）。
- 默认 **hardlink**（同一文件系统，瞬间完成、不占额外空间）；数据只读用,安全。已抽查内容映射正确（combined ep50 == randomized ep0 等）。

### ② HDF5 → LeRobot v2.1

脚本：`RoboTwin/script/convert_robotwin_to_lerobot.py`（自写）。**用 `fastwam-robotwin` 这个 conda env 跑**（有 h5py/ffmpeg/PIL，且与 cosmos venv 隔离）。

```bash
cd /pfs/pfs-7jnepv/shukaigong/code/RoboTwin
python script/convert_robotwin_to_lerobot.py \
  --src /pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550 \
  --dst /pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v2.1 \
  --robot-type aloha-agilex \
  --workers 16
```

做了什么 / 关键映射：
- **输入**：`data/episode{N}.hdf5`(`joint_action/vector` 形状 `(T,14)`，`observation/{head,left,right}_camera/rgb` 是 JPEG bytes) + `instructions/episode{N}.json`（取 `seen[0]` 作为该 episode 的语言任务）。
- **相机映射**：`head→cam_high`、`left→cam_left_wrist`、`right→cam_right_wrist`，三路各编码成 h264/yuv420p mp4，**30 fps**。
- **输出**(LeRobot v2.1)：`data/chunk-000/episode_{N:06d}.parquet` + `videos/chunk-000/<cam>/episode_*.mp4` + `meta/{info,tasks,episodes,episodes_stats}.jsonl/json`；`info.json` 里 `codebase_version=v2.1, fps=30, robot_type=aloha-agilex`。
- **`--workers 16`** 用 `ProcessPoolExecutor` 并行处理 episode；`--dst` 若已存在会**先删再写**。
- **action_dim=14**：aloha-agilex 双臂 = 2×(6 关节 + 1 夹爪)。与 DROID 的 8 维 panda 动作空间不同(见 §7 适配告警)。

⚠️ **两个要记住的简化**(转换脚本现状)：
1. **`observation.state` 与 `action` 是同一份 `joint_action/vector` 的拷贝**——源数据没单独的本体感知 state 列,这里直接复用动作向量当 state。若 cosmos recipe 需要"真·当前状态 ≠ 目标动作",这里要改。
2. **图像 stats 全写 0**(`_image_zero_stats` 占位)；v2.1→v3.0 的 `aggregate_stats` 会沿用这些零值。图像归一化一般在别处算,通常无碍,但需知情。

### ③ LeRobot v2.1 → v3.0

用 lerobot 自带转换器，**在 cosmos 的 `.venv` 里跑**（lerobot 装在这）。

```bash
source /pfs/pfs-7jnepv/shukaigong/code/cosmos-framework/.venv/bin/activate && export LD_LIBRARY_PATH=
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id aloha-agilex_combined_550_lerobot \
  --root /pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left \
  --push-to-hub false
```

⚠️ **路径大坑**：转换器内部是 `root = Path(--root) / --repo-id`（`convert_dataset_v21_to_v30.py:472`）。所以 **`--root` 必须是父目录、`--repo-id` 必须是数据集文件夹名**，二者拼起来才是真实数据集路径。若把完整路径塞给 `--root`、`--repo-id` 另给一个名字（如 `local/place_a2b_left`），它会去拼一个不存在的子路径→判定非本地→转去 HuggingFace Hub 下载→报错退出（数据不受损但不转换）。

做了什么：
- 校验是 v2.1 → 把每 episode 的 parquet 按 ~100MB 合并成 `data/chunk-000/file_*.parquet`；每相机 mp4 按 ~500MB 合并成 `videos/<cam>/chunk-000/file_*.mp4`(带 from/to 时间戳)。
- 重建元数据为 parquet：`meta/episodes/...parquet`、`meta/tasks.parquet`、`meta/stats.json`；`info.json` 升 `codebase_version=v3.0`。
- **原地替换**：原 v2.1 目录被 `mv` 成 `..._old` 备份,新 v3.0 落到原路径。本流程最终手动把两者重命名为带 `_v2.1` / `_v3.0` 后缀以便区分。
- 需要 `ffmpeg`(已确认可用)；`--push-to-hub false` 不碰 Hub。

> ~~**TODO（接训练）**~~：**已完成,见 §8.5。**

---

## 8.5 RoboTwin 训练接入 Cosmos3（place_a2b_left 验证实验）

目标：仿 DROID,用 RoboTwin place_a2b_left 的 v3.0 数据训一个 **联合"未来视频+未来动作"** 的 Cosmos3-Nano-Policy-RoboTwin。命名一律用 `robotwin`（后续加别的 robotwin 任务可复用）。

### 关键前提：v3.0 可直接复用 cosmos 基类
你转的 v3.0(`.../aloha-agilex_combined_550_lerobot_v3.0`)和 DROID 的 v3.0 **schema 对齐**:`meta/episodes` 带齐 `videos/<key>/{from_timestamp,chunk_index,file_index}`、`data/{chunk,file}_index`、`length`、`tasks`;`data/file-*.parquet` 聚合 550 ep / 82387 帧,列 `observation.state`(14)/`action`(14)/index/episode/task/timestamp/frame。所以 `ActionBaseDataset` 原样能读,**不动 base_dataset.py**。

### 🔴 tasks.parquet 坑（每次转 v3.0 都会遇到）
lerobot v2.1→v3.0 转换器把任务文本写成无名 pandas index 列 `__index_level_0__`,而基类读 `row["task"]`(`base_dataset.py:73`)→ **会崩**。修法(内容不变,只改列名):
```bash
python - <<'PY'
import pyarrow.parquet as pq
p="<v3.0>/meta/tasks.parquet"; t=pq.read_table(p)
if "task" not in t.column_names:
    names=["task" if c=="__index_level_0__" else c for c in t.column_names]
    pq.write_table(t.rename_columns(names).select(["task_index","task"]), p)
PY
```
> 建议在你的转换流程里直接把该列命名为 `task`,一劳永逸。

### 建模决策（对齐 DROID,改成 aloha）
- `domain_name="robotwin"`,新 `domain_id=16`;`action_dim=14`,`max_action_dim=64` pad。
- action_spec = `Joint(6,left)+Gripper(left)+Joint(6,right)+Gripper(right)` = 14 维。
- `use_state=True`:窗口 chunk+1 帧,row0=初始 `observation.state`,rows[1:]=`action` chunk。
- `action_normalization=None`(原始关节角,不归一化);**夹爪不翻转**(DROID 翻了 `1-g`;这里直接出原始 14 维,方便后续接 RoboTwin 执行)。
- `viewpoint="concat_view"`:head(cam_high)整幅在上,left/right wrist 各缩半拼下半幅。
- `chunk_length=32`、fps=30 → `encode_exact_durations=[33]`;`resolution="480"`(源 320×240 会上采样,可降 256)。
- 联合视频+动作:`vision_gen=True`+`action_gen=True`,`loss_scale=10`+`action_loss_weight=10`(同 DROID)。无 keep_ranges。

### 新增/改动文件清单
**新增**
1. `cosmos_framework/data/vfm/action/datasets/robotwin_lerobot_dataset.py` — `RoboTwinLeRobotDataset`(14 维 joint,3 相机,复用基类 `_rows/_episodes/_tasks`,82k 帧无需 DROID 的 compact 优化)。
2. `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py` — experiment,镜像 `action_policy_droid_nano`。
3. `examples/toml/sft_config/action_policy_robotwin.toml`
4. `examples/launch_sft_action_policy_robotwin.sh`(桥接 `DATASET_PATH→ROBOTWIN_ROOT`)
5. `scripts/train_robotwin.sh`

**增量改动**(DROID 路径不受影响)
6. `cosmos_framework/data/vfm/action/domain_utils.py` — `robotwin`: domain_id 16 / raw_action_dim 14
7. `cosmos_framework/data/vfm/action/datasets/__init__.py` — 导出 `RoboTwinLeRobotDataset`
8. `cosmos_framework/data/vfm/action/datasets/action_sft_dataset.py` — `get_action_robotwin_sft_dataset`
9. `cosmos_framework/configs/base/config.py` — 注册导入新 experiment

### 验证（已通过,未碰运行中的 DROID）
✅ 导入 / action_spec=14 维 / domain 16 / experiment 注册 / dryrun 配置解析 / **dataset 单测** `ds[0]` → `video (3,33,360,320) uint8`、`action (33,14)`、`len(ds)=64787`。GPU smoke 因当时 8 卡被 DROID 占满,留给后面跑。

### 跑法
```bash
# smoke（GPU 空闲后；先确认 tasks.parquet 已修）
cd /pfs/pfs-7jnepv/shukaigong/code/cosmos-framework
source .venv/bin/activate && export LD_LIBRARY_PATH= && export HF_HUB_OFFLINE=1
export DATASET_PATH=/pfs/pfs-7jnepv/shukaigong/data/robotwin2.0/place_a2b_left/aloha-agilex_combined_550_lerobot_v3.0
export ROBOTWIN_ROOT=$DATASET_PATH
export BASE_CHECKPOINT_PATH=/pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano-dcp
export WAN_VAE_PATH=/pfs/pfs-7jnepv/shukaigong/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export NPROC_PER_NODE=8
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32 model.config.compile.enabled=False"
bash examples/launch_sft_action_policy_robotwin.sh

# 正式训练（smoke 通过后）
bash scripts/train_robotwin.sh   # 已配 batch 32 + lr 3.5e-5、无 keep_ranges、wandb 超时放宽
```

> 待调 knob:夹爪翻转、resolution(480 vs 256)、batch/lr。RoboTwin 闭环评测仍是后续阶段。

---

## 9. 杂项 / 坑

- `scripts/train_droid.sh` 必须 `cd` 到 repo 根再跑（脚本含相对路径 `source .venv/...`、`bash examples/...`）。
- `WANDB_API_KEY` 一度明文写进脚本 → 别 `git add`，建议 rotate，改从环境变量/`~/.netrc` 读。
- checkpoint 每个 ~32G；正式跑每 1000 步存一次。`IMAGINAIRE_OUTPUT_ROOT` 决定落盘位置（不设默认 `<repo>/outputs/train`）。
- 导出 HF safetensors：`cosmos_framework.scripts.export_model`。

---

## 10. RoboTwin 闭环评测设施（server/client 进程分离）

### 为什么不能 in-process
robotwin conda env = **py3.10 / torch 2.4.1+cu121 / sapien 3.0.0b1**；cosmos `.venv` = **py3.13 / torch 2.10+cu130**，二者无法同进程。FastWAM 能 in-process 是因为它的依赖和 robotwin 兼容；cosmos 不行 → **必须 server(cosmos)/client(robotwin) 分离**，走 socket。

### 架构 & 新增文件
```
Cosmos 推理 server (cosmos .venv)        socket        RoboTwin eval client (robotwin env)
 加载训好的 RoboTwin 权重                  ◄──────►      script/eval_policy.py 驱动 sapien sim
 obs(3图+14维state+prompt)→action[32,14]   JSON+np      policy/cosmos_policy 包装层
```
- **server**：`cosmos_framework/scripts/action_policy_server_robotwin.py`（复用 robolab 的 `OmniInference.create` 加载 + `ActionTransformPipeline`；**复刻训练预处理**：3 相机 concat_view、res480、14 维 state 行、domain robotwin、**不翻夹爪**、不归一化）。纯 TCP（4 字节长度前缀 + JSON，ndarray base64），所以 robotwin 侧零 torch/openpi 依赖。
- **client**：`RoboTwin/policy/cosmos_policy/{__init__.py, deploy_policy.py, deploy_policy.yml, eval.sh}`（实现 `encode_obs/get_model/eval/reset_model`；socket 连 server；receding-horizon 执行 `replan_steps` 步后重推理；`take_action(a,"qpos")`）。
- **launcher**：`scripts/serve_robotwin_policy.sh`（cosmos 侧起 server）；`scripts/smoke_robotwin_client.py`（无 sim 的合成 obs 冒烟）。

### 🔴 关键坑
- **必须先 export 再 serve**：把训练 DCP（`…/action_policy_robotwin/checkpoints/iter_<N>`）用 `export_model --experiment action_policy_robotwin_nano` 导成自带 `config.json`/`checkpoint.json` 的 safetensors 目录，server 才能从 metadata 恢复 tokenizer、并用 EMA 权重。直连 DCP 是 fallback（需 `--experiment` + `ROBOTWIN_ROOT`）。
- **预处理必须和训练一致**（concat 几何、res480、14 维 state、不翻夹爪、不归一化），否则分数无意义。
- **Blackwell**：eval.sh 已导 `VK_ICD_FILENAMES`/`NVIDIA_DRIVER_CAPABILITIES`；还需把 robotwin env 的 SAPIEN bundled OIDN 升到 2.3.3（见 FastWAM NOTE §5），否则分数虚低。
- **`policy_name=cosmos_policy`** 靠 `eval_policy.py` 的 `sys.path.append("./policy")` + 包内 `__init__.py` 的 `from .deploy_policy import *` 解析。
- **夹爪约定**是最可能的首跑 bug：训练时不翻；若 sim 里开合反了，在 client 翻一下即可（无需重训）。

### 跑法（待有 checkpoint 后）
```bash
# 1) cosmos 侧 export（每个 ckpt 一次）
ITER=…/action_policy_robotwin/checkpoints/iter_000000100
PYTHONPATH=. python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$ITER" --experiment action_policy_robotwin_nano \
  -o /pfs/pfs-7jnepv/shukaigong/weights/Cosmos3-Nano-Policy-RoboTwin-iter100
# 2) cosmos 侧起 server（挑空闲 GPU）
SERVE_GPU=0 CKPT=/pfs/.../Cosmos3-Nano-Policy-RoboTwin-iter100 bash scripts/serve_robotwin_policy.sh
# 3) 冒烟（可选，无 sim）
python scripts/smoke_robotwin_client.py --port 9876     # 期望 action (32,14) finite
# 4) robotwin env 真评测
cd /pfs/pfs-7jnepv/shukaigong/code/RoboTwin
bash policy/cosmos_policy/eval.sh place_a2b_left demo_clean iter100 0 0
```

### 实跑遇到并已解决的 2 个坑
1. **server 加载报缺 VAE**：`--experiment` 加载时不应用训练 TOML 的 `WAN_VAE_PATH` 覆盖，默认 vae_path 是 registry 路径 → 下载失败。已在 server 加 `--vae-path`(默认指向本地 `Wan2.2_VAE.pth`)+ 注入 `model.config.vlm_config.pretrained_weights.enabled=False` / `diffusion_expert_config.load_weights_from_pretrained=False`(冻结子权重从 DCP 出)。
2. **robotwin env 跑不了 Blackwell**：用户新建的 `robotwin` conda env 是 **torch 2.4.1+cu121(最高 sm_90)**,B20Z 是 sm_100 → 任何 CUDA 算子(含 curobo `normalize_quaternion`)报 “no kernel image available”,`CuroboPlanner` import 失败。**curobo 本身没装错**,是 torch 太老。修法:对齐能跑的 `fastwam-robotwin`——升 `torch==2.7.1+cu128`+`torchvision==0.22.1+cu128`、pytorch3d 重装 `0.7.8+pt2.7.1cu128`、curobo 用 CUDA 12.8 + `TORCH_CUDA_ARCH_LIST="9.0;10.0"` 重编(`/pfs/.../RoboTwin/envs/curobo`)。sapien/mplib/numpy/warp 两环境一致、不用动。

### 验证状态(✅ 端到端跑通)
✅ server 导入/wire codec；✅ client py3.10 编译；✅ 直连 DCP(iter_000000100)加载成功(62s,GPU7 ~33–41GB);✅ 合成 obs 冒烟 `action (32,14)` finite;✅ **真 sim 闭环**:`bash policy/cosmos_policy/eval.sh place_a2b_left demo_clean iter100 0 1` → client 连上 server、跑完 400 步 episode、`take_action(qpos)` 执行、成功率正常计算(iter_100 欠训,分数仅验证链路)。server 推理 UniPC 4 步 ~亚秒/次。
> 真评测应在训练充分的 checkpoint 上做(iter_100 仅为链路验证)。`test_num=100` 在 `eval_policy.py:162` 硬编码,想少跑几个 episode 需改它。共享机器上可能有他人的 eval 进程,`pkill -f eval_policy.py` 会误伤,慎用。
