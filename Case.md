# 分布式RLHF训练系统故障注入实验方案

## 实验目标

证明分布式RLHF训练系统中存在三个独立的技术挑战：

**挑战A：架构导致的故障耦合（Fault Coupling）**

- 一个阶段的硬件故障在另一个阶段产生错误日志
- 由单控制器、共卡模式、SPMD等架构必然导致

**挑战B：故障检测和定位的困难性（Fault Diagnosis Difficulty）**

- 错误日志具有误导性，与真实根因不一致
- 超长超时、cleanup错误掩盖原始故障

**挑战C：RL特有的故障模式（RL-Specific Fault Patterns）**

- 在传统SFT训练或单纯推理中不存在的故障模式
- 源于RL训练的多阶段协作架构（rollout → train → weights_update）

---

## 场景1：权重同步网络故障导致双阶段超时（挑战A + C）

### 故障描述

对应FO.md场景2：NCCL/RDMA权重同步网络故障耦合

### RL独有性

**为什么在传统训练中不存在？**

- 传统SFT训练：只有train workers，没有rollout workers，权重不需要跨异构系统同步
- 传统推理：推理系统独立运行，不需要从训练系统实时获取权重更新
- RL训练特有：train workers训练完成后必须将权重同步到rollout workers进行新的样本生成，形成闭环

**架构依赖**：

- Train workers（32个GPU，TP=4, PP=2, DP=4）训练actor模型
- Rollout workers（16个GPU）使用更新后的weights进行推理生成新样本
- MetaServer（在Controller进程中）协调权重传输
- 并行执行：`actor.upload_weights()` 和 `rollout.update_weights()` 同时进行

代码位置：`examples/grpo_trainer.py:353-368`

### 故障注入方案

**注入位置**：Train Worker 5（共32个train workers，编号0-31）

**注入故障类型**：

1. **网络延迟**（注入能力5）：

   - 源：Train Worker 5
   - 目标：MetaServer所在的Controller节点
   - 基础延迟：1000ms，抖动：500ms
   - 持续时间：整个weights_update阶段（约60秒）
   - 指定端口：MetaServer端口（通常8080）
   - 网卡设备：eth0（HTTP通信网卡）

2. **可选叠加：网络丢包**（注入能力6）：

   - 源：Train Worker 5
   - 目标：MetaServer
   - 丢包率：10%
   - 指定端口：8080

### 预期收集指标

#### 系统级指标

| 指标 | 收集位置 | 正常值 | 故障时预期值 | 收集方法 |

|------|---------|--------|-------------|---------|

| **weights_update阶段总耗时** | Controller | 15-20秒 | 3600秒（超时） | `stats_tracker.record_timing("weights_update_step")` |

| **Train端upload_weights耗时** | Train Worker 0-31 | 15秒 | Worker 5: 3600秒超时<br>Worker 0-4,6-31: 15秒完成 | HTTP请求日志时间戳 |

| **Rollout端update_weights耗时** | Rollout Worker 0-15 | 18秒 | 3600秒超时（等待MetaServer数据） | HTTP请求日志时间戳 |

| **MetaServer内存占用** | Controller | 14GB（7B模型） | 0GB（未成功PUT） | `psutil.Process().memory_info().rss` |

| **网络吞吐量** | Train Worker 5 | 10 Gbps | <100 Mbps | `iftop`, `nload` |

#### Worker级细粒度指标

**Train Worker 5（故障worker）**：

| 指标 | 正常值 | 故障时值 | 收集方法 |

|------|--------|---------|---------|

| GPU利用率 | 95%（训练中）→ 0%（上传权重） | 0%（hang住，无计算） | `nvidia-smi dmon -s u` |

| GPU显存占用 | 75GB（模型+优化器） | 75GB（未释放） | `nvidia-smi` |

| TCP连接状态 | ESTABLISHED | ESTABLISHED但无数据传输 | `netstat -an \| grep 8080` |

| CPU利用率 | 5%（等待网络） | 5% | `top` |

| HTTP请求重传次数 | 0 | >1000次 | `tcpdump` + wireshark分析 |

**Rollout Worker 0-15（被动等待）**：

| 指标 | 正常值 | 故障时值 | 说明 |

|------|--------|---------|------|

| GPU利用率 | 0%（等待权重更新） | 0% | 无法区分是正常等待还是hang |

| 网络接收速率 | 5 Gbps（下载权重） | 0 Mbps（MetaServer无数据） | 关键诊断线索 |

### 日志分析：故障传播路径

**时间线**：

```
T0 (step=10, 14:30:00)
  Controller: 开始weights_update阶段
  Train Worker 0-31: 并行调用upload_weights
  Rollout Worker 0-15: 并行调用update_weights

T1 (14:30:01)
  Train Worker 5: 向MetaServer发送PUT请求，网络延迟1000ms
  Train Worker 0-4,6-31: 正常PUT，15秒完成
  Rollout Worker 0-15: 向MetaServer发送GET请求，但key不存在（Worker 5未PUT成功）

T2 (14:30:15)
  Train Worker 0-4,6-31: 日志显示"upload_weights success" ✓
  Train Worker 5: 仍在重传HTTP包

T3 (15:30:00, 3600秒后)
  Train Worker 5: HTTP超时，日志显示
    "TrainEngineError: UploadWeightsError: Upload weights request timeout"
  
T4 (15:30:02)
  Rollout Worker 0-15: HTTP超时，日志显示
    "InferenceEngineError: UpdateWeightError: Failed to update weights: NCCL timeout"
```

### 论证：故障耦合（挑战A）

**跨阶段传播路径**：

```
Train Worker 5网络延迟（根因）
    ↓
MetaServer未收到完整权重数据
    ↓
Rollout Worker 0-15获取权重失败（表象）
```

**关键论证点**：

1. **错误日志出现在Rollout端**：Rollout workers日志显示"update weights timeout"
2. **真实根因在Train端**：Train Worker 5的网络延迟
3. **从日志无法区分**：

   - Rollout端日志：可能是Rollout自身网络问题，也可能是Train端问题
   - 需要同时查看32+16=48个worker日志才能定位

4. **MetaServer作为单点**：无法区分哪一端首先故障

**指标证据**：

- Train Worker 5的网络吞吐量暴跌至<100 Mbps
- 其他31个Train Workers正常完成（15秒）
- Rollout Workers网络接收速率为0（等不到数据）

### 论证：RL特有（挑战C）

**传统训练对比**：

| 维度 | 传统SFT训练 | RL训练（本场景） |

|------|-----------|----------------|

| 权重同步目标 | 无（单阶段） | Train → Rollout（跨阶段） |

| 网络故障影响范围 | 单个worker卡住 | 两个异构系统同时卡住 |

| 诊断难度 | 低（单阶段日志） | 高（需同时分析train和rollout） |

**RL架构依赖**：

- `ThreadPoolExecutor`并行执行upload和update（`examples/grpo_trainer.py:356-365`）
- `wait_future_ordered`要求两个future都完成，一个卡住导致整体失败
- MetaServer作为协调者，其不可用影响双方

---

## 场景2：Rollout阶段CPU争用导致Train阶段数据错误（挑战A + C）

### 故障描述

对应FO.md场景5：分布式数据切分故障传播

### RL独有性

**为什么在传统训练中不存在？**

- 传统训练：数据从DataLoader读取，已经过预处理，shape和size固定
- RL训练：数据从rollout阶段动态生成，每个worker生成的数据量不确定（取决于生成的序列长度）
- DistributedBatchMemory要求所有rollout workers的数据拼接后切分给train workers，单个rollout worker失败导致数据不完整

**架构依赖**：

- Rollout workers（16个）并行生成样本，每个worker返回可变长度的序列
- Controller端`DistributedBatchMemory.split()`按FFD算法切分数据到train workers
- Train workers期望收到完整的数据batch，shape必须对齐

代码位置：

- `controller/rollout_controller.py:327-354`：rollout调用和数据拼接
- `dataset/distributed_batch_memory.py:41-58, 196-231`：数据切分逻辑
- `controller/train_controller.py:155-203`：train处理分布式batch

### 故障注入方案

**注入位置**：Rollout Worker 8（共16个rollout workers）

**注入故障类型**：

1. **CPU资源争用**（注入能力1）：

   - Worker: Rollout Worker 8
   - CPU负载：从20%提升到95%
   - 持续时间：rollout阶段（约120秒）

2. **可选叠加：内存资源耗尽**（注入能力2）：

   - Worker: Rollout Worker 8
   - 内存占用：从40GB提升到接近64GB（总内存）
   - 占用速率：1GB/秒（模拟内存泄漏）
   - 结果：Worker 8在rollout中途OOM

### 预期收集指标

#### 系统级指标

| 指标 | 收集位置 | 正常值 | 故障时预期值 |

|------|---------|--------|-------------|

| **rollout阶段总耗时** | Controller | 120秒 | 7200秒（超时）或立即失败（OOM） |

| **train阶段启动延迟** | Controller | 0秒（rollout完成后立即开始） | 7200秒或立即触发异常 |

| **rollout吞吐量（samples/sec）** | Controller | 100 samples/sec | 94 samples/sec（Worker 8慢6倍） |

| **数据batch size** | Train Controller | 1600（16 workers × 100） | 1500（缺少Worker 8的数据） |

#### Rollout Worker 8细粒度指标

| 指标 | 正常值 | 故障时值 | 收集方法 |

|------|--------|---------|---------|

| CPU利用率 | 20%（推理bound by GPU） | 95%（被注入的竞争进程占用） | `top`, `htop` |

| GPU利用率 | 85%（推理计算） | 40%（CPU成为瓶颈） | `nvidia-smi dmon -s u` |

| GPU SM占用率 | 80% | 35% | `nvidia-smi dmon -s u` |

| 推理延迟（单个sample） | 1.2秒 | 7.5秒（6倍） | Worker日志时间戳 |

| 系统内存占用 | 40GB | 63GB（接近OOM） | `free -h` |

| OOM killer触发 | 否 | 是（如果叠加内存注入） | `dmesg | grep oom` |

#### Train Workers指标（受害者）

| 指标 | 正常值 | 故障时值 | 说明 |

|------|--------|---------|------|

| 接收到的batch size | 100 samples/worker | 不对齐（有些94，有些106） | FFD算法重新分配 |

| Tensor shape验证失败 | 否 | 是（shape mismatch） | `dataset/distributed_batch_memory.py:17-28` |

### 日志分析：故障传播路径

**时间线**：

```
T0 (step=10, 14:30:00)
  Controller: 开始rollout阶段
  Rollout Worker 0-15: 并行生成样本

T1 (14:30:01)
  Rollout Worker 8: CPU负载突增至95%
  GPU利用率从85%降至40%（CPU瓶颈）

T2 (14:30:10)
  Rollout Worker 0-7,9-15: 完成生成（10秒），等待Worker 8
  Rollout Worker 8: 仍在缓慢生成（进度5/100 samples）

T3 (14:30:45)
  Rollout Worker 8: 内存占用达到63GB
  Linux OOM killer触发，杀死Worker 8进程
  日志：Killed (signal 9)

T4 (14:30:46)
  Controller: 收到Worker 8的异常（或超时）
  wait_future_ordered(exit_on_exception=True)立即抛出异常
  日志："RuntimeError: rollout_distributed_batch failed"

T5 (14:30:46)
  Controller: 尝试进入train阶段（异常处理逻辑）
  但rollout数据不完整，DistributedBatchMemory.split()出错
  
T6 (14:30:47)
  Train Controller: train_distributed_batch调用失败
  日志："FrameworkError: DistributedBatchMemoryError: Batch size mismatch"
```

### 论证：故障耦合（挑战A）

**跨阶段传播路径**：

```
Rollout Worker 8 CPU争用/OOM（根因）
    ↓
Rollout阶段数据不完整（15/16 workers成功）
    ↓
DistributedBatchMemory数据切分异常
    ↓
Train阶段shape mismatch错误（表象）
```

**关键论证点**：

1. **错误日志出现在Train阶段**：`DistributedBatchMemoryError: Batch size mismatch`
2. **真实根因在Rollout阶段**：Worker 8的CPU争用导致OOM
3. **从Train日志无法追溯**：

   - Train workers只知道收到的batch size不对
   - 不知道是哪个rollout worker失败
   - 不知道是CPU、内存还是其他原因

4. **时间差**：Rollout失败（T3）→ Train报错（T6），相差1秒

**指标证据**：

- Rollout Worker 8的CPU利用率95%，GPU利用率仅40%（不匹配）
- Rollout Worker 8的SM占用率从80%暴跌至35%
- Train阶段收到的batch size 1500，而非预期1600

### 论证：RL特有（挑战C）

**数据流差异**：

| 维度 | 传统训练 | RL训练（本场景） |

|------|---------|----------------|

| 数据来源 | 静态Dataset（磁盘） | 动态生成（Rollout） |

| 数据shape | 固定（预处理） | 可变（生成长度不确定） |

| Worker失败影响 | 重新从Dataset读取 | 无法重新生成（在线） |

| 跨阶段依赖 | 无 | Rollout → Train强依赖 |

**RL架构依赖**：

- `DistributedBatchMemory`的split/merge机制要求所有worker数据对齐
- `wait_future_ordered(exit_on_exception=True)`一个worker失败导致整体失败
- 无法"跳过"失败的worker，因为RL需要连续的训练数据流

---

## 场景3：Train阶段GPU OOM被Cleanup错误掩盖（挑战B）

### 故障描述

对应FO.md场景6：RPC Server端异常导致Controller端误判

### RL独有性

**在传统训练中的表现差异？**

- 传统训练：GPU OOM立即报错并退出，日志清晰显示OOM
- RL训练：GPU OOM → HTTP 500 → Cleanup超时 → 最后显示的错误是"Failed to stop job"
- 由于RL的单控制器架构和RPC通信，cleanup的网络问题掩盖了原始的GPU故障

**架构依赖**：

- RPC通信：所有异常统一返回HTTP 500（`scheduler/rpc/rpc_server.py:77-84`）
- Cleanup机制：异常处理自动调用`scheduler.cleanup_jobs()`（`examples/grpo_trainer.py:523-528`）
- 单控制器：cleanup失败的日志时间戳比原始故障晚30秒

### 故障注入方案

**注入位置**：Train Worker 5 + Asystem Scheduler API Server

**注入故障类型**：

1. **进程内存注入到Train Worker 5**（注入能力2）：

   - 目标：Train Worker 5的megatron server进程
   - 内存占用：从75GB（正常）提升到79GB（接近GPU显存上限80GB）
   - 占用速率：1GB/5秒
   - 结果：Train阶段触发CUDA OOM

2. **网络延迟注入到Cleanup路径**（注入能力5）：

   - 源：Controller节点
   - 目标：Asystem Scheduler API Server
   - 基础延迟：15000ms（15秒）
   - 抖动：5000ms
   - 持续时间：60秒（覆盖cleanup阶段）
   - 指定端口：8081（Asystem API端口）
   - 触发时机：在Train阶段失败后（通过fotest框架的动态hook触发）

### 预期收集指标

#### 系统级指标

| 指标 | 收集位置 | 正常值 | 故障时预期值 |

|------|---------|--------|-------------|

| **train阶段耗时** | Controller | 180秒 | 2秒（快速失败） |

| **cleanup阶段耗时** | Controller | 5秒 | 30秒（网络超时） |

| **故障检测延迟** | 从GPU OOM到Controller感知 | N/A | 2秒（快速） |

| **日志时间差** | GPU OOM日志 vs Cleanup错误日志 | N/A | 30秒 |

#### Train Worker 5细粒度指标

| 指标 | 正常值 | OOM前值 | OOM时刻 | 收集方法 |

|------|--------|---------|---------|---------|

| GPU显存占用 | 75GB | 79GB | OOM异常 | `nvidia-smi dmon -s m` (1秒采样) |

| GPU显存分配速率 | 0 MB/s（稳定） | 200 MB/s | N/A | 计算显存占用差分 |

| CUDA分配失败次数 | 0 | 0 | 1次立即失败 | Megatron日志 |

| SM利用率 | 90% | 95% | 0%（crash） | `nvidia-smi dmon -s u` |

| Tensor Core利用率 | 85% | 90% | 0% | `nvidia-smi dmon -s u` |

#### Cleanup阶段网络指标

| 指标 | 正常值 | 故障时值 | 说明 |

|------|--------|---------|------|

| Controller → Asystem API延迟 | 5ms | 15000ms | 注入的网络延迟 |

| stop_job HTTP请求耗时 | 1秒 | 30秒超时 | `requests.post(..., timeout=30)` |

| 其他worker cleanup成功数 | 3个job（actor, rollout, ref） | 0个 | 全部超时 |

### 日志分析：错误掩盖现象

**日志时间线**：

```
# Train Worker 5日志
[14:30:15.123] INFO: Received /train_batch request
[14:30:15.234] INFO: Moving data to GPU...
[14:30:15.456] INFO: GPU memory allocated: 75.2 GB
[14:30:16.123] INFO: GPU memory allocated: 78.8 GB  ← 注入生效
[14:30:16.789] ERROR: CUDA out of memory
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 2.00 GiB (GPU 0; 79.35 GiB total capacity; 
78.89 GiB already allocated; 0.23 GiB free)
[14:30:16.891] ERROR: Train batch failed

# Controller日志
[14:30:15.100] INFO: start to train, step: 10, epoch: 0
[14:30:16.950] ERROR: Non-retryable HTTP error: 500
[14:30:16.951] ERROR: train_distributed_batch failed
[14:30:16.952] INFO: An error occurred during training, scheduler begin cleanup resource.

[14:30:16.953] INFO: Stopping job with UID: train_worker_job_abc123
[14:30:16.954] INFO: Sending POST to http://asystem-api:8081/.../cancel
# ... 30秒静默（网络延迟15秒 + 重试） ...
[14:30:46.955] ERROR: Error stopping job train_worker_job_abc123: Request timeout  ← 最后显示的错误！
requests.exceptions.Timeout: HTTPConnectionPool(host='asystem-api', port=8081): Read timed out.

[14:30:46.956] INFO: Stopping job with UID: rollout_worker_job_def456
[14:31:16.957] ERROR: Error stopping job rollout_worker_job_def456: Request timeout

[14:31:16.958] INFO: scheduler cleanup_jobs finished
```

### 论证：日志误导性（挑战B）

**错误掩盖机制**：

```
Train Worker 5 GPU OOM (14:30:16.789) ← 真实根因
    ↓
Controller收到HTTP 500 (14:30:16.950)
    ↓
开始cleanup_jobs (14:30:16.952)
    ↓
stop_job网络超时 (14:30:46.955) ← 最后显示的错误
```

**从下往上看日志的误导**：

1. **开发者首先看到**（最后一行）：`Error stopping job: Request timeout`
2. **可能误判为**：网络故障导致cleanup失败
3. **实际根因**：Train Worker 5的GPU OOM（需要往上翻30秒的日志）
4. **时间戳差异**：30秒（cleanup超时）掩盖了0.2秒（GPU OOM）的真实故障时间

**指标证据**：

- Train Worker 5的GPU显存占用曲线：75GB → 79GB → OOM（清晰的因果）
- Controller到Asystem API的延迟：5ms → 15000ms（异常跳变）
- 日志时间戳：GPU OOM时刻 vs Cleanup错误时刻相差30秒

**诊断困难量化**：

- 需要查看的日志文件数：1（Controller） + 32（Train Workers） = 33个
- 需要跨越的日志行数：约1000行（30秒 × 平均每秒10行日志）
- 需要对比的时间戳：2个关键时刻（OOM vs Cleanup）

---

## 场景4：共卡模式Event通知失败导致GPU死锁（挑战C）

### 故障描述

对应FO.md场景10：共卡模式下event通知失败导致死锁

### RL独有性

**为什么在传统训练中不存在？**

- 传统训练：只有train workers，GPU资源不需要在不同工作负载间切换
- 传统推理：推理系统独占GPU，不需要与训练系统协调GPU使用权
- RL训练特有：共卡模式（colocation）让train和rollout worker共享GPU，通过HTTP event通知切换，任何网络故障破坏GPU切换协议

**架构依赖**：

- Colocation模式：Train worker和Rollout worker运行在同一节点的同一GPU上
- Event通知机制：
  - `rollout.notify_event("rollout_start")` → Rollout worker占用GPU
  - `rollout.notify_event("rollout_end")` → Train worker重新占用GPU
- GPU锁机制：Train worker在收到"rollout_start"时应释放GPU锁，等待"rollout_end"

代码位置：

- `examples/grpo_trainer.py:268-286, 376-384`：Colocation初始化和event通知
- `extension/asystem/remote_hybrid_inference_worker.py:778-810`：Rollout worker处理event
- `extension/asystem/remote_hybrid_train_worker.py:396-431`：Train worker处理event

### 故障注入方案

**注入位置**：Colocation节点0（Train Worker 0 + Rollout Worker 0共享GPU 0）

**注入故障类型**：

1. **网络延迟注入到event通知路径**（注入能力5）：

   - 源：Controller节点
   - 目标：Colocation节点0的Rollout Worker
   - 基础延迟：300000ms（5分钟）
   - 持续时间：覆盖"rollout_start" event
   - 指定端口：Rollout Worker的HTTP端口（如8080）
   - 网卡设备：eth0

2. **可选叠加：DNS故障**（注入能力7）：

   - 使Controller无法解析Rollout Worker 0的hostname
   - 结果：event HTTP请求立即失败（Connection refused）

### 预期收集指标

#### 系统级指标

| 指标 | 收集位置 | 正常值 | 故障时预期值 |

|------|---------|--------|-------------|

| **rollout_start event耗时** | Controller | 0.1秒 | 600秒（超时，`timeout=600`） |

| **rollout阶段总耗时** | Controller | 120秒 | 7200秒（rollout执行但GPU被train占用） |

| **weights_update到rollout间隔** | Controller | 1秒 | 601秒（等待event超时） |

#### Colocation节点0 GPU细粒度指标

| 指标 | 正常值（rollout期间） | 故障时值 | 说明 |

|------|---------------------|---------|------|

| GPU利用率 | 85%（rollout推理） | 0%（死锁） | Train和Rollout都在等待 |

| GPU显存占用 | 40GB（rollout模型） | 75GB（train模型未释放） | Train未释放GPU |

| GPU进程数 | 1（rollout） | 2（train和rollout都在） | 资源冲突 |

| CUDA context数 | 1 | 2（但只有1个active） | 死锁 |

| SM占用率 | 80% | 0% | 无计算 |

#### Rollout Worker 0细粒度指标

| 指标 | 正常值 | 故障时值 | 收集方法 |

|------|--------|---------|---------|

| Event接收延迟 | 50ms | 300000ms（5分钟） | Rollout worker日志时间戳 |

| CUDA分配尝试次数 | 1次成功 | >100次失败 | `cudaMalloc` 重试日志 |

| CUDA OOM错误 | 否 | 是（GPU被train占用） | Megatron日志 |

| 网络接收延迟 | 50ms | 300000ms | `tcpdump` |

#### Train Worker 0细粒度指标

| 指标 | 正常值 | 故障时值 | 说明 |

|------|--------|---------|------|

| GPU锁状态 | 释放（收到rollout_start） | 持有（未收到event通知） | 代码内部状态 |

| 等待rollout_end时长 | 120秒 | 7200秒 | Train worker hang住 |

### 日志分析：死锁现象

**时间线**：

```
T0 (14:30:00)
  Controller: weights_update完成
  准备开始rollout阶段

T1 (14:30:01)
  Controller: 发送 rollout.notify_event("rollout_start", global_step=10)
  目标：Rollout Worker 0
  网络延迟注入生效：event HTTP请求卡住

T2 (14:30:01)
  Train Worker 0: 未收到任何通知，继续持有GPU锁
  Rollout Worker 0: 未收到"rollout_start" event，不敢占用GPU

T3 (14:40:01, 10分钟后)
  Controller: rollout_start event超时（timeout=600秒）
  日志：WARNING: notify_event timeout
  Controller继续执行，调用 rollout.rollout(batch_data)

T4 (14:40:02)
  Rollout Worker 0: 收到rollout请求，尝试分配GPU内存
  CUDA: cudaMalloc失败（GPU被Train Worker 0占用）
  日志：InferenceEngineError: RolloutError: CUDA out of memory

T5 (14:40:02)
  Train Worker 0: 仍在等待rollout_end event（永远不会到来）
  Rollout Worker 0: 持续重试CUDA分配（每次失败）
  结果：死锁
```

### 论证：RL特有（挑战C）

**GPU资源协调差异**：

| 维度 | 传统训练 | 传统推理 | RL训练（Colocation） |

|------|---------|---------|---------------------|

| GPU所有权 | Train workers独占 | Inference workers独占 | Train和Rollout共享，动态切换 |

| 协调机制 | 无需协调 | 无需协调 | 依赖HTTP event通知 |

| 网络故障影响 | 仅数据传输 | 仅数据传输 | GPU资源死锁 |

| 故障恢复 | 重启worker | 重启worker | 无法恢复（需重启整个训练） |

**Colocation架构依赖**：

- 为了节省GPU资源，将train和rollout调度到同一GPU（`scheduler/asystem/__init__.py:375-381`）
- Event通知作为协调协议，无ACK确认机制（`examples/grpo_trainer.py:376-384`）
- 网络故障破坏协议 → GPU资源死锁

**指标证据**：

- GPU利用率0%但显存占用75GB（资源被占用但无计算）
- Rollout Worker的CUDA分配尝试次数>100次（持续失败）
- Event接收延迟300000ms vs 正常50ms（600倍差异）

---

## 场景5：Checkpoint保存故障延迟暴露（挑战A）

### 故障描述

对应FO.md场景4：文件系统IO故障跨阶段传播

### RL独有性

**在传统训练中的表现差异？**

- 传统训练：Checkpoint保存失败立即报错，下次保存覆盖损坏的文件
- RL训练：
  - 当前step的checkpoint保存失败（latest_recover_save）
  - 下个step的weights_update从损坏的checkpoint读取权重
  - 错误延迟1个step才暴露（时间差可能数小时）

**架构依赖**：

- disk模式权重同步：train调用`upload_weights(type="disk")`从checkpoint目录读取权重
- Checkpoint保存和权重读取共享同一文件系统路径
- latest_checkpoint使用parity机制（`_odd`/`_even`），损坏可能影响2个step

代码位置：

- `examples/grpo_trainer.py:460-482`：latest_recover_save调用
- `recover/latest_checkpoint.py:70-174`：save写入文件系统
- `examples/grpo_trainer.py:335-352`：disk模式upload_weights

### 故障注入方案

**注入位置**：Train Worker 0（DP rank 0，负责保存checkpoint）

**注入故障类型**：

1. **磁盘故障**（注入能力3）：

   - Worker: Train Worker 0
   - 目标路径：NFS挂载点 `/storage/checkpoints/`
   - 填充百分比：从50%提升到99.9%（接近满）
   - 触发时机：latest_recover_save阶段（step 10）

2. **IO争用**（注入能力4）：

   - Worker: Train Worker 0
   - 增加IO负载：每秒写入1GB随机数据到NFS
   - 块大小：4MB
   - 持续时间：覆盖整个save阶段（约60秒）
   - 结果：fsync hang住，RPC超时

### 预期收集指标

#### 系统级指标

| 指标 | 收集位置 | 正常值 | 故障时预期值 |

|------|---------|--------|-------------|

| **latest_recover_save耗时** | Controller (step 10) | 30秒 | 3600秒（超时） |

| **weights_update耗时** | Controller (step 11) | 20秒 | 立即失败（2秒） |

| **NFS可用空间** | NFS server | 10TB | 10GB（0.1%） |

| **故障检测延迟** | 从save失败到weight读取失败 | N/A | 1个step（可能数小时） |

#### Train Worker 0细粒度指标（Checkpoint保存，step 10）

| 指标 | 正常值 | 故障时值 | 收集方法 |

|------|--------|---------|---------|

| 磁盘写入速率 | 500 MB/s | 5 MB/s（慢100倍） | `iostat -x 1` |

| NFS写入延迟 | 10ms | 5000ms | `nfsstat -m` |

| fsync耗时 | 100ms | 3600秒（hang住） | `strace -T -e fsync` |

| Checkpoint文件大小 | 14GB（完整） | 2GB（部分写入） | `du -h` |

| IO wait百分比 | 5% | 95% | `top`, `iowait` |

#### Train Worker 0细粒度指标（权重读取，step 11）

| 指标 | 正常值 | 故障时值 | 说明 |

|------|--------|---------|------|

| Checkpoint加载耗时 | 15秒 | 立即失败（文件损坏） | Worker日志 |

| 文件完整性校验 | 通过 | 失败（EOF或magic number错误） | PyTorch `torch.load()` |

| Disk read错误数 | 0 | >10（读取损坏文件） | `dmesg | grep ext4` |

### 日志分析：延迟暴露现象

**时间线**：

```
# Step 10 (14:30:00)
T0: Train阶段完成
T1: Controller调用 latest_recover.save(actor, ...)
T2: Train Worker 0开始保存checkpoint到NFS

# Train Worker 0日志（step 10, 14:30:10）
[14:30:10.123] INFO: Saving checkpoint to /storage/checkpoints/expr/trial/step_10_even
[14:30:10.234] INFO: Saving HF model...
[14:30:20.345] INFO: HF model saved (14.2 GB)
[14:30:20.456] INFO: Saving mcore checkpoint...
[14:30:25.567] INFO: Checkpoint file written, calling fsync...
# ... 3600秒hang住（NFS IO争用） ...
[15:30:25.678] ERROR: requests.exceptions.Timeout: Write timeout
[15:30:25.789] ERROR: Checkpoint save failed, but some files already written

# Controller日志（step 10, 15:30:25）
[15:30:25.890] WARNING: latest_recover_save timeout, continuing to next step
[15:30:25.891] INFO: Step 10 completed

# Step 11 (15:30:30)
T10: 下一个step开始
T11: weights_update阶段，使用disk模式

# Train Worker 0日志（step 11, 15:30:35）
[15:30:35.123] INFO: upload_weights begin, type=disk
[15:30:35.234] INFO: Reading checkpoint from /storage/checkpoints/expr/trial/latest
[15:30:35.345] INFO: Loading model state_dict...
[15:30:35.456] ERROR: Checkpoint corrupted: EOFError at position 2147483648
[15:30:35.567] ERROR: TrainEngineError: UploadWeightsError: Invalid checkpoint format

# Controller日志（step 11, 15:30:35）
[15:30:35.678] ERROR: Weight update failed
[15:30:35.789] ERROR: RuntimeError: upload_weights failed
```

### 论证：故障耦合（挑战A）

**跨阶段+跨step传播路径**：

```
Step 10: latest_recover_save阶段NFS故障（根因）
    ↓
Checkpoint文件部分写入（损坏状态）
    ↓
Controller继续执行（未检测到损坏）
    ↓
Step 11: weights_update阶段读取checkpoint（表象）
    ↓
Train端报错"Invalid checkpoint format"
```

**关键论证点**：

1. **错误日志出现在step 11的weights_update阶段**
2. **真实根因在step 10的latest_recover_save阶段**（1小时前）
3. **时间差导致诊断困难**：

   - Step 10 save失败的日志可能被忽略（仅WARNING）
   - Step 11 load失败时，无法直接关联到step 10
   - 需要回溯1小时的日志才能找到根因

4. **文件系统不可见性**：

   - Controller无法检测checkpoint文件是否损坏
   - 只有在实际读取时才发现（延迟检测）

**指标证据**：

- Step 10的fsync耗时3600秒（异常）
- Step 10的Checkpoint文件大小2GB vs 预期14GB（明显不完整）
- Step 11的Checkpoint加载立即失败（EOF错误）
- NFS可用空间从10TB降至10GB（根本原因）

---

## 通用监控指标收集方案

### 硬件级指标（所有场景）

**GPU指标**（每秒采样）：

```bash
# GPU利用率、SM占用率、显存占用
nvidia-smi dmon -s pucvmet -c 0 -d 1 > gpu_metrics.csv
```

**网络指标**（每秒采样）：

```bash
# 网络吞吐量
iftop -t -s 1 -L 1000 > network_throughput.txt

# 网络延迟（ping）
ping -i 0.1 <target_host> > network_latency.txt

# TCP连接状态
watch -n 1 "netstat -an | grep ESTABLISHED" > tcp_connections.txt
```

**系统资源指标**（每秒采样）：

```bash
# CPU、内存
top -b -d 1 > system_metrics.txt

# 磁盘IO
iostat -x 1 > disk_io.txt

# NFS统计
nfsstat -m > nfs_stats.txt
```

### 应用级指标（通过stats_tracker）

**RL训练特有指标**：

- Reward: `stats_tracker.scalar(**{"rewards": reward})`
- KL散度: 计算ref_logprobs和actor_logprobs的KL
- 吞吐量: `samples_per_second = batch_size / stage_duration`
- 阶段耗时: `stats_tracker.record_timing("rollout_step")`

**数据流指标**：

- Rollout生成长度: `stats_tracker.scalar(**{"seqlen": len(seq)})`
- Batch size: `stats_tracker.scalar(**{"batch_size": len(batch)})`
- 数据staleness: `stats_tracker.scalar(**{"stale_version_num": stale_count})`

### 日志收集和分析

**时间戳同步**：

所有节点使用NTP同步时间，确保日志时间戳对齐（误差<10ms）

**日志聚合**：

使用ELK stack或Loki聚合所有worker日志，支持：

- 按request_id跨worker追踪
- 按时间窗口聚合多个worker的日志
- 关键词搜索（OOM, timeout, error）

**故障标记**：

在注入故障时，同时写入标记文件：

```json
{
  "fault_injection_start": "2024-11-24T14:30:00Z",
  "fault_type": "cpu_contention",
  "target_worker": "rollout_worker_8",
  "expected_failure_stage": "train"
}
```

---

## 论文实验设计建议

### 实验流程

1. **Baseline运行**（无故障）：

   - 运行10个steps，收集所有指标
   - 建立正常值范围（均值±2σ）

2. **故障注入运行**（每个场景）：

   - 在特定阶段注入故障
   - 收集相同指标
   - 记录故障检测时间、定位时间

3. **盲测诊断**（验证诊断难度）：

   - 只给其他研究人员看Controller日志
   - 计时他们定位到真实根因的时间
   - 统计误诊率（认为是另一个阶段的问题）

### 评估指标

**挑战A（故障耦合）**：

- 故障发生阶段 vs 错误日志出现阶段的差异
- 需要查看的日志文件数量
- 跨阶段传播的hop数

**挑战B（诊断困难）**：

- 平均定位时间（MTTD: Mean Time To Diagnose）
- 误诊率（错误归因到其他阶段）
- 需要关联的日志行数

**挑战C（RL特有）**：

- 对比传统SFT训练的故障表现差异
- 列出RL架构特有的耦合点（如Colocation, MetaServer）
- 量化RL多阶段对故障传播的放大效应

---

## 实验预期结果

### 量化指标（预期值）

| 挑战 | 场景 | 关键指标 | 预期结果 |

|------|------|---------|---------|

| **A** | 场景1 | 故障传播hop数 | 2 hops（Train → MetaServer → Rollout） |

| **A** | 场景2 | 错误阶段差异 | Rollout故障 → Train报错（跨1个阶段） |

| **A** | 场景5 | 故障检测延迟 | 1个step（可能数小时） |

| **B** | 场景3 | 日志时间差 | 30秒（Cleanup掩盖GPU OOM） |

| **B** | 场景3 | MTTD | >10分钟（需要跨33个日志文件） |

| **C** | 场景4 | Colocation独有 | 传统训练无此问题 |

| **C** | 场景1 | MetaServer单点 | 传统训练无跨系统权重同步 |

### 论文中的可视化

**图1：故障传播路径图**

- X轴：时间
- Y轴：系统组件（Controller, Train, Rollout, MetaServer）
- 显示故障如何从一个组件传播到另一个组件

**图2：指标异常检测**

- X轴：时间
- Y轴：指标值（GPU利用率、网络延迟等）
- 标注故障注入时刻和错误日志出现时刻的时间差

**图3：诊断难度对比**

- X轴：场景
- Y轴：MTTD（平均定位时间）
- 对比有/无细粒度监控的诊断时间

### 论文论证结构

**Section 3: Motivation - Three Challenges**

- 3.1 Challenge A: Fault Coupling across Stages
  - 用场景1、2、5证明
  - 强调单控制器、SPMD架构的必然性
- 3.2 Challenge B: Misleading Error Logs
  - 用场景3证明
  - 强调超长超时、Cleanup掩盖的普遍性
- 3.3 Challenge C: RL-Specific Fault Patterns
  - 用场景4证明
  - 对比传统训练，强调RL多阶段协作的独特性

**Section 4: Fault Injection Experiments**

- 4.1 Experimental Setup
- 4.2 Scenario 1-5 Results
- 4.3 Quantitative Analysis（MTTD, 误诊率等）

**Section 5: Discussion**

- 为什么这些故障难以通过简单日志排查？
- 为什么细粒度监控是必需的？