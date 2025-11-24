# 分布式RLHF训练系统故障耦合分析报告

## 系统架构概述

该系统采用**单控制器架构**，主要包含以下阶段：

1. **weights_update**：权重同步（train → rollout）
2. **rollout**：推理生成样本
3. **reference**：计算参考模型logprobs（可选）
4. **train**：模型训练
5. **latest_recover_save**：保存最新检查点
6. **periodic_recover_save**：定期保存检查点

关键架构特征：

- **单控制器（grpo_trainer.py）**：串行协调所有阶段，任何阶段失败导致整体失败
- **RPC同步阻塞调用**：通过`_rpc_call`并行调用多个worker，但`wait_future_ordered`等待所有worker完成
- **SPMD并行模式**：train/reference worker采用数据并行+模型并行
- **共卡模式（Colocate）**：train和rollout共享GPU，通过HTTP event通知切换
- **HTTP通信**：RPC超时7200秒，无stream，依赖TCP连接稳定性

---

## 故障耦合场景分析

### 场景1：共卡模式下GPU资源耦合

**阶段**：rollout阶段
**硬件故障**：GPU硬件故障（ECC错误、CUDA卡死、显存污染）
**耦合机制**：共卡模式下train和rollout共享GPU内存空间
**故障传播**：

1. rollout阶段GPU出现硬件故障（如ECC错误）导致推理卡死
2. 由于共卡模式，train worker的GPU context与rollout worker共享
3. 当weights_update阶段train调用`upload_weights`时，通过HTTP请求train worker的`/update_weights`端点
4. train worker尝试访问被污染的GPU内存或CUDA context，产生CUDA错误
5. 错误日志显示：`TrainEngineError: UploadWeightsError: Status code: 500, CUDA error: device-side assert triggered`

**诊断困难**：

- 错误日志出现在weights_update阶段的train worker上传权重时
- 实际根因是rollout阶段的GPU硬件故障
- GPU资源在OS层面是共享的，无法从代码层面隔离验证
- 共卡模式的架构设计导致GPU故障必然跨阶段传播

**代码证据**：

- `examples/grpo_trainer.py:268-286`：共卡模式初始化，`colocate_with=actor`
- `extension/asystem/remote_hybrid_train_worker.py:166-196`：upload_weights HTTP调用
- `scheduler/asystem/__init__.py:375-381`：ScheduleStrategy colocation调度策略

---

### 场景2：NCCL/RDMA权重同步网络故障耦合

**阶段**：weights_update阶段
**硬件故障**：RDMA网卡故障、NCCL IB超时、网络分区
**耦合机制**：train.upload_weights和rollout.update_weights并行执行，通过元数据服务器（MetaServer）协调NCCL通信组
**故障传播**：

1. weights_update阶段，train worker调用`upload_weights(type="nccl")`上传权重到元数据服务器
2. 并行地，rollout worker调用`update_weights(type="nccl")`从元数据服务器拉取权重
3. train worker端RDMA网卡故障，导致NCCL集合通信（AllGather/Broadcast）卡死
4. rollout worker的NCCL通信操作因等待train worker而超时（3600秒）
5. rollout阶段产生错误日志：`InferenceEngineError: UpdateWeightError: Failed to update weights: NCCL timeout`
6. 但由于单控制器的`wait_future_ordered`机制，trainer主循环会同时抛出两个异常

**诊断困难**：

- rollout worker错误日志显示"update weights timeout"
- 实际根因是train worker端的RDMA网卡故障
- NCCL集合通信的SPMD架构要求所有rank必须同时参与，一个rank故障会hang住所有rank
- 从错误日志无法判断是train还是rollout的网络故障
- MetaServer作为协调者无法区分哪一端首先故障

**代码证据**：

- `examples/grpo_trainer.py:435-447`：并行执行upload和update
- `extension/asystem/remote_hybrid_train_worker.py:166-196`：train端NCCL upload
- `extension/asystem/remote_hybrid_inference_worker.py:545-582`：rollout端NCCL update
- `extension/asystem/meta_server.py:62-207`：MetaServer协调NCCL通信
- `scheduler/asystem/__init__.py:38-57`：NCCL相关环境变量配置

---

### 场景3：RPC调用网络延迟导致多阶段超时级联

**阶段**：rollout阶段
**硬件故障**：网络延迟突增、交换机拥塞、网卡丢包
**耦合机制**：单控制器架构下，各阶段通过RPC串行调用，任何阶段超时会阻塞后续阶段
**故障传播**：

1. rollout阶段，controller通过`_rpc_call`向所有rollout worker并行发送RPC请求
2. 其中一个worker所在节点网络延迟突增（如交换机拥塞），导致HTTP请求超时（7200秒）
3. 由于`wait_future_ordered`等待所有worker完成，整个rollout阶段卡住
4. 7200秒后，rollout超时，trainer主循环继续执行
5. 进入reference阶段，controller调用`ref.compute_logprobs_with_distributed`
6. reference worker尝试通过RPC从rollout结果读取数据，但由于rollout超时，数据不完整或损坏
7. reference阶段产生错误日志：`RefEngineError: ComputeLogpError: Invalid tensor shape`

**诊断困难**：

- 错误日志显示在reference阶段的数据处理错误
- 实际根因是rollout阶段的网络延迟导致数据不完整
- RPC调用的同步阻塞特性导致故障传播到下一阶段才暴露
- 从reference的错误日志无法追溯到rollout的网络问题
- 超时7200秒过长，掩盖了实际网络故障的发生时间

**代码证据**：

- `controller/rollout_controller.py:59-114`：`_rpc_call`并行调用并wait_future_ordered
- `scheduler/rpc/rpc_client.py:42-97`：RPC超时7200秒，最多重试3次
- `controller/train_controller.py:205-257`：reference阶段compute_logprobs调用
- `extension/asystem/remote_hybrid_train_worker.py:905-1036`：compute_logprobs数据处理

---

### 场景4：文件系统IO故障跨阶段传播

**阶段**：train阶段
**硬件故障**：存储节点磁盘故障、NFS挂载点不可用、文件系统满
**耦合机制**：disk模式权重同步和checkpoint保存共享同一文件系统路径
**故障传播**：

1. train阶段完成后，进入latest_recover_save阶段
2. controller调用`latest_recover.save(actor, ...)`保存checkpoint
3. actor调用`save(meta)`将模型写入NFS文件系统
4. NFS挂载点由于存储节点故障变为只读或不可用
5. 文件系统写入失败，产生OSError，但部分文件已写入（损坏状态）
6. 下一个step的weights_update阶段，train调用`upload_weights(type="disk")`
7. train尝试从损坏的checkpoint目录读取权重文件进行上传
8. weights_update阶段产生错误日志：`TrainEngineError: UploadWeightsError: Invalid checkpoint format`

**诊断困难**：

- 错误日志显示在weights_update阶段的train端权重加载错误
- 实际根因是latest_recover_save阶段的文件系统故障导致checkpoint损坏
- 文件系统故障发生在前一个step，但错误在下一个step暴露
- 从weights_update的日志无法看出是checkpoint保存问题
- 文件系统IO操作是不可避免的关键操作，无法从代码层面避免此类耦合

**代码证据**：

- `examples/grpo_trainer.py:460-482`：latest_recover_save调用
- `recover/latest_checkpoint.py:70-174`：save调用actor.save(meta)
- `extension/asystem/remote_hybrid_train_worker.py:218-272`：save写入文件系统
- `examples/grpo_trainer.py:428-434`：disk模式upload_weights从文件系统读取
- `extension/asystem/remote_hybrid_train_worker.py:197-216`：disk模式save调用

---

### 场景5：分布式数据切分故障传播

**阶段**：rollout阶段
**硬件故障**：rollout worker节点CPU资源争用、内存溢出
**耦合机制**：DistributedBatchMemory通过split切分数据到多个worker，单个worker故障导致整体数据不完整
**故障传播**：

1. rollout阶段，controller调用`rollout.rollout(batch_data, workflow)`
2. rollout内部调用`_rpc_call("rollout", batches, workflow)`并行向所有worker发送数据
3. 其中一个worker节点因CPU资源争用（如其他任务抢占）导致处理缓慢
4. 该worker的rollout任务超时或内存溢出（OOM）而失败
5. 由于`wait_future_ordered(exit_on_exception=True)`，rollout阶段立即失败
6. controller进入异常处理，尝试进入train阶段
7. train阶段调用`train_distributed_batch`时，DistributedBatchMemory因数据不完整而产生shape mismatch错误
8. train阶段产生错误日志：`FrameworkError: DistributedBatchMemoryError: Batch size mismatch`

**诊断困难**：

- 错误日志显示在train阶段的数据维度错误
- 实际根因是rollout阶段某个worker的CPU资源争用或OOM
- rollout的部分worker成功返回数据，部分失败，导致数据不完整
- DistributedBatchMemory的split/merge机制要求所有worker数据对齐，单个worker故障会导致后续阶段数据错误
- 从train的日志无法追溯到rollout的资源问题
- CPU争用和OOM属于OS层面资源竞争，代码无法预测和避免

**代码证据**：

- `controller/rollout_controller.py:341-354`：rollout调用_rpc_call和数据concat
- `dataset/distributed_batch_memory.py:41-58`：split数据切分逻辑
- `controller/train_controller.py:155-203`：train_distributed_batch调用
- `dataset/distributed_batch_memory.py:17-28`：_validate_dict_dataset验证batch size
- `utils/util.py`：wait_future_ordered的exit_on_exception行为

---

### 场景6：RPC Server端异常导致Controller端误判

**阶段**：train阶段
**硬件故障**：train worker节点GPU故障、CUDA OOM
**耦合机制**：RPC Server通过HTTP返回异常信息，Controller无法区分网络故障和worker内部故障
**故障传播**：

1. train阶段，controller通过RPC调用所有train worker执行`train_distributed_batch`
2. 其中一个worker节点GPU发生OOM（如KV cache耗尽）
3. worker的megatron server捕获CUDA OOM异常，返回HTTP 500错误
4. RPC Client收到500错误，判断为non-retryable error，立即抛出异常
5. 由于RPC超时设置为7200秒，controller无法区分是网络超时还是worker内部故障
6. Controller捕获异常后，进入`scheduler.cleanup_jobs()`清理所有worker
7. cleanup过程中，尝试stop其他正常运行的worker，但网络延迟导致stop请求超时
8. cleanup阶段产生错误日志：`SchedulerError: Failed to stop job: Request timeout`

**诊断困难**：

- 错误日志显示在cleanup阶段的调度器stop失败
- 实际根因是train阶段某个worker的GPU OOM
- RPC协议无法区分网络故障和worker内部故障，统一返回HTTP错误码
- Controller的异常处理会触发cleanup，cleanup的网络问题掩盖了原始的GPU故障
- 从cleanup的日志无法追溯到train worker的GPU OOM根因
- HTTP协议的无状态特性和超时机制使得故障诊断依赖日志时间戳而非调用栈

**代码证据**：

- `controller/train_controller.py:105-123`：_rpc_call捕获异常并raise
- `scheduler/rpc/rpc_server.py:34-84`：RPC Server返回HTTP 500
- `scheduler/rpc/rpc_client.py:63-97`：RPC Client重试逻辑和超时处理
- `examples/grpo_trainer.py:523-528`：异常处理中调用scheduler.cleanup_jobs()
- `scheduler/asystem/__init__.py:139-146`：cleanup_jobs调用stop_job

---

### 场景7：reference阶段网络故障导致train阶段数据异常

**阶段**：reference阶段
**硬件故障**：reference worker网卡故障、网络丢包
**耦合机制**：reference计算的logprobs通过内存传递给train阶段，网络故障导致数据传输不完整
**故障传播**：

1. reference阶段，controller调用`ref.compute_logprobs_with_distributed(dis_batch)`
2. reference worker通过HTTP返回logprobs tensor，但由于网卡丢包，部分数据未传输完整
3. Controller端的TCP连接未检测到错误（TCP可能重传成功但数据部分损坏）
4. Controller将不完整的logprobs添加到`rollout_res_dict["ref_logprobs"]`
5. 进入train阶段，`actor.train_distributed_batch(dis_batch)`处理数据
6. train worker在计算KL散度时，发现ref_logprobs的shape与rollout_logprobs不匹配
7. train阶段产生错误日志：`TrainEngineError: TrainBatchError: Tensor shape mismatch for KL computation`

**诊断困难**：

- 错误日志显示在train阶段的KL计算tensor shape错误
- 实际根因是reference阶段的网络丢包导致logprobs数据不完整
- TCP的重传机制可能掩盖了网络丢包，使得Controller未检测到传输错误
- reference和train之间通过内存传递数据，无checksum验证
- 从train的日志无法判断ref_logprobs是在reference阶段还是传输过程中损坏
- 网络丢包属于底层网络硬件故障，代码层面无法避免，只能依赖TCP协议

**代码证据**：

- `examples/grpo_trainer.py:423-438`：reference阶段计算logprobs并添加到dict
- `controller/train_controller.py:205-257`：compute_logprobs_with_distributed并行调用和concat
- `extension/asystem/remote_hybrid_train_worker.py:905-1036`：compute_logprobs HTTP返回
- `extension/asystem/remote_hybrid_train_worker.py:560-867`：train中使用ref_logprobs计算KL
- `dataset/distributed_batch_memory.py:233-241`：merge操作无数据完整性校验

---

### 场景8：periodic_recover_save文件系统故障导致下次训练失败

**阶段**：periodic_recover_save阶段
**硬件故障**：NFS存储节点故障、磁盘IO hang、文件系统元数据损坏
**耦合机制**：checkpoint保存与恢复共享文件系统，当前step的保存故障会影响未来step的恢复
**故障传播**：

1. periodic_recover_save阶段，controller调用`periodic_recover.save(actor, ...)`
2. actor通过RPC调用所有train worker保存模型和优化器状态
3. NFS存储节点发生故障，导致文件系统IO hang（如卡在fsync）
4. RPC请求超时（3600秒），periodic_recover_save失败
5. trainer主循环继续执行，进入下一个step
6. 若干step后，训练进程意外crash（如OOM）
7. 重启训练时，尝试从latest或periodic checkpoint恢复
8. 恢复阶段发现checkpoint文件不完整或元数据损坏
9. 恢复阶段产生错误日志：`RecoverError: Failed to load checkpoint: Corrupted file`

**诊断困难**：

- 错误日志显示在恢复阶段的checkpoint加载失败
- 实际根因是若干step前periodic_recover_save阶段的NFS故障
- checkpoint保存失败时可能未产生明显错误日志（仅超时警告）
- 恢复时无法判断checkpoint是何时、为何损坏
- 文件系统故障可能间歇性发生，导致部分checkpoint正常、部分损坏
- NFS的IO hang属于分布式文件系统的固有问题，代码无法避免

**代码证据**：

- `examples/grpo_trainer.py:484-507`：periodic_recover_save调用
- `recover/periodic_checkpoint.py:99-167`：save调用actor.save并处理异常
- `extension/asystem/remote_hybrid_train_worker.py:218-272`：save写入文件系统，timeout=3600
- `examples/grpo_trainer.py:145-172`：恢复时load checkpoint
- `recover/latest_checkpoint.py:235-246`：Recover.load异常处理

---

### 场景9：MetaServer故障导致权重同步卡死

**阶段**：weights_update阶段
**硬件故障**：MetaServer所在节点内存溢出、CPU hang、网络分区
**耦合机制**：NCCL/astate模式下，MetaServer作为协调者，其故障会hang住所有train和rollout worker
**故障传播**：

1. weights_update阶段，train worker调用`upload_weights(type="astate")`
2. train worker向MetaServer的`/v1/put_binary`端点PUT权重元数据
3. MetaServer因内存溢出（存储了大量权重元数据）而进程crash
4. train worker的HTTP请求超时（3600秒）
5. 同时，rollout worker调用`update_weights(type="astate")`
6. rollout worker尝试从MetaServer GET权重元数据，因MetaServer crash而hang住
7. 由于单控制器的并行执行，两个HTTP请求都超时
8. 错误日志同时出现在train和rollout：

- Train: `TrainEngineError: UploadWeightsError: Upload weights request timeout`
- Rollout: `InferenceEngineError: UpdateWeightError: Failed to update weights: Connection refused`

**诊断困难**：

- 错误日志在train和rollout两个阶段同时出现
- 实际根因是MetaServer的内存溢出
- 从worker的日志无法判断是MetaServer故障还是网络故障
- MetaServer作为单点故障，其crash会hang住整个权重同步流程
- MetaServer在主控制进程中以daemon线程运行，crash时可能无明显日志
- 内存溢出可能由前序step的权重数据累积导致，故障时间点难以确定

**代码证据**：

- `examples/grpo_trainer.py:97-114`：start_meta_server启动MetaServer
- `extension/asystem/meta_server.py:62-206`：MetaServer作为HTTP服务器，存储在内存
- `extension/asystem/meta_server.py:94-108`：put_binary/get_binary存储权重
- `extension/asystem/remote_hybrid_train_worker.py:166-196`：train端upload调用MetaServer
- `extension/asystem/remote_hybrid_inference_worker.py:545-582`：rollout端update调用MetaServer

---

### 场景10：共卡模式下event通知失败导致死锁

**阶段**：rollout阶段
**硬件故障**：网络延迟、HTTP连接中断、worker进程hang
**耦合机制**：共卡模式下，train和rollout通过HTTP event通知切换GPU使用权，event失败导致GPU资源死锁
**故障传播**：

1. weights_update完成后，controller调用`rollout.notify_event("rollout_start", global_step)`
2. rollout worker收到event，开始准备使用GPU进行推理
3. 同时，train worker应该释放GPU，等待rollout完成
4. 但由于网络延迟，rollout_start event的HTTP请求超时（600秒）
5. rollout worker未收到start event，GPU仍被train worker占用
6. controller继续执行，调用`rollout.rollout(batch_data, workflow)`
7. rollout worker尝试分配GPU内存进行推理，但GPU被train占用，产生CUDA OOM
8. rollout阶段产生错误日志：`InferenceEngineError: RolloutError: CUDA out of memory`
9. 同时，train worker因未收到rollout_end event，一直持有GPU锁，hang住

**诊断困难**：

- 错误日志显示rollout阶段的CUDA OOM
- 实际根因是rollout_start event通知失败，导致GPU资源未正确切换
- 共卡模式依赖event的HTTP通知，网络故障会破坏GPU切换协议
- CUDA OOM可能是GPU内存不足，也可能是GPU资源死锁
- 从rollout的日志无法判断event是否成功送达
- event通知机制无状态验证，无法确认GPU切换是否完成
- 共卡架构的设计使得GPU资源竞争无法从代码层面避免

**代码证据**：

- `examples/grpo_trainer.py:376-384`：rollout_start event通知，timeout=600
- `extension/asystem/remote_hybrid_inference_worker.py:778-810`：rollout worker处理event
- `extension/asystem/remote_hybrid_train_worker.py:396-431`：train worker处理event
- `examples/grpo_trainer.py:268-286`：共卡模式初始化，colocate_with=actor
- `controller/rollout_controller.py:116-199`：rollout初始化时传递colocate_with

---

## 总结

以上10个场景涵盖了系统中由于**架构本身设计**（单控制器、SPMD、共卡模式）和**不可避免的关键操作**（RPC调用、集合通信、文件系统IO、网络传输）导致的故障耦合。这些故障的共同特点是：

1. **跨阶段传播**：一个阶段的硬件故障在另一个阶段产生错误日志
2. **难以从代码验证避免**：故障源自OS/网络/硬件层面，代码无法预测
3. **架构固有耦合**：单控制器、共卡、SPMD等架构设计必然导致资源共享和依赖
4. **日志误导性强**：错误日志的阶段与根因阶段不一致，给诊断带来困难

**建议的诊断策略**：

- 启用全链路tracing（如OpenTelemetry），记录跨阶段的请求ID
- 增加硬件层面的监控（GPU ECC、RDMA计数器、NFS IO延迟）
- 在关键数据传输点添加checksum验证
- 为event通知机制添加ack确认和重试
- 定期检查checkpoint完整性，避免恢复时才发现损坏