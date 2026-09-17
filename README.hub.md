# FireRedAudio 受管接入

正式来源为 `FC-TTD/FireRedAudio`（上游 `FireRedTeam/FireRedAudio`）的 `hub` 分支。
先在上游 `88b826378023eb9a49b297214568398a300e5c32` 上保存原 edge preview 的42个运行源码文件，
再新增 `hub_runtime/` 与 `deploy/hub/`。逐文件来源见 [provenance](docs/hub-takeover/provenance.json)。
基础UI与CPU decoder改造来自原手工preview；embedding/multi/single理解扩展已在
`FC-TTD/ttd-dme-index`的`deploy/firered-embedding/`入库，来源没有丢弃。

原7类UI能力、8个Gradio named endpoints及REST参数保留，UI采用原副本最小修改。
PyTorch/Transformers BF16主模型按租约放到单卡，RedAE decoder保持原CPU；没有新增量化、
权重offload或更换推理框架。约106亿参数的BF16权重占19.77GiB，显存主要是真实权重。

CPU HTTP宿主与独立GPU进程分离，真实callback/RPC持有Runtime activity；取消异步等待
会settle实际工作后清临时音频。保留原生共享推理锁与Gradio队列，不自行扩大并发。
`/health.model_loaded`是实际权重状态，`runtime_ready`表示本地Runtime可受理，允许冷态
提交；健康查询本身不加载GPU。使用旧`model_loaded`硬检查的消费者需要升级。

`POST /embedding/unload`保留“卸载权重”语义，现与推理共用原生锁，避免清引用时仍有
推理引用、下次误加载第二份模型。它不宣称已释放Hub租约；完整GPU进程回收和租约释放
由Hub drain/资源竞争完成。压力卸载后，元数据按engine代次失效，不假报已加载。

入口目标为 `http://firered-audio`；新服务仅worker调度，三张3090可按UUID/序号绑定。
原生短样本全能力11次调用峰值21194MiB；28.76秒单/双音频理解等4次调用峰值21354MiB。
初始预算21.5GiB，同时保留全池2GiB安全余量；尚不能把这些结果等同于所有长度/组合的峰值。

数据使用 `/TTD-Data/firered/{pretrained_models,hf-cache,outputs,gradio-cache}`，模型只读。
迁移先校验文件哈希，原NVMe目录不删除；旧输出和Gradio缓存须保留原路径及文件链接。
`legacy-bridge.yml`为旧edge17880/17881准备纯CPU兼容转发，目标是受管CPU entry，
不会重新暴露独立GPU后端。实际切换仍需完成新服务与消费者验收，旧preview此前保持。

旧 `Dockerfile.preview`、`compose.preview.yaml` 与 `README.preview.md` 仅保留来源。
正式部署从固定源码快照和镜像digest构建，以 `deploy/hub/compose.yml`、`business.yml`
及Hub仓库的 `scripts/firered_release.py` 完成维护屏障、actor登记与真实验收。

当前CPU合同测试12项通过，含真实SDK异步归属检查、Gradio队列和进程往返。

## 2026-09-17受管验收与过渡

worker actor `firered-adopt-e8893a4bf5e1`已部署，`http://firered-audio`真实API/UI可用。
11条完整能力调用、4条较长输入、浏览器识别/克隆/下载及冷态DME客户端均通过；
受管峰值21334MiB，20个Runtime/Hub活动终态一致。embedding与原preview逐值相同。
并发提交的权重unload会等待推理；Hub drain确认空进程组和租约释放，旧音频仍可下载。
首次共享存储加载190.5秒、实际消费者冷重载约78秒，请求后保持热驻留。

完整结果见[Hub接管报告](https://github.com/FC-TTD/ttd-hub/blob/main/docs/proposals/model-compute-pool/firered-online-2026-09-17.md)。
部署镜像源commit为`e8893a4bf5e1cc723ca6099d330c2ee34fb23ff7`，后续文档不改变制品身份。

**旧入口交接尚未结束。** 现有Mac消费者的冷态兼容补丁在
[ttd-dme-index PR #2](https://github.com/FC-TTD/ttd-dme-index/pull/2)，需其维护者更新并重启。
目前edge旧17880/17881和preview GPU仍保留，CPU桥资产已验证但未切换；客户端就绪后
再关闭旧准入、正常退出旧实例、增量同步文件并切换桥接，不操作个人数据库或进程。
