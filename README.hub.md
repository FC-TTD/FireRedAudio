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

当前CPU合同测试11项通过，含真实Gradio队列与SDK进程往返；真实受管GPU上线结论随后
记入Hub `docs/proposals/model-compute-pool/`，不以本地fixture宣称部署完成。
