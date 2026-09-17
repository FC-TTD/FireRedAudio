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
不会重新暴露独立GPU后端。旧端口已切换并通过真实业务验收，旧GPU preview已停止。

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

消费者冷态兼容及超时补丁
[ttd-dme-index PR #2](https://github.com/FC-TTD/ttd-dme-index/pull/2)已合并为`71ce481`。
历史Mac部署记录不能证明当前消费者位置、版本或使用情况；个人runtime更新不是provider
迁移的前置条件。未升级的旧消费者仍可能拒绝冷态，CPU桥只保留原地址和HTTP合同。

旧入口切换使用`legacy-bridge.yml`，退役声明使用`legacy-retired.yml`。先保存旧Compose、
镜像及进程身份，关闭新连接并确认连接/队列排空。**停止容器会丢失tmpfs**，必须在排空后、
停止前归档`/tmp/gradio`和`/outputs`，校验共享存储上的文件；正常停止后再核对持久输出。
将退役声明合入原部署文件，并设旧容器restart=no，保留停止的容器、原数据和回滚资产。
启用CPU桥后，验证两旧端口的实际推理、UI上传和旧音频下载，再记录交接完成。
回滚先停止CPU桥，恢复归档Compose配置及原重启策略，再启动保留的原容器；将归档的
`gradio-cache.tar`还原至新tmpfs `/tmp/gradio`并校验后再开放入口。原GPU必须有足够容量。
回滚后新增文件留在共享存储，不删除或回滚用户数据。

本轮交接已完成：旧实例exit0、释放20934MiB，原容器restart=no、部署声明replicas=0。
两旧端口8次API回归、旧17880浏览器识别/克隆/4秒WAV下载均通过；迁移前4个文件在三个
入口共12次下载SHA一致。最终30个Runtime/Hub活动成功、pending0，10个受管服务开放。
门户链接更新为友好域名，第二GPU调度节点仍未启用。
