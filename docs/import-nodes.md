# 文档导入实现与学习说明

节点名称和图连线保持不变。7 个节点均有业务代码；代码实现不等于当前机器已完成模型、服务的真实部署验收。

## 数据契约

- `chunks`: `content` 加 `metadata`。元数据包含来源路径、文档版本 ID、文件标题、顺序 part（从 0 开始）、字符 start/end（左闭右开）、标题、父标题、原 PDF 页码。原文缺少页码则为 null。
- `embeddings_content`: 每个 chunk 加 `dense`（1024 个 Python float）与 `sparse`（非负 int token ID 到有限非负 float 的字典）。
- `item_embedding`: 品名本身的向量，不使用某个切片向量冒充。
- `import_result`: 两个 Collection 均收到确认且 flush 成功才填写。

## Markdown 图片节点

输入 md_path/md_content；正文为空时按 UTF-8-sig 读文件；相对图片路径按 MD 文件所在目录解析为 file URI。远程链接和已有 file URI 保留，不发网络请求。缺失本地图片给出通用 warning，保留引用。代码示例不改写。

正则仅支持普通内联图片和尖括号路径；带嵌套括号、引用式图片及 HTML 图片保留原文。不提供对象存储、OCR 或多模态识别。本地 URI 不是浏览器可用的 HTTP 图片服务。

知识点：Path、URL 编码、相对路径基准、正则的适用边界、代码与正文区分。

## 文档切分节点

读取切分参数，按 Markdown 标题建立分段，短段在长度允许时向后合并，超长段按换行优先的字符窗口切分并重叠。记录原文偏移、来源和顺序，生成包含原文与来源的 SHA256 文档版本 ID。

标题/代码块识别是轻量策略，长表格或代码块仍可能跨 chunk；chunk_min_size 是合并目标，不保证每个尾片达到下限。字符数不同于 token 数，Embedding 节点另作 token 校验。

知识点：窗口推进、重叠、哈希稳定 ID、元数据与检索溯源。

## 品名识别节点

有 key 使用已有 OpenAI 兼容配置，限制超时与重试，模型只能返回原文连续出现的品名。输出不可验证或服务失败则失败，不把服务失败包装为识别成功。

无 key 依次使用正文 H1、正文型号、文件标题。规则是明确 baseline，不代表语义识别效果；多型号选择和复杂品名仍需语料评估。没有新增多模型架构。

知识点：模型输出校验、提示注入边界、规则降级、超时与重试。

## BGE-M3 节点

延迟加载 FlagEmbedding 模型并缓存，复用配置中的模型路径、device、fp16、batch_size、max_len。先用 tokenizer 检查长度，超过限制失败，避免 encode 静默丢弃尾部。品名与切片分批生成真实稠密和稀疏向量，校验数量、1024 维度、有限值、非零向量和稀疏格式。

模型加载失败、设备不可用均报错。没有假向量降级。模型调用缓存目前为单进程缓存；并发 GPU 调度需后续生产化处理。

知识点：字符与 token、批处理、模型复用、NumPy 到 Python 格式转换、向量契约。

## Milvus 节点

使用配置 URI 和两个 Collection 名，按既有注释契约要求兼容 Schema：

- 文档级：pk（VARCHAR 主键）、file_title、item_name（VARCHAR）、dense（FLOAT_VECTOR/1024）、sparse（SPARSE_FLOAT_VECTOR）。
- 切片级：chunk_id（VARCHAR 主键）、content、title、parent_title、file_title、item_name（VARCHAR）、part（INT64）、dense/sparse。
- auto_id 必须关闭，主键长度至少 64。字符串长度使用 UTF-8 字节数检查。额外非空且无默认值的必填字段会拒绝。

只检查和写入已有表，不创建 Collection、不改 Schema、不删除数据。缺表时需先提供/确认建表方案。索引由已有 Collection 提供，本节点不擅自修改索引。批量 upsert 校验服务返回数量，最后 flush。写入失败任务为 failed，连接关闭。

同来源、同内容重复导入 ID 稳定；内容或来源改变会保留另一版本，**不自动清理旧版本**。两个 Collection 没有跨表事务，部分批次可能已写入；重试同版本可补写，不能承诺回滚。来源/offset/page 等完整元数据当前保留在 State，写入字段限于以上既有 Schema。

SDK 自带 dotenv 读取会使本地 MILVUS_URI 在导入阶段被错误解析为远程默认连接。get_sdk 仅在 SDK 导入期间将进程默认 URI 屏蔽并恢复，实际客户端显式使用 settings URI；不修改 .env。

知识点：Schema 预检、确定性主键、幂等范围、批量写入确认、部分失败与跨表事务限制。

## 测试与当前环境

```bash
LOG_FILE_ENABLE=false .venv/bin/python -m pytest -q
LOG_FILE_ENABLE=false .venv/bin/python -m pytest tests/test_remaining_nodes.py -m 'not integration' -v
# 此命令会加载真实 BGE 权重，可能触发下载，需先准备设备和模型：
LOG_FILE_ENABLE=false RUN_BGE_INTEGRATION=1 .venv/bin/python -m pytest tests/test_remaining_nodes.py -m integration -v
```

主图编排测试明确 Mock 模型和数据库边界；真实 PDF、MD 读取和切分仍执行。Mock 数据仅存在 tests 中，不能证明真实模型与数据库可用。

本次检查：配置路径下未找到本地 BGE 权重；使用本地 Milvus URI，但 milvus-lite 未安装。没有下载模型、调用云端模型、安装 Lite、连接或改写实际数据库。需先准备兼容 Collection 与数据库运行依赖，再做真实全链路验收。
