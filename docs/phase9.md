# Phase 9：所有新增与修改 Python 文件逐函数说明

本次新增 7 个业务模块、1 个 demo 和 1 个测试文件，修改 5 个业务模块。
用户原有 Writer 专用模型/超时设置保留；演示文档没有被修改。

## 1. app/rag/models.py（新增）

没有自定义业务函数，只定义共享数据：RagRecord 是 Pydantic 校验基类；DocumentPage 保存原文与
PDF 页码；LocalDocument 保存路径/标题/归档状态/版本/生效日期及页面；DocumentChunk 保存原文
片段和定位；ChunkHit 保存片段及相似度；RagError 统一导入、向量与索引错误。
这些结构独立于 AgentState，不把 FAISS 对象塞进状态，也不要求工具了解任务 ID。

## 2. app/rag/loader.py（新增）

### discover_files(paths, max_files)

输入文件/目录 Path 序列，输出去重、稳定排序的支持文件列表。展开目录，支持 .md/.txt/.pdf，
解析绝对路径，阻止目录内链接指向根目录之外，限制总文件数。显式不支持路径、空目录或缺失
路径报 RagError；目录中的其他类型忽略。它不读取正文。

### extract_metadata(text, path)

从首个非空行取标题，从演示文档头部取“状态/版本/生效日期”，输出四元组。
归档/失效/废止状态或“历史版本”标题标为 archived。只是约定格式提取，不是任意制度的时态解析。
内部 field(name)：以正则匹配中英文冒号、可选列表符号的字段行，返回字段值或 None。

### load_document(path, settings)

输入单文件和配置，输出 LocalDocument。先检查大小；文本按 UTF-8-sig 读取（支持 BOM），PDF
按页提取可用文字并保留真实页码，拒绝加密或零文本 PDF。空页跳过，不把页码重新编号。
空/损坏/编码不支持文件报安全错误，不索引猜测文字。提取元数据，并按绝对路径生成文档稳定 ID。
PDF 没有 OCR、表格结构恢复或图片阅读；复杂排版提取质量仍需人工检查。

### load_documents(paths, settings)

组合 discover_files 和 load_document，返回全部已校验文档。一个被选中的文件读失败会终止整个
导入，避免使用不完整资料却没有提示。

## 3. app/rag/chunker.py（新增）

### split_sections(text)

识别 Markdown 标题与 TXT“第 N 节”行，返回 (标题, 原文区段) 列表。保留标题和前言原文，
无标题时把全文作为一个 section。不调用 LLM，不改写材料。

### chunk_document(document, chunk_size, overlap)

先校验 0<=overlap<chunk_size，再逐页逐章节按字符窗口切分。长章节窗口重叠，短章节保持完整，
不跨页/章节，避免把现行制度和其他主题拼在同一片段。文本只去首尾空白，仍是原文连续子串。
用文档 ID、页、标题、窗口起点及内容生成稳定 chunk ID；location 保存页/章节/窗口范围/ID，
加上已解析版本和生效日期。窗口字符位置相对于章节，不是全文件偏移。
输出 DocumentChunk 列表，字符切分不是 token 切分，不保证句子或表格绝不被窗口切断。

## 4. app/rag/embeddings.py（新增）

### EmbeddingModel.embed(texts)

Protocol 接口：接收文本序列，返回同顺序向量列表。文档与查询必须使用同一个实现。

### HashingEmbeddings.__init__(dimensions=1024)

校验正维度并保存配置，无网络请求或模型下载。

### HashingEmbeddings.embed(texts)

英文/数字按词提取，中文按 1/2/3 字 n-gram 提取；用 SHA-256 确定桶位置和正负号，按对数词频
累加，最后 L2 归一化。输出确定性浮点向量。不是学习到的语义 Embedding，不擅长同义改写、
跨语言或深层语义；全标点文本产生零向量，可触发空检索。

### ApiEmbeddings.__init__(settings, *, transport=None)

要求独立 embedding_api_key，保存配置和可选 HTTP transport，构造时不调用 API。

### ApiEmbeddings.embed(texts)

按 embedding_batch_size 分批，逐批调用 _embed_batch，拼接为与输入一致的向量列表。
空输入自然返回空列表。不同批次维度是否一致最终由矩阵校验检查。

### ApiEmbeddings._embed_batch(texts)

调用配置地址的 /embeddings，发送 model/input，不强行声明所有供应商都支持的维度参数。
按 data.index 还原输入顺序，要求索引完整、不重复、向量矩阵非空/有限且不为零。
网络、响应格式、429/408/5xx 有限重试；401 等不可重试 HTTP 立即失败，不输出原始服务响应或密钥。
输出列表；超出尝试次数报 RagError。尚未实现指数退避/Retry-After。

### build_embeddings(settings)

按 embedding_provider 返回 HashingEmbeddings 或 ApiEmbeddings。当前默认 API BGE-M3；
哈希仅用于离线模式。BGE-M3 校验 1024 维；其他 API 模型先按预期维度或首批响应锁定维度。
embedding_dimensions 只控制 hashing，具体补充见 bge-m3.md。

## 5. app/rag/vector_store.py（新增）

### normalized_matrix(vectors, rows, dimensions=None)

将向量转成 float32 连续矩阵，校验行数、非空维度、有限数值，以及可选的预期维度，再 L2 归一化。
返回适合 FAISS 的 NumPy 数组；非法结果报 RagError。

### FaissVectorStore.__init__(chunks, embeddings)

拒绝空片段，保存片段元数据，调用 EmbeddingModel 构建文档向量，创建 IndexFlatIP 并 add。
IndexFlatIP 是精确内积搜索；归一化后内积等价于余弦。适合 MVP 小文档集，没有近似索引参数。

### FaissVectorStore.search(query, *, top_k, min_score)

检查查询与阈值，生成并归一化查询向量，确认维度与索引一致。零查询向量返回 []。
FAISS 返回最近邻位置与分数，位置映射回 DocumentChunk，仅返回达到阈值的 ChunkHit。
分数是向量相似度，不是事实可信度；当前不做 rerank、关键词融合或正确答案判定。

## 6. app/tools/retriever.py（新增）

### build_retriever(store, *, top_k, min_score)

绑定已构建向量库和检索参数，输出 retrieve_documents(query) 函数；构建时不检索。

### retrieve_documents(query)：内部闭包

空查询报 ToolError(invalid_input)；调用 store.search，把 RagError 转成统一 ToolError。
将 ChunkHit 转成 ResearchMaterial：local/local_chunk、原文、完整路径和位置。
不传 task_id，不判断充分性，不生成引用 ID。Researcher 复用原有证据与来源逻辑。

## 7. app/rag/pipeline.py（新增）

LocalRag 冻结数据类保存文档、向量库、检索函数和排除的归档数量，是图外的依赖容器。

### LocalRag.describe_documents()

输出文件名、标题、版本、生效日期、归档标记和排除数量，作为 Planner/Critic 的文档清单。
不给 Planner 塞全部正文；这些描述仍是不可信数据，提示词禁止把它当指令。

### build_local_rag(paths, settings, *, include_archived=False, embeddings=None)

加载文档→默认排除归档→切块→构建 Embedding/FAISS→包装 Retriever，返回 LocalRag。
可注入自定义 EmbeddingModel；过滤后零文档报错。实际返回条数取 min(rag_top_k, search_top_k)，
与现有 Researcher 的材料预算一致。每次执行重建内存索引，没有持久化/增量更新。

## 8. app/config.py（修改）

Settings 增加 chunk size/overlap、检索数量和阈值、文件数量/大小预算、Embedding provider/model/
URL/独立 key/维度/批次/超时/尝试数。当前默认 API；切换 hashing 不需要 Embedding key。

### Settings.valid_chunk_overlap(self)

Pydantic 模型校验器，要求 overlap 小于 chunk_size，防止窗口不能向前推进，返回 self。

### load_settings()

按需返回 Settings()，行为不变。用户已有 Writer 独立超时设置没有覆盖。

## 9. app/agents/planner.py（修改）

### plan_research(..., local_document_context="", web_search_available=True)

原有计划生成与结构校验保留。新增文档清单输入和 Web 可用性说明；local-only 时模型返回 Web
任务会被拒绝并有限修复。本地事实优先中文检索，公共研究仍可选 Web。
文档清单只是选工具上下文，不意味着 Planner 已经阅读文档或确认答案存在。

### build_planner_node(...)

捕获新增清单/工具能力，返回 planner_node。

### planner_node(state)：内部闭包

原有初始状态检查不变；调用 plan_research 时传入新增参数，返回计划与第一个任务。

## 10. app/agents/critic.py（修改）

### query_key(tool, query)

原有查询大小写/空白去重规则，未修改。

### uncovered_tasks(state, accepted_ids)

原有最低证据覆盖与补搜祖先传播，未修改。

### evaluate_evidence(..., local_document_context="", web_search_available=True)

available_tools 现在反映真实 Web 开关，并提供文档清单；local-only 时拒绝 Web 补搜建议。
提示词新增现行/历史、预算/支出、试点/生产的区别，避免把未知企业事实通过公开搜索补写。
其余结构/引用/评分与有限修复逻辑保留。

### build_critic_node(...)

捕获新增文档清单和可用工具开关，返回 critic_node。

### critic_node(state)：内部闭包

评估、预算控制、追加补搜和终止机制不变。调用评估函数时传入真实工具能力和文档清单。

## 11. app/graph.py（修改）

### route_after_research(state)

继续本批任务或进入 Critic，未修改。

### route_after_critic(state)

补搜或进入 Writer，未修改。

### research_recursion_limit(settings)

按 Planner/最大研究任务/Critic/Writer 计算图步数上限，未修改。RAG 在工具内部发生，不新增图节点。

### build_research_graph(..., local_document_context="", web_search_available=True)

文档清单和工具开关注入 Planner/Critic。禁止 Web 且未配置 Retriever 时拒绝构图，确保至少一个工具。
Researcher 仍接受相同 retrieve_documents 函数，无需知道 FAISS 或 Embedding 实现。

### run_research(...)

把新增参数传给图装配，其他初始化、执行、最终校验不变。保留 writer_model 参数和用户已有超时能力。

## 12. main.py（修改）

### build_live_search(settings, *, enabled=True)

返回延迟装配的 search 函数，本地任务无需搜索 key。构建时不创建 Tavily 后端。

### search(query)：内部闭包

Web 被禁用时直接 ToolError；允许时才构建后端并调用已有搜索包装器。保留既有搜索重试逻辑。

### main()

新增 --documents（文件/目录）、--retrieve（仅检索）、--include-archived、--local-only、--demo-rag。
校验参数组合；读取 Settings 后在图外构建 RAG。retrieve 输出文档/片段/排除数量和材料 JSON。
真实研究注入检索函数、文档清单及 Web 开关，现有报告输出/导出行为保留。
demo-rag 强制 hashing，使用 examples/doc 或显式指定路径，不调用任何 LLM/搜索/Embedding API。
local-only 只禁用 Web Search；真实研究仍调用模型，API Embedding 也会发文档/查询到配置服务。
导入/检索失败显示安全错误并返回 1，check-config 标记 Phase 9 并显示 RAG 核心参数。

## 13. examples/rag_demo.py（新增，demo）

### DemoRagModel.generate(system, user, schema)

按 Schema 区分规划/提取/写作/评估请求。规划固定两个本地任务；提取把实际检索片段原样作为
claim/excerpt；写作每个任务选一个片段展示；评估复用固定模拟实现。
它验证真实 RAG 与图连接，不代表真实 LLM 的推理或充分性判断效果。

### run_demo(settings, rag)

把真实 RAG 和固定模型注入 run_research，以 local-only 执行，返回可序列化 State。

### disabled_search(query)：内部闭包

抛 AssertionError，使演示一旦错误调用 Web 立即失败。正常流程不会执行到这里。

## 14. tests/test_rag.py（新增，测试）

| 函数 | 验证内容 |
| --- | --- |
| settings(**changes) | 构造独立测试配置，不读 .env |
| test_demo_document_ingestion_and_metadata | 5 份文档加载、版本日期、默认归档过滤 |
| test_known_current_policy_and_budget_retrieve_correct_sections | 真实 FAISS 命中现行 30/180 天与 18,000 元预算 |
| test_archived_document_requires_explicit_opt_in | 默认排除、显式纳入旧版 60/90 天 |
| test_chunk_windows_are_grounded_bounded_and_ids_are_stable | 片段是原文子串、长度受限、ID 稳定、坏 overlap 拒绝 |
| test_empty_invalid_unreadable_and_oversized_inputs | 空目录/文件、坏编码/PDF、缺失与超限失败 |
| test_file_deduplication_and_archive_only_input | 重复路径去重、只有归档文档时明确失败 |
| test_pdf_text_and_page_locations_and_scanned_pdf | 构造两页文本 PDF，校验真实页码；空文本 PDF 拒绝 |
| test_hashing_is_deterministic_and_cosine_threshold_returns_empty | 离线向量稳定、单位长度、阈值/零向量空返回 |
| test_api_embeddings_batch_order_and_retry | Mock HTTP 分批、返回乱序、429 后成功 |
| test_embedding_failures_and_index_dimensions | 缺失 key、坏索引/零/NaN、401 不重试、查询维度不一致 |
| test_local_demo_runs_graph_and_preserves_real_file_citations | 本地整图、原文证据和存在的本地文件引用 |
| test_mixed_graph_chooses_local_and_web_tools | 同一图执行 Local 与 Web，并让 Planner 看到清单 |
| test_local_only_rejects_planner_web_tasks_and_lazy_web_needs_no_key | 禁用工具执行边界与 Planner 输出校验 |

嵌套函数：handler(request) 模拟第一次429与后续乱序向量；unauthorized(request) 记录401只调用一次；
BadModel.embed(texts) 故意让文档/查询维度不一致；MixedModel.generate(**kwargs) 验证清单并输出
一条本地、一条 Web 任务，其余复用固定模型；HTTP lambda 只生成无效响应，不访问网络。
