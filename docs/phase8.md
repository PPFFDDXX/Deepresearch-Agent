# Phase 8：逐模块与逐函数说明

本阶段新增 report_models.py、citations.py、agents/writer.py；修改 state.py、graph.py、config.py、main.py。
其余业务文件没有修改，下面也说明它们在完整流程中的作用。测试与 demo 文件不在本说明范围。

## 一、报告模型：app/report_models.py（新增）

这个模块只定义报告数据，不依赖 AgentState、不调用 API，避免 State 和 Writer 之间循环导入。

- ReportRecord：报告模型的基类，禁止额外字段、冻结属性、清理字符串首尾空白。内部列表并非深度不可变。
- CitedParagraph：text 是一段正文；evidence_ids 至少包含一个 ID。每段限 3000 字符、最多 10 个证据 ID。
- AnalysisSection：详细分析的小节标题和段落列表，避免模型把整篇报告塞进一个字符串。
- ReportDraft：模型生成的四部分正文：executive_summary、key_findings、detailed_analysis、conclusion。没有 sources、URL 或引用编号字段。
- ReportCitation：程序创建的展示引用，包括 number、citation_id、title、source_type、locator、locations。

### CitedParagraph.unique_evidence_ids(cls, value)

Pydantic 字段校验器。输入一个段落的证据 ID 列表，拒绝重复 ID 和空白 ID，返回原列表。
它不判断 ID 是否来自当前研究，成员有效性由 citations.validate_report 完成。

### iter_paragraphs(report)

输入 ReportDraft，按实际阅读顺序依次 yield 摘要、发现、分析各小节、结论的 CitedParagraph。
这是生成器，不复制整个段落列表。验证、来源编号和 State 关联检查共享这一个遍历规则。

## 二、引用与渲染：app/citations.py（新增）

这个模块不调用模型。来源编号、已有来源元数据和报告排版全部由应用程序控制。

### approved_evidence(state)

输入完整 AgentState，输出最后一轮评估放行的 Evidence 列表，顺序沿用 State。
先调用 validate_state，再要求研究已经终止、当前任务为空、最后评估存在且轮次和状态一致。
critic_error 返回空列表，不能把未评估的候选证据当作可用。其他情况按最后评估的
accepted_evidence_ids 过滤，不回退使用早期评估或所有历史证据。

### validate_plain_text(text)

输入模型正文或小节标题，成功返回 None；包含方括号、尖括号、HTTP URL、www、doi:/arxiv:/file://
等格式时抛 ValueError。模型应只写纯文本，并通过 evidence_ids 表达依据。
这是保守的格式规则，不是事实检查器，不能发现自然语言里所有编造的人名、论文或事实。

### validate_report(report, evidence)

输入已解析 ReportDraft 和允许使用的 Evidence。遍历全部正文，检查纯文本、检查每个 evidence_id
是否属于允许集合，也检查分析小节标题。空引用或重复 ID 已由 Pydantic 拒绝。
不会自动证明段落与证据的语义蕴含；现阶段靠提示词、最终 Critic 的候选筛选和明确局限控制。

### build_report_citations(report, evidence, citations)

输入草稿、可用证据和已收集来源，输出 ReportCitation 列表。先 validate_report，再按正文首次
出现的 Evidence ID 找到 Citation ID，以有序字典为来源编号。多个证据指向相同来源只保留一个编号。
来源 title/locator/source_type 直接复制已有 Citation，locations 来自实际引用证据且去重。
未在正文使用的搜索来源不进入 Sources。Web 地址必须是有效的 HTTP(S) 形式；本地路径作为文本显示。

### escape_markdown(text)

输入不可信文本，归并换行和空白为单行，转义 Markdown/HTML 相关字符，返回用于展示的文本。
用于用户问题、模型正文、来源标题和本地位置。原始结构化数据保持不变，只在渲染边界转义。

### render_limitations(state, evidence)

输入研究状态和本报告可用证据，输出 Limitations 小节 Markdown。
固定说明 Critic 分数不是事实认证；不足/评估失败时注明不能给出确定答案；使用搜索片段时说明未核验
全文；无可用证据时说明没有证据放行。同时展示实际 missing_topics、未覆盖任务和 Critic concerns。
这些是运行状态和模型评估说明，明确标注来源，不作为额外研究结论。

### render_report(report, state, evidence)

输入合法草稿、状态和可用证据，输出 (Markdown 文本, ReportCitation 列表)。
先构建来源编号，依次生成五个要求的小节和 Limitations。研究不足时在报告顶部强制添加 PARTIAL REPORT。
Sources 的 Web 地址由程序编码并生成 Markdown 链接；本地来源显示路径及位置。

内部 paragraph_text(paragraph)：从段落 evidence_ids 查到来源编号，按段内首次出现去重，
返回“转义后的正文 + [N] 标记”。例如同一来源的两个 Evidence 只显示一个 [1]。
这一闭包不调用模型，捕获的是当前报告的证据索引与编号映射。

### render_unavailable_report(state, *, writer_error=False)

不调用模型，返回含五个要求小节的说明性报告。没有放行证据时说明无法生成有依据的答案；
Writer 失败时说明没有发布未经验证的草稿。不生成 Sources 条目或研究结论。
写作失败时仍把实际可用证据传给局限渲染，避免错误声称“没有任何证据”。

## 三、Writer：app/agents/writer.py（新增）

WriterError：在有限尝试内没有获得合法报告草稿的异常，节点会将它转为 writer_error 状态。

### write_report(state, evidence, model, settings)

输入状态、最终放行证据、模型和配置，输出 ReportDraft；零证据直接拒绝。
读取 writer.txt，给模型提供原问题、研究状态、可用 Evidence、缺失主题、未覆盖任务和 Critic concerns。
不把 Sources 元数据交给模型生成；Evidence 中的 citation_id 只作为内部关联信息。
在 writer_max_attempts 内调用 generate、解析 Pydantic 草稿、运行 validate_report。
无效 JSON、无效引用或可重试 API 错误会提供修复反馈；不可重试 API 错误立即失败；耗尽后抛 WriterError。
函数不修改 State，也不输出未经验证的模型结果。当前没有独立的模型语义复核或完整 token 预算。

### build_writer_node(model, settings)

接收依赖，返回 writer_node 闭包供 StateGraph 注册；构建时不调用模型。

### writer_node(state)：内部闭包

输入 Critic 终止后的完整状态，输出报告字段的 StateUpdate。
1. 用 approved_evidence 校验并筛选候选证据，要求 report_status=not_started，防止重复写作。
2. 无可用证据：直接生成说明性报告，report_status=no_evidence，不调用模型，不发布引用。
3. 有证据：write_report → render_report，生成正文、编号和来源元数据。
4. WriterError：保留研究历史，生成失败说明，report_status=writer_error，记录安全 report_error。
5. 成功：研究充分为 generated，研究不足为 partial；保存 report_draft、report_citations、final_report。

Writer 不改变 research_status、不增加 retry_count。写作成功与研究充分性是两种不同状态。
损坏 State 或来源元数据等程序问题不会伪装成普通模型写作失败。

## 四、State：app/state.py（修改）

原有模型职责：Record 为校验基类；ResearchTask/ResearchPlan 保存累计任务；Citation 保存来源；
SearchResult 保存检索材料；Evidence 保存 Claim 与 Excerpt；ResearchOutcome 保存任务执行情况；
EvidenceIssue/CriticAssessment 保存评估问题及每轮决策。
新增报告字段：report_status、report_draft、report_citations、report_error。StateUpdate 同步增加可选更新字段。
报告结果只生成一次，使用普通覆盖字段，不使用历史记录 reducer。研究证据/评估仍使用 reducer 累积。

### ResearchPlan.unique_tasks(self)

检查任务 ID 唯一，返回 self；原有逻辑不变。补搜计划保留旧任务时仍保证关联标识不冲突。

### merge_records(left, right)

按 id 合并旧记录和新增记录，相同记录重放可去重，相同 ID 内容冲突则抛错，不修改输入。
本阶段没有改动它；报告引用列表不用这个 reducer，因为展示编号属于本次最终报告。

### create_initial_state(user_query)

清理问题、拒绝空问题、创建独立状态。增加 report_status=not_started、report_draft=None、
report_citations=[]、report_error=None，原有 final_report=None 保留。

### validate_state(state)

保留原有任务/来源/轮次检查，新增“评估不能同时接受和拒绝同一证据”。
检查报告状态合法、未开始时没有结果、已生成时研究已终止且有 final_report。
writer_error 必须有错误原因；正常结果不能带错误。no_evidence/writer_error 不得发布结构化草稿和引用。
generated 要求研究 sufficient，partial 要求研究 insufficient，二者必须存在 ReportDraft。
草稿引用必须属于最后一轮 accepted 集合；引用编号连续、来源不重复、只包含实际用到的来源；
展示标题/地址/类型必须与收集到的 Citation 完全一致。
这些检查保护结构关联，不对 Markdown 中每句自然语言做事实验证。

## 五、图：app/graph.py（修改）

### route_after_research(state)

校验合并后的状态；有当前任务继续 researcher，否则进入 critic。本阶段没有改动。

### route_after_critic(state)

运行中且有任务返回 researcher；终止且无任务现在返回 writer，以前是 done。
状态与任务矛盾则抛错。Critic 失败也进入 Writer，但 Writer 会跳过模型并生成无法评估的说明。

### research_recursion_limit(settings)

总步数预算现在加上 1 次 Writer。公式为 1 Planner + 最大首次任务数 + 最大补搜任务数 +
(max_retries+1) Critic + 1 Writer + 3 余量，默认 19。
写作 API 的两次尝试发生在节点内部，不等于两个图步骤。

### build_research_graph(model, settings, *, web_search, retrieve_documents=None)

注册四个节点 planner/researcher/critic/writer，复用模型与配置注入。
批次遍历和 Critic 补搜保留，新增 critic→writer 与 writer→END。
输出编译图，装配时无网络请求。

### run_research(user_query, model, settings, *, web_search, retrieve_documents=None)

创建图和独立初始状态，设置总步数预算执行 invoke，验证最终状态后返回。
现在返回值包含最终 Markdown、结构化草稿、展示引用和原研究审计记录。

## 六、配置：app/config.py（修改）

Settings 集中定义模型、工具、任务和各阶段尝试/补搜预算；Pydantic 从 DR_ 环境变量和 .env 加载。
新增 writer_max_attempts=2，可用 DR_WRITER_MAX_ATTEMPTS 覆盖，允许 1~3。
它只限制写作请求/输出修复，不会导致新的研究轮次。

### load_settings()

返回 Settings()，逻辑不变。配置在命令入口按需加载，不在导入模块时加载。

## 七、入口：main.py（修改）

### main()

解析互斥的操作参数，加载配置，选择真实或离线依赖；Phase 8 增加 --report-only 和 --output FILE。
两个输出选项只能与 --research/--demo-graph/--demo-critic 使用，不合法组合由 argparse 拒绝。
图执行完后，将 State 中 Pydantic 对象转为字典用于 JSON 输出。--report-only 则直接打印 final_report。
--output 创建父目录并以 UTF-8 写 Markdown，指定已有文件时覆盖；包括降级/失败说明报告。
writer_error 或 critic_error 返回 1，insufficient 输出提示后返回 0；文件读写失败由 OSError 分支处理。
check-config 输出增加 writer_max_attempts，阶段改为 8。原有规划、搜索、单任务入口不变。
文件末尾的 __main__ 分支将 main 返回值作为进程退出码，它不是额外函数。

## 八、Planner：app/agents/planner.py（未修改）

TaskDraft 是模型建议的 question/query/tool；PlanDraft 限制任务列表；PlannerError 表示规划失败。

### plan_research(user_query, model, settings, *, local_documents_available=False)

加载 planner.txt，调用模型输出结构化任务。检查数量、重复问题和工具能力；格式错误与可重试
模型错误进行有限修复。程序分配 task-N ID，返回 ResearchPlan，不返回不完整计划。

### build_planner_node(model, settings, *, local_documents_available=False)

捕获模型依赖，返回 planner_node。

### planner_node(state)：内部闭包

校验初始状态并要求尚无计划，调用 plan_research，返回 research_plan 和第一个 current_task。
不修改输入，不参与补搜；补搜归 Critic 管理。

## 九、Researcher：app/agents/researcher.py（未修改）

EvidenceDraft 保存 material_id/claim/excerpt；ExtractionDraft 保存候选列表；ResearcherError 表示提取失败。

### record_id(prefix, *parts)

对组成标识的字符串序列做 JSON 编码和 SHA-256，返回稳定 prefix-hash，用于来源和证据去重。

### extract_evidence(task, materials, model, settings)

为材料分配 material-N，模型提取 EvidenceDraft；检查原文连续摘录、材料 ID、数量，有限重试。
整批校验，一条失败可能导致整批拒绝。返回草稿列表，不决定充分性。

### build_researcher_node(model, settings, *, web_search, retrieve_documents=None)

装配模型和工具，返回 researcher_node。

### researcher_node(state)：内部闭包

每次执行当前任务，按 task.tool 调用 Web Search 或本地检索，标准化材料，保存来源和检索结果，
提取证据、记录 outcome，选择下一任务。所有历史已执行 ID 都跳过；补搜采用新 ID。
预期工具/提取失败会记录并推进；节点只返回增量。

### next_task()：内部闭包

扫描累计计划，返回第一个不在已完成 ID 集合中的任务，没有则返回 None。
它读闭包中的计划与 completed，不单独维护另一份调度队列。

## 十、Critic：app/agents/critic.py（未修改）

RefinementDraft 建议补搜及关联；EvaluationDraft 是评分/问题/任务建议；CriticError 是评估失败。

### query_key(tool, query)

统一查询大小写与空白，返回工具和查询键，仅做精确字符串去重。

### uncovered_tasks(state, accepted_ids)

按可用证据任务关联与补搜祖先关系计算最低覆盖缺口。

### evaluate_evidence(state, model, settings, *, local_documents_available=False)

将原问题、累计计划/证据/来源/outcome、上次评估与预算传给模型，结构化校验评分、引用和任务能力，
有限修复后返回 EvaluationDraft。不执行补搜。

### build_critic_node(model, settings, *, local_documents_available=False)

返回注入依赖的 critic_node。

### critic_node(state)：内部闭包

只在批次完成后评估；排除问题证据、检查覆盖/阈值/缺失主题，决定充分、补搜、预算耗尽、无新任务
或评估失败。补搜追加计划并增加 retry_count，每轮记录 CriticAssessment。Writer 使用最后这条结果。

## 十一、模型传输：app/llm.py（未修改）

### ModelError.__init__(message, *, retryable=False)

保存安全错误说明和可重试标记，业务 Agent 决定是否再尝试。

### StructuredModel.generate(system, user, schema)

Protocol 的接口声明，不做具体实现；所有 Agent 依赖同一个结构化文本接口，因此能替换模型或离线实现。

### ChatCompletionsModel.__init__(settings, *, transport=None)

检查模型配置，保存 Settings 和可选 HTTP transport，不发请求。transport 允许测试注入模拟 HTTP。

### ChatCompletionsModel.generate(system, user, schema)

构造 Chat Completions HTTP 请求；当前 GLM 分支使用 JSON mode 并在提示中附 Schema，其他分支
按现有适配器使用 JSON Schema 格式。处理超时、网络、HTTP、拒绝、截断和响应结构错误。
返回 JSON 文本，不判断研究或报告质量；各节点自行解析自己的 Pydantic 模型。
当前跨供应商兼容性以既有适配器实现为准，并非任意模型均支持严格 Schema。

## 十二、工具数据：app/tools/models.py（未修改）

- WebSearchHit：校验标题、HTTP URL 和 snippet。
- WebSearchResponse：查询、供应商、结果列表和尝试次数，空结果合法。
- ResearchMaterial：统一 Web/Local 材料，包含来源、正文、位置和 origin。

### ToolError.__init__(code, message, *, retryable=False, attempts=0)

保存错误类别、说明、是否可重试及已尝试次数。Researcher 据此记录任务失败，不依赖供应商异常类型。

## 十三、搜索适配器：app/tools/web_search.py（未修改）

### SearchBackend.search(query, *, top_k, timeout)

Protocol 声明一次搜索尝试的接口，返回校验过的 WebSearchHit 列表或 ToolError。

### TavilySearchBackend.__init__(settings, *, transport=None)

检查搜索配置并保存凭据/传输依赖，无搜索调用。

### TavilySearchBackend.search(query, *, top_k, timeout)

发起单次 Tavily 请求，返回标题/URL/片段；不请求原始全文或供应商生成答案。
将超时、网络、HTTP、无效结果转换为统一 ToolError。

### build_web_search(backend, settings)

绑定供应商与预算，返回独立 web_search(query) 函数，Agent 不需要了解 Tavily。

### web_search(query)：内部闭包

检查非空查询，有限尝试调用 backend，按去除 fragment 的 URL 去重、限制 top_k，包装结果与尝试次数。
空列表直接返回，不盲目重复同一查询；改写搜索问题由 Critic 负责。

## 十四、包入口与 RAG 占位（未修改）

app/__init__.py、app/agents/__init__.py、app/tools/__init__.py、app/rag/__init__.py
用于组织 Python 包，没有执行研究流程的业务函数。rag 当前是后续 Phase 9 的目录占位，
已有 retrieve_documents 注入接口并不等于已经实现 Document→Chunk→Embedding→Vector Store。
