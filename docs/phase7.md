# Phase 7：文件与函数学习说明

本阶段修改 5 个业务 Python 文件，新增 app/agents/critic.py；测试和演示另有更新。
Planner、模型传输、搜索适配器没有修改。Writer 与真正 Local RAG 留待后续阶段。

## app/state.py：共享数据契约

### 数据模型

- Record：所有记录的 Pydantic 基类，禁止额外字段、冻结属性、清除字符串首尾空白。冻结属性不意味着内部 list 深度不可变，节点仍需遵守不修改输入的约定。
- ResearchTask：原有 id/question/query/tool，新增 addresses_task_ids 元组。初始任务默认空元组；补搜任务通过它说明自己修复哪些旧任务。模型不能生成最终任务 ID，程序分配 ID。
- ResearchPlan：有序任务列表。现在保留首次与全部补搜任务，避免旧证据的 task_id 失去关联。
- Citation：来源标题、类型、定位地址，未改变；一个来源可能被多个任务使用。
- SearchResult：任务获得的检索片段，未改变；本身不等于已验证证据。
- Evidence：观点、原文摘录、来源和任务关联，未改变。
- ResearchOutcome：某个任务在哪一轮执行、是否提取成功，未改变；不能表示研究充分性。
- EvidenceIssue：新增。evidence_id 指向被质疑的候选证据；category 为 unsupported_claim / irrelevant / weak_source；reason 解释问题。一项证据只保留一个 issue，可在 reason 合并多个理由。
- CriticAssessment：新增。一轮评估的分数、缺失主题、问题列表、未被拒绝的证据 ID、未覆盖任务、最终 decision 与 summary。round 是评估的研究轮次，id 由程序分配。
- AgentState：新增 critic_assessments（使用 merge_records 累积）与 research_status（普通覆盖字段）。
- StateUpdate：同步新增相同字段，但 total=False，节点只返回需要更新的字段。

### ResearchPlan.unique_tasks(self)

输入是已构造的 ResearchPlan；输出是 self。通过比较任务 ID 集合大小与任务列表大小拒绝重复 ID。
原有函数没有修改，仍保证追加补搜计划时 ID 唯一。它不做问题/查询的语义去重。

### merge_records(left, right)

输入为旧记录列表和本次节点的新记录列表，输出为新合并列表。按 id 保留一份；相同 ID、相同内容
可安全重放；相同 ID、不同内容抛错。不修改输入。本阶段直接复用它存储评估历史，所以每轮评估
使用 assessment-0、assessment-1 等不同 ID，而不是修改旧评估。

### create_initial_state(user_query)

去除问题首尾空白，拒绝空问题，创建完整 AgentState。现在还初始化 critic_assessments=[] 和
research_status="running"。每次调用都创建新的列表；所有节点应使用它启动新的研究请求。

### validate_state(state)

输入为完整 State，成功返回 None，关联不合法抛 ValueError。保留原先的问题、轮次、重复记录、
current_task、来源/任务引用检查；新增以下约束：

1. addresses_task_ids 不重复，且只能引用计划中更早的任务，避免向前引用和循环依赖。
2. research_status 只能取允许值。
3. outcome 和 assessment 的 round 不能超过 retry_count。
4. assessment 的 accepted_evidence_ids 与 evidence_issues 必须引用已有 Evidence。
5. uncovered_task_ids 必须引用已知任务。
6. sufficient 必须与 decision="sufficient" 一致。

TypedDict 只是类型说明，不做运行期校验，因此节点入口、路由和最终输出仍需显式调用此函数。
旧 JSON 没有新增字段，不能直接当作当前完整 State 使用；这不是自动状态迁移实现。

## app/config.py：集中管理预算

Settings 新增 critic_max_attempts=2、critic_max_tasks=3、critic_min_score=0.7。
分别限制单次评估尝试、一轮补搜任务数，以及模型的三个评估分数阈值。
原有 max_retries=2 现在成为真正的补搜批次数上限。
这些值通过 DR_ 前缀环境变量或 .env 覆盖，Pydantic 对取值范围校验。

### load_settings()

无参数，返回 Settings()，行为未改变。延迟到入口调用，避免导入模块时就加载配置。
图、节点和测试都接收 Settings 对象，不直接读取环境变量。

## app/agents/critic.py：语义评估与确定性决策

### 数据模型与异常

- RefinementDraft：继承 Planner 的 TaskDraft，新增必填 addresses_task_ids 列表。模型返回建议任务，程序转换为 ResearchTask 并分配新 ID。
- EvaluationDraft：模型必须输出 sufficient、三项 0~1 分数、missing_topics、concerns、evidence_issues、new_tasks、summary。输出为结构化数据，不解析自然语言来控制图。
- CriticError：模型调用或结构化评估耗尽尝试后抛出。节点捕获并记录 critic_error；损坏 State 等编程错误不会冒充普通研究不足。

### query_key(tool, query)

输入工具名和查询字符串，输出 (tool, normalized_query)。casefold 统一大小写，split/join
将重复空白归一化。它让 "  RAG   COST " 与 "rag cost" 被视为同一工具查询。
不同工具允许相同查询；语义相近但字符串不同的查询不会被识别为重复。

### uncovered_tasks(state, accepted_ids)

输入已知计划、证据和本轮未被拒绝的证据 ID 集合；输出尚未有这些证据关联的任务 ID 列表。
先从 Evidence.task_id 标记覆盖，再用显式栈向 addresses_task_ids 的祖先传播覆盖。
例如 t3→t2→t1，t3 有可用证据则三个任务都计入覆盖；不会传播到无关联的兄弟任务。
这是一项最低覆盖检查，不是完整语义回答判定。父任务关联仍来自模型建议，语义适配需 Critic 判断。

### evaluate_evidence(state, model, settings, *, local_documents_available=False)

输入完整 State、结构化模型、配置与本地能力标记；输出经过校验的 EvaluationDraft。
节点调用前已经保证存在计划且批次完成。函数步骤：

1. 读取 app/prompts/critic.txt，生成 Pydantic JSON Schema，将 new_tasks.maxItems 设置为配置上限。
2. 构造模型输入：原问题、累计计划、全部候选证据、来源、全部执行结果、上一轮评估、最低覆盖缺口、工具能力、轮次和预算。
3. 在 critic_max_attempts 范围内调用 model.generate，解析 JSON。
4. 检查证据问题 ID 有效且不重复；补搜的父任务 ID 有效且不重复；本地工具已配置；任务数受限；主题与 concern 不为空白。
5. 格式错误或可重试 ModelError 提供修复反馈后重试；不可重试 ModelError 立即转为 CriticError。
6. 耗尽尝试后抛 CriticError，保留安全错误原因，不输出原始模型响应。

这个函数只评估并校验模型建议，既不调用搜索，也不修改 State 或增加研究轮次。
输入包含累计历史，目前没有完整 token 预算；上下文过长是后续工程扩展点。

### build_critic_node(model, settings, *, local_documents_available=False)

输入依赖，输出 Callable[[AgentState], StateUpdate]。构建函数不调用模型，只创建一个捕获这些
依赖的闭包。LangGraph 到达 critic 节点时才执行闭包 critic_node。

### critic_node(state)：build_critic_node 内部闭包

输入研究批次结束后的完整 State；输出评估历史增量、研究状态，以及可选的补搜计划和当前任务。

1. 校验 State，要求计划存在、current_task=None、研究状态仍是 running、本轮没有重复评估。
2. 调用 evaluate_evidence。失败则记录 decision=error、research_status=critic_error，保留全部候选证据，未确认任何 accepted ID。
3. 从当前证据中排除模型标记的问题项，得到 accepted_evidence_ids；重新计算覆盖缺口。
4. 充分性必须同时满足：模型 sufficient=True、至少一项未被拒绝证据、没有覆盖缺口、没有 missing_topics、三项分数均达到阈值。否则程序可否决模型结论。
5. 若充分：decision=sufficient、状态 sufficient、结束。
6. 若仍不足且 retry_count>=max_retries：decision=exhausted、状态 insufficient、结束。
7. 若仍有预算：按 query_key 去除历史及本批重复查询，将建议任务转换为 retry-N-task-M，并追加到原计划。设置 current_task 为第一项，retry_count 加 1、decision=retry、状态 running。
8. 若去重后无任务：decision=no_new_tasks、状态 insufficient、结束；不会消耗新轮次。
9. 写入本轮 CriticAssessment，保留模型解释及程序添加的 guard/去重说明。

评估第 0 轮时如果安排补搜，assessment.round=0，而新 State.retry_count=1，二者含义不同。
只有排入新补搜任务时才增加 retry_count。即便预算耗尽也会评估最后一轮，再明确终止。
被标记的原 Evidence 没有删除；下一轮会重新评估所有证据，提示词要求保留尚未解决的问题。
accepted 表示模型本轮未标记，并不证明真实；这是 MVP 的模型评估边界。

## app/agents/researcher.py：跨轮次推进

### record_id(prefix, *parts)

原有函数未修改。把构成标识的字符串序列 JSON 编码并计算 SHA-256，返回 prefix-hash。
用于来源、检索结果、证据与任务执行记录的稳定标识；相同材料可由 reducer 去重。

### extract_evidence(task, materials, model, settings)

原有函数未修改。对材料分配 material-N，调用结构化模型，校验数量、material_id 和原文连续摘录。
输入任务和工具材料，输出 EvidenceDraft 列表；模型/格式失败进行有上限的修复尝试。
仍然整批校验，一条摘录失败会拒绝整次提取。Phase 7 加入补搜，不改变该规则。

### build_researcher_node(model, settings, *, web_search, retrieve_documents=None)

输入模型与工具依赖，返回 researcher_node 闭包。构建时无搜索调用，节点执行时才调工具。

### researcher_node(state)

每次处理一个 current_task：校验状态、选择工具、保存 Citation/SearchResult、提取 Evidence、
记录 ResearchOutcome、选择下一任务，返回增量。预期工具/提取失败也写 outcome 并继续。
本阶段关键变化是 completed 使用所有历史 outcome.task_id，不再只看当前轮次。
因为计划现在保留旧任务，如果只看新轮次记录，就会把旧任务重新选中。
补搜任务有全新 ID，即使修复旧问题也不会被误判为执行过。outcome.round 仍记录当前 retry_count。

### next_task()：researcher_node 内部闭包

无显式参数，从闭包中的 plan 和 completed 找到第一项尚未执行的任务，输出 ResearchTask 或 None。
当前任务写入完成集合后再次调用，就推进到下一个新任务；不修改计划本身。
只执行全新 ID 是现在的调度契约；需要重新研究旧问题，应创建关联旧问题的新任务。

## app/graph.py：条件边和总步数

### route_after_research(state)

校验合并后的 State。current_task 存在返回 researcher；否则返回 critic。
前者是同一批次内任务遍历，后者是整批完成后的评估。

### route_after_critic(state)

校验 State。状态 running 且有 current_task，返回 researcher；状态终止且无 current_task，
返回 done（映射到 END）。若 running 无任务或终止却有任务则抛错，防止损坏状态静默运行。
路由函数不调用模型、不修改 State；最终决策已由 critic_node 写入。

### research_recursion_limit(settings)

输入配置，输出图总步数上限：

```text
1 Planner
+ planner_max_tasks 首轮 Researcher
+ max_retries × critic_max_tasks 补搜 Researcher
+ (max_retries + 1) Critic
+ 3 余量
```

默认值为 18。业务轮次上限先正常结束；图步数上限是异常路由的兜底，超过时抛 GraphRecursionError。
直接使用 build_research_graph(...).invoke/stream 的调用方，也应传 config 中的这个上限。

### build_research_graph(model, settings, *, web_search, retrieve_documents=None)

装配 StateGraph(AgentState)，注册 planner/researcher/critic，注入一致的工具能力。
固定边为 START→planner→researcher；研究路由指向 researcher/critic，评估路由指向 researcher/END。
输出编译后的图。补搜不重新调用 Planner，Critic 提供针对缺口的新任务。

### run_research(user_query, model, settings, *, web_search, retrieve_documents=None)

构建图，创建独立初始状态，用 research_recursion_limit 执行 invoke，校验最终 State 并返回。
返回值包括所有轮次证据和评估历史；函数不写报告、不打印结果。

## main.py：命令行入口

### main()

输入命令行参数，输出退出码。仍然加载 Settings、选择动作、装配依赖并显示结果。
新增 --demo-critic，使用离线模型演示不足→补搜→充分。--research 无需改命令即执行 Phase 7 图。
--demo-graph 仍演示两任务流程，其模拟模型增加了固定的充分评估，不产生真实研究结论。
输出 State JSON 后检查 research_status：critic_error 打印失败说明并返回 1；insufficient 打印证据
不足提示并返回 0。--check-config 增加三项 Critic 配置，phase 改为 7。
程序启动入口将 main 的返回值作为进程退出码；目前没有新增全流程日志或 Writer。

## 为什么 Critic 不只是一次 LLM 调用？

模型负责难以用简单规则判断的语义相关性、claim 支持和来源质量。程序负责模式校验、引用完整性、
最低覆盖、分数阈值、查询去重、轮次预算、任务调度、历史记录和错误终止。两层共同决定是否继续。
这不能消除模型评估错误，但避免仅凭模型一句“够了”跳过明显缺口或无限追加搜索。
