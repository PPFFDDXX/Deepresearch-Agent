# BGE-M3 / Phase 9 接入补充

保留用户设置的服务地址和密钥，模型修正为 SiliconFlow 的 BAAI/bge-m3。
仅完善 Phase 9，未修改正在开发的 main.py / Phase 10 入口。
本次独立测试和样例验证均为离线；真实服务验证需要发送样例文档片段到配置服务。

## 配置

```dotenv
DR_EMBEDDING_PROVIDER=api
DR_EMBEDDING_MODEL=BAAI/bge-m3
DR_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
DR_EMBEDDING_API_KEY=你自己的密钥
```

官方接口接受 model/input/encoding_format。请求明确使用 encoding_format=float，不发送不适用
BGE-M3 的自定义 dimensions 参数。模型 dense 输出为 1024 维；不是启用所有 sparse/ColBERT 模式。
参考： https://api-docs.siliconflow.cn/docs/api/embeddings-post
模型说明： https://huggingface.co/BAAI/bge-m3

DR_EMBEDDING_DIMENSIONS 只影响 hashing。DR_EMBEDDING_EXPECTED_DIMENSIONS 是可选 API 校验，
BGE-M3 只能为 1024 或未设置，其他模型可明确指定。当前 FAISS 会随每次运行重建，避免新旧模型混用。
默认 0.05 阈值没有经过真实 BGE-M3 校准；不得根据离线哈希或 Mock 响应宣称实际模型准确率。

## 修改 app/config.py

Settings 的 embedding_model 默认值修正，增加 embedding_expected_dimensions，密钥类型允许 None，
这样构造配置成功后可由 API 适配器给出明确缺失密钥错误；用户已有密钥默认值没有替换。
valid_chunk_overlap、load_settings 的函数行为不变。

## 修改 app/rag/embeddings.py

- ApiEmbeddings.__init__：检查 key，识别 BAAI/bge-m3 与 Pro/BAAI/bge-m3，设定 1024 维，并拒绝冲突的显式维度。其他模型使用配置的预期维度或暂不锁定。
- ApiEmbeddings.embed：拒绝单个字符串、空白文本和非字符串元素；空列表不发请求。保留分批，文档向量与查询向量使用同一实例和同一维度。
- ApiEmbeddings._embed_batch：明确请求 float，保留输入数组、Bearer 与响应索引排序。校验实际维度；未声明维度的其他模型由首批建立，后续批次和查询必须一致。维度冲突立即报安全错误，不把向量传给 FAISS。
- EmbeddingModel.embed 是接口声明；HashingEmbeddings.__init__/embed 和 build_embeddings 行为不变。hashing 仍是显式离线实现，不伪装为 BGE-M3。

## 修改 tests/test_rag.py

settings(**changes) 显式默认 hashing 和 fixture 模型，防止业务默认改为 API 后普通单元测试意外联网。
其余测试函数未修改。它仍依赖 main.py；Phase 10 入口缺失依赖时，这个旧测试文件会导入失败，
因此提供了独立 BGE 测试。没有通过删除旧测试或改写 Phase 10 入口掩盖这一情况。

## 新增 tests/test_bge_embeddings.py

- settings：构造测试用 API 配置与 fixture key，不读 .env。
- response：用离线哈希生成指定维度向量，再包装为乱序的服务响应；只模拟 API 契约，不模拟 BGE 语义效果。
- test_siliconflow_contract_and_batch_order：验证 endpoint、Bearer、model、float、无 dimensions 参数，以及乱序还原和分批。
- test_bge_wrong_dimensions_fail_before_indexing：768 维输出被两个 BGE 模型名拒绝。
- test_incompatible_bge_expected_dimension_is_rejected：配置与固定维度冲突时失败。
- test_generic_model_locks_dimensions_across_calls：其他模型首批 128、下次 256 时失败。
- test_generic_model_expected_dimension_is_enforced：显式预期 128 与返回 256 不一致时失败。
- test_empty_input_validation_never_sends_requests：非法输入不触发 HTTP；空列表返回空。
- test_missing_key_fails_without_leaking_details：缺失 key 给出配置提示。
- test_api_selection_and_offline_override：按 provider 正确选择实现。
- test_timeout_retry_is_bounded：超时只尝试预算次数。
- test_mocked_bge_transport_connects_documents_faiss_and_queries：真实文档→Mock HTTP dense 响应→FAISS→查询，验证 4 份、20 chunks、1024 维及三项样例。此结果不能作为真实模型检索效果。

嵌套 handler 函数分别验证请求、模拟超时、记录实际文档/查询批次；forbidden 在非法输入发请求时
使测试失败。HTTP lambda 仅返回准备好的 Mock 响应，没有真实网络调用。
