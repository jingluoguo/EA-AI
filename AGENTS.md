# Agent Instructions

## 结构智能约束

本项目的目标不是穷举自然语言问法，也不是把每个失败样例补成一条端到端模板。运行时不保留问法表、正则 parser 或字符串 fallback；后续演进必须以训练数据和可替换学习能力为中心。实现和修改 `struct_llm` 时，必须遵循这条链条：

```text
观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证
```

具体要求：

- 观察现象：先收集失败输入、已有结构输出、期望答案，确认失败发生在陈述解析、问题解析、规则推导还是答案生成。
- 训练优先：遇到新失败样例时，先补训练/反馈数据，记录 `observation`、上下文、已知世界状态、信念状态、期望 `INTENT/QUERY/STATE` 或答案，再用数据集评估现有能力；只有数据证明结构标签缺失或模型槽位不足时，才改 parser、state、query 或 inference。
- 学习主路径：陈述和 Query 默认都必须走神经输入模型，样本 schema 和标签构造仍放在 `comprehension/statement.py`、`comprehension/query.py`。JSONL 数据集只作为训练/反馈输入；默认运行时优先加载 `data/statement_neural_model.pt/json`、`data/query_neural_model.pt/json` 这类神经能力产物。新增失败样例先进入数据集、重新训练并评估，不新增问法分支。
- 剥离次要因素：把主动句、被动句、换序、语气词、时间副词等表层差异视为可归一化因素，不能直接把这些差异变成互不相关的答案规则。
- 数据集边界：意图分析样本放入 `data/intent_examples.jsonl` 或同 schema 的 JSONL 文件；Query 样本放入 `data/query_examples.jsonl` 或同 schema 的 JSONL 文件；陈述样本放入 `data/statement_examples.jsonl` 或同 schema 的 JSONL 文件。数据读写、校验、编译和模型加载放在对应 dataset/learning 模块；样本消费、模型编译或模型预测放在 `comprehension/intent.py`、`comprehension/query.py`、`comprehension/statement.py`。后续新增学习型能力也应先有 dataset/schema，再接训练产物或推理实现。
- 槽位规范化：从句子中抽出的实体、事件角色、查询目标必须先清理首尾语气词、时间副词和“我想知道”这类提问框架，再用已知实体表校正槽位边界，然后才进入结构匹配，避免把“你可以告诉我芯片”、“我想知道芯片”或“托盘的了”之类表层残留当成实体。
- 查询候选：输入里可能有寒暄型疑问和真正任务片段，例如“你知道吗？你知道的话，可以告诉我……”。解析时必须保留所有非陈述片段和无结尾标点的尾段，从候选集合中寻找可计算核心子句，不能只解析第一个问号片段。
- 顺序状态：陈述句必须按原文顺序处理。`FRAME` 保留历史事件，`ROLE` 表示事件角色，`STATE` 表示当前世界状态；后发生的放入、移动、交接、涂色必须覆盖同一对象的旧当前状态。主动/被动移动都要归一为当前 `STATE at(thing, place)`。历史事件查询不能依赖当前状态反推旧事件槽位，事件 frame 自身要保留必要角色约束。
- 构建理想模型：优先把语言归一到实体、`FRAME/ROLE`、`STATE`、`QUERY`、规则等中间结构。`EVENT/REL` 只能作为兼容线性化视图，不能成为新增推理能力的主模型。例如“谁把芯片放进托盘？”、“芯片是谁放进托盘的？”、“芯片被谁放进托盘的？”应归一为同一个事件角色查询。
- 数学表达：让推理依赖可计算的 frame 角色匹配、状态查询、关系闭包、事件角色约束或状态转换，而不是依赖整段文本正则。
- 模块边界：新增切句/候选保留能力放在 `perception/lexer.py`；新增表层剥离和槽位清理放在 `perception/normalizer.py`；新增指代消解放在 `perception/reference.py`；新增陈述样本消费放在 `comprehension/statement.py`；新增 Query 样本消费和模型替换放在 `comprehension/query.py`；新增意图样本消费放在 `comprehension/intent.py`；新增状态转移放在 `world/state.py`；新增事件 schema 放在 `world/event_schema.py`；新增条件规则和因果展开放在 `world/causal.py`；新增规则和答案优先放在 `reasoning/core.py`，`reasoning/pipeline.py` 只保留门面，后续按领域拆入 `reasoning/rules/` 和 `reasoning/answers/`；新增置信度策略放在 `metacognition/confidence.py`；新增反馈和对话输出放在 `motor/feedback.py`、`motor/dialogue.py`。`capabilities.py` 负责定义认知内核可插拔能力接口和组合方式；`kernel.py` 是唯一结构推理流水线，负责串联感知、理解、世界状态、推理、元认知和输出层。
- 能力注册：陈述学习、状态投影、状态覆盖、Query 学习、规则推导、答案生成都必须以小能力函数注册到 `CognitiveCapabilities`。新增能力时优先更新数据集或替换学习能力，不能为了一个新问法改成端到端特殊分支。
- 实验验证：每次新增能力都要加入覆盖主动、被动、换序或同义表达的测试，证明它们映射到同一个中间结构。训练相关改动还必须测试数据 loader、反馈写入、schema 校验或评估路径，不能只测试某条句子是否答对。

不得新增表层问法 pattern、正则问法或字符串 fallback。表层差异只能通过通用归一化和训练样本进入统一结构。
