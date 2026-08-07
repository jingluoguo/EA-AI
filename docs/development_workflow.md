# 训练优先开发流程

这份文档承接 README 里移出的细节。项目约束仍以 [AGENTS.md](../AGENTS.md) 为准：新增能力必须以训练数据、结构中间层和可替换学习能力为中心，不绕过结构学习主路径。

## 判断失败层级

遇到新失败样例时，先记录：

- 原始输入和上下文。
- 当前 `Structure.linearize()` 输出。
- 期望的 `ENTITY/TYPE/ATTRIBUTE/FRAME/ROLE/STATE/TIME/SOURCE/BELIEF/CONDITION/CAUSE/QUERY/INTENT` 或答案。
- 失败位置：切句、归一化、陈述解析、Query 解析、指代消解、状态更新、信念视图、因果展开、规则推导或答案生成。

推荐落点：

- 标点、尾句、寒暄问题没有被保留：改 `perception/lexer.py`。
- 语气词、同义动作、槽位边界污染：改 `perception/normalizer.py`。
- 陈述表达没有映射到已有 `FRAME/ROLE`：先追加 `data/statement_examples.jsonl`，再跑 `make train`。
- 用户换了问法但语义相同：先追加 `data/query_examples.jsonl`，再跑 `make train`。
- 新事件改变当前世界：改 `world/state.py` 或 `world/event_schema.py`。
- 已有 `QUERY` 和 `FRAME/STATE` 但无规则：改 `reasoning/rules/`。
- 已命中规则但答案不对：改 `reasoning/answers/`。

## 目标语义层

复杂句子可以同时携带多种信息。新增能力时，先判断应该落到哪一层：

- 对象和概念：`ENTITY`、类型、别名、集合成员。
- 属性和状态：颜色、存在性、开闭、偏好、身份、数量、单位。
- 事件和角色：`FRAME/ROLE`、参与者、受事、来源、目标、结果。
- 时间和顺序：当前、历史、之前、之后、最初、最新、反事实重放切片。
- 极性和模态：否定、修正、可能、必须、实际、假设。
- 来源和信念：谁说、谁认为、事实世界、个人信念世界、证据来源。
- 条件和因果：如果/那么、因为/所以、条件触发、因果解释。
- 集合和比较：包含闭包、排除项、数量统计、多少比较。
- 指代和焦点：它、这里、这个、前者/后者、对话上下文承接。
- 查询和意图：单一查询、复合查询、对话行为、用户目标、待学习反馈。
- 元认知和输出：置信度、确认策略、可信回答来源、答案排序。

## 完整实现流程

| 阶段 | 做什么 | 主要代码 / 产物 |
| --- | --- | --- |
| 收集样本 | 保存失败案例、人工反馈、已确认记忆 | `data/*.jsonl` |
| 样本校验 | 检查字段、结构、槽位 | `comprehension/query.py`、`comprehension/statement.py`、`comprehension/intent_dataset.py` |
| 训练神经能力 | 训练 Query/Statement 输入模型 | `data/query_neural_model.*`、`data/statement_neural_model.*` |
| 运行时加载 | 加载权重、可信回答、长期记忆 | `my_neural.py`、`kernel.py`、`capabilities.py` |
| 状态投影 | 从 `FRAME` 得到当前 `STATE` | `world/state.py`、`world/event_schema.py` |
| 结构推理 | 根据结构推导规则和答案 | `reasoning/selection/`、`reasoning/rules/`、`reasoning/answers/` |
| 不确定性决策 | 直接回答、询问确认或入队 | `metacognition/confidence.py`、`motor/feedback.py` |
| 回归验证 | 验证 schema、loader、评估和端到端答案 | `make check`、`tests/test_*.py` |

## Query 神经训练

Query 训练样本放在 `data/query_examples.jsonl`：

```json
{"question":"芯片在哪里","entities":[{"role":"item","name":"芯片"}],"query":{"intent":"location","target":"$item#1","qualifiers":[]},"source":"training","split":"train"}
```

`$item#1`、`$container#1`、`$place#1` 是结构槽位，不是正则。训练后模型会把输入映射到结构标签，再在运行时还原成 `QUERY`。

常用命令：

```bash
make train
uv run struct eval-query --query-data data/query_examples.jsonl
```

## Statement 神经训练

Statement 训练样本放在 `data/statement_examples.jsonl`：

```json
{"sentence":"小张认为芯片在托盘里","sentence_template":"$person#1认为芯片在托盘里","entities":[{"role":"person","name":"$person#1"}],"frames":[{"frame_type":"believe","roles":{"person":"$person#1","proposition":"芯片在托盘里"}}],"source":"human_feedback","split":"train"}
```

主动句、被动句、换序和“里面/里边/里头”等表层差异应先归一到同一类 `FRAME/ROLE` 模型。

常用命令：

```bash
make train
uv run struct eval-statement --statement-data data/statement_examples.jsonl
```

## 交互式自学习

`struct ask --learn-on-fail` 使用统一的不确定性策略：

| 置信度 | 行为 |
| --- | --- |
| `>= 0.90` | 结构足够确定，直接回答。 |
| `0.50 - 0.90` | 先询问用户是否为相似结构；确认后写回 JSONL 并重训。 |
| `< 0.50` | 写入 `data/unrecognized_examples.jsonl`，等待离线整理。 |

回答不会从运行时交互里直接生成；只有可信来源的回答样本会进入 `data/dialog_answer_model.json`。

## 记忆和知识

长期记忆分两类：

- `data/memory_direct_examples.jsonl`：显式写入的状态条目。
- `data/memory_chat_examples.jsonl`：聊天中人工确认后的沉淀条目。

编译和写入：

```bash
uv run struct compile-memory
uv run struct add-memory --state name 我 小王
uv run struct add-memory "我叫小王"
```

长期知识单独放在：

- `data/memory_knowledge_examples.jsonl`
- `data/memory_knowledge_model.json`

```bash
make knowledge QUESTION="为什么天是蓝的？" ANSWER="因为短波长蓝光更容易被大气散射。"
make knowledge FILE=path/to/qa.jsonl
```

## 意图数据

意图分析推荐喂“行为观察 -> 心智假设”：

```json
{"observation":"孩子伸手去拿杯子","context":["杯子在桌上"],"world_state":["at(杯子,桌上)"],"belief_state":["believes(孩子,visible(杯子))"],"answer":"孩子想拿到杯子。","source":"human_feedback","split":"train","intention":{"subject":"孩子","goal":"拿到杯子","belief":"孩子认为杯子在眼前","strategy":"伸手抓取杯子","evidence":"孩子伸手去拿杯子","confidence":0.85,"source":"human_feedback"}}
```

```bash
uv run struct add-intent "孩子伸手去拿杯子" \
  --subject "孩子" \
  --goal "拿到杯子" \
  --belief "孩子认为杯子在眼前" \
  --strategy "伸手抓取杯子"

uv run struct eval-intent --train-data data/intent_examples.jsonl
```

## 测试要求

每次新增能力都要补测试，证明主动、被动、换序或同义表达能映射到同一个中间结构。训练相关改动还必须覆盖数据 loader、反馈写入、schema 校验或评估路径。
