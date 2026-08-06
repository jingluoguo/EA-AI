# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

这是一个最小的“结构智能型 LLM”实验骨架。它不从堆大模型开始，而是先验证一个更小的问题：

> 模型能否从句子里抽取实体、关系、事件和规则，再基于这些结构推理出答案？

当前主路径是标准库即可运行的显式结构推理 baseline。

## 设计思想

这个项目要验证的是“结构智能”，不保留问法表、正则 parser 或字符串 fallback。后续演进以训练数据和可替换学习能力为中心，每次新增能力都遵循这条链：

```text
观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证
```

它对应到代码里的含义是：

- 观察现象：先看失败输入、当前结构输出、期望答案，判断问题发生在切句、归一化、陈述解析、Query 解析、状态更新、规则推导还是答案生成。
- 训练优先：先把失败样例写成数据，补齐 observation、context、world_state、belief_state、期望结构或答案，再跑评估；只有数据证明结构标签或模型槽位不足时，才改代码。
- 剥离次要因素：把“放进/放到/放入”、“盒子里/盒子里面”、主动句、被动句、语气词、寒暄话术等先归一化，不让它们污染实体槽位。
- 构建理想模型：把自然语言变成 `ENTITY`、`FRAME/ROLE`、`STATE`、`QUERY` 这些中间结构。
- 数学表达：推理只依赖 frame 角色匹配、当前状态查询、关系闭包、状态覆盖等可计算规则。
- 实验验证：新增能力必须配测试，证明不同表层表达会落到同一个结构模型；训练相关改动还要测试数据 loader、反馈写入和 schema 校验。

### 核心结构

`FRAME/ROLE/STATE/QUERY/INTENT` 是当前 symbolic baseline 的核心语义层：

- `ENTITY`：识别出的对象，例如人、物品、容器、地点。
- `FRAME`：按原文顺序保留的历史事件，例如 `put_in`、`move`、`give`、`paint`。
- `ROLE`：事件角色，例如 `actor`、`theme`、`goal`、`recipient`。
- `STATE`：由事件投影出的当前世界状态，例如 `in(芯片,盒子)`、`at(托盘,实验室)`。
- `QUERY`：把用户问题归一成可计算查询，例如 `actor_for_event(put_in,item=芯片,holder=盒子)`。
- `INTENT`：从观察到的行为推断出的心智假设，例如 `subject`、`goal`、`belief`、`strategy`、`evidence`、`confidence`。
- `RULE`：推理阶段命中的规则，例如 `event_actor_matches`、`container_moves_contents`。

`REL` 和 `EVENT` 仍会在 CLI 输出里显示，但它们只是兼容和阅读视图：`REL` 来自当前 `STATE`，`EVENT` 来自历史 `FRAME`。新增推理能力应该优先基于 `FRAME/ROLE/STATE/QUERY`，不要把 `REL/EVENT` 当主模型。

### 数据流

```text
原始文本
  -> perception/lexer.py 切句和查询候选保留
  -> perception/normalizer.py 表层剥离和槽位归一
  -> comprehension/statement.py 加载 statement_model.json，实例化 Entity + FRAME/ROLE
  -> world/state.py FRAME 投影为当前 STATE
  -> comprehension/query.py 加载 query_model.json，把查询候选抽象为 QUERY
  -> intent_analyzers 从完整结构和训练样本推断 INTENT
  -> reasoning/pipeline.py 门面转发到 reasoning/core.py 根据 QUERY + FRAME/STATE 推导 RULE
  -> motor/dialogue.py 补充已学习的回答能力
  -> kernel.py 编排并返回 Prediction
```

这个顺序很重要。陈述句按原文顺序处理，后发生的放入、移动、交接、涂色会覆盖同一对象的旧当前状态；历史事件仍保留在 `FRAME` 里，所以“现在在哪里”和“谁曾经把 X 放进 Y”可以同时成立。

### 完整实现流程

项目现在按“训练数据沉淀能力，运行时加载能力产物”的方式工作：

| 阶段 | 做什么 | 用到的方法 / 技术 | 主要代码 / 产物 |
| --- | --- | --- | --- |
| 1. 收集样本 | 保存用户输入、失败案例和人工反馈 | JSONL 追加式数据集；训练样本 schema；人工确认反馈 | `data/query_examples.jsonl`、`data/statement_examples.jsonl`、`data/intent_examples.jsonl`、`data/dialog_answer_examples.jsonl` |
| 2. 样本校验 | 确认样本字段完整、结构合法 | 结构化 schema 校验；`dataclass` 样本对象；槽位字段检查 | `comprehension/query.py`、`comprehension/statement.py`、`comprehension/intent_dataset.py` |
| 3. 编译模型产物 | 把训练样本沉淀成运行时能力文件 | 结构模板聚合；抽象问题模式；回答结构聚合；源数据 `sha256` 指纹；原子写入 JSON | `data/query_model.json`、`data/statement_model.json`、`data/dialog_answer_model.json` |
| 4. 运行时加载 | 启动时加载编译后的能力，而不是扫描训练集 | 模型 artifact 加载；能力函数注册；可替换 learner 接口 | `kernel.default_capabilities()`、`LearnedQueryParser`、`LearnedStatementParser`、`LearnedDialogActAnswerer` |
| 5. 文本切分 | 把输入拆成陈述片段和查询候选 | 标点切句；逗号/分号候选拆分；聊天片段保留；尾句保留 | `perception/lexer.py` |
| 6. 表层归一化 | 剥离不影响语义的表层差异 | 语气词清理；提问外壳清理；同义动作归一；容器后缀归一；`啥 -> 什么` | `perception/normalizer.py` |
| 7. 陈述理解 | 把陈述句变成实体和历史事件 | 句子模板槽位抽取；实体角色实例化；`FRAME/ROLE` 结构模板实例化 | `comprehension/statement.py`、`ENTITY`、`FRAME`、`ROLE` |
| 8. Query 理解 | 把问题变成可计算查询 | 抽象问题匹配；字符 bigram 相似度；角色槽位实例化；复合查询组合 | `comprehension/query.py`、`QUERY` |
| 9. 状态投影 | 从历史事件得到当前世界状态 | 事件 schema；状态 projector；状态 reducer；后发生事件覆盖旧状态 | `world/state.py`、`world/event_schema.py`、`STATE` |
| 10. 结构推理 | 根据结构推导规则和答案 | frame 角色匹配；状态查询；关系闭包；事件约束；反事实重放；答案生成器 | `reasoning/core.py`、`reasoning/pipeline.py`、`reasoning/rules/`、`reasoning/answers/`、`RULE`、answerer |
| 11. 不确定性决策 | 按置信度决定回答、确认或学习 | 置信度分段；`>=0.90` 直接回答；`0.50-0.90` 询问确认；`<0.50` 引导学习 | `metacognition/confidence.py`、`motor/feedback.py` |
| 12. 自学习反馈 | 未命中时确认相似含义或记录待整理样本 | 中置信相似结构召回；低置信写入待整理队列；回答只加载可信来源；重新编译；立即重试 | `motor/feedback.py`、`motor/learning_queue.py`、`motor/dialogue.py`、`struct-ask --learn-on-fail` |
| 13. 实验验证 | 验证训练集、模型产物和端到端行为 | 数据集评估；unittest 回归；结构线性化断言；端到端答案断言 | `make check`、`uv run python -m unittest discover -q -b` |

这里的“模型产物”目前不是神经网络权重，而是由训练样本编译出的结构能力文件。它保存抽象后的问题模式、句子模板、槽位角色、结构模板、特征单元、样本数量和数据指纹。后续如果替换成分类器、向量检索、生成模型或真正的神经模型，只需要替换 `comprehension/query.py`、`comprehension/statement.py` 里的学习能力实现，不需要把逻辑堆回 `kernel.py`。

### 用到的技术

当前实现刻意保持轻量，核心路径只依赖 Python 标准库：

- Python `dataclass`：定义 `Entity`、`Frame`、`Role`、`State`、`Query`、`Intention`、训练样本和编译模型结构。
- JSONL 数据集：用 `data/query_examples.jsonl`、`data/statement_examples.jsonl`、`data/intent_examples.jsonl`、`data/dialog_answer_examples.jsonl` 保存可增量追加的训练/反馈样本。
- 编译 JSON 模型：用 `data/query_model.json`、`data/statement_model.json`、`data/dialog_answer_model.json` 保存从训练集中沉淀出的运行时能力。
- 结构槽位：用 `$item#1`、`$container#1`、`$person#1` 这类槽位表达实体角色和出现顺序。
- 表层归一化：在 `perception/normalizer.py` 中统一语气词、同义动作、容器后缀、提问外壳和“啥/什么”等表层差异。
- 抽象特征匹配：Query 编译后使用抽象问题和字符 bigram 特征寻找相似结构；未命中时也用同一套相似度给用户推荐可能含义。
- 不确定性策略：`metacognition/confidence.py` 统一管理置信度阈值，避免在 CLI 或推理层写散落判断。
- 模板实例化：Statement 编译后使用句子模板抽取槽位，再实例化为 `ENTITY + FRAME/ROLE`。
- 状态投影：`world/state.py` 把历史 `FRAME` 转成当前 `STATE`，并处理后发生事件覆盖旧状态。
- 结构推理：`reasoning/core.py` 基于 frame 角色匹配、状态查询、关系闭包、事件约束和反事实重放生成 `RULE` 和答案，`reasoning/pipeline.py` 只保留稳定入口。
- 能力注册：`CognitiveCapabilities` 把陈述学习、Query 学习、状态投影、状态覆盖、规则推导和答案生成组合为可替换能力。
- 反馈学习服务：`motor/feedback.py` 和 `motor/dialogue.py` 封装相似建议、新聊天能力、可信回答样本加载和模型重编译；CLI 只负责交互展示。
- CLI 与 Makefile：`struct-ask`、`struct-compile-*`、`struct-eval-*` 提供命令入口；`make model`、`make check`、`make ask` 是日常使用入口。
- unittest 回归：`tests/test_reasoner.py` 覆盖数据 loader、反馈写入、模型编译、运行时加载、结构推理和端到端回答。

### 能力组合

当前认知内核用 `CognitiveCapabilities` 组合学习、状态、推理和答案能力：

```python
capabilities = default_capabilities()
prediction = predict(text, capabilities)
```

可插拔能力分成七类：

- `statement_parsers`：陈述学习能力，输入句子，输出 `Entity + Frame`。
- `state_projectors`：状态投影能力，输入 `Frame`，输出 `State`。
- `state_reducers`：状态覆盖能力，决定新状态如何更新当前世界。
- `query_parsers`：Query 学习能力，输入候选问题，输出 `Query`。
- `rule_inferers`：规则推导能力，输入完整 `Structure`，输出命中的规则名。
- `answerers`：答案生成能力，输入带规则的 `Structure`，输出自然语言答案。
- `intent_analyzers`：可学习意图分析能力，输入原文和完整 `Structure`，输出 `Intention` 假设。

新增能力时，先补对应 JSONL 数据集和评估样本，再替换相应学习能力；不通过新增问法函数扩展。

### 训练优先工作流

遇到新失败样例时，推荐顺序是：

1. 记录样例：保存原始 observation、上下文、当前结构输出、期望 `INTENT/QUERY/STATE` 或答案。
2. 写入数据集：用 JSONL 沉淀成训练/反馈样本，而不是先扩展问法匹配。
3. 评估当前学习能力，定位缺的是标签、结构槽位还是模型能力。
4. 训练或替换能力：优先更新数据集或对应 analyzer/learner/projector/inferer。
5. 回归验证：补 loader、schema、评估和端到端测试，确认能力来自数据闭环，而不是单句特殊分支。

### Query 训练与编译

Query 的流程很简单：

1. 把样本写进 `data/query_examples.jsonl`。
2. 运行编译命令，生成 `data/query_model.json`。
3. `struct-ask` 默认读这个模型文件，不直接扫训练集。

```json
{"question":"芯片在哪里","entities":[{"role":"item","name":"芯片"}],"query":{"intent":"location","target":"$item#1","qualifiers":[]},"source":"training","split":"train"}
```

样本里的 `$item#1`、`$container#1`、`$place#1` 是结构槽位，不是正则。编译时会把多条样本压成一个可加载的 `QUERY` 模型产物。

```bash
uv run struct-compile-query \
  --query-data data/query_examples.jsonl \
  --output data/query_model.json
```

需要检查效果时，跑这条：

```bash
uv run struct-eval-query \
  --query-data data/query_examples.jsonl \
  --query-model data/query_model.json
```

`struct-ask` 默认使用 `data/query_model.json`。调试时可以显式指定模型：

```bash
uv run struct-ask \
  --query-model data/query_model.json \
  "研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
```

一句话：先喂样本，再编译成模型，运行时只加载模型。

### 陈述训练与编译

陈述句也是同样的三步：

1. 把样本写进 `data/statement_examples.jsonl`。
2. 运行编译命令，生成 `data/statement_model.json`。
3. `struct-ask` 默认读这个模型文件，不直接扫训练集。

```json
{"sentence":"小张认为芯片在托盘里","sentence_template":"$person#1认为芯片在托盘里","entities":[{"role":"person","name":"$person#1"}],"frames":[{"frame_type":"believe","roles":{"person":"$person#1","proposition":"芯片在托盘里"}}],"source":"human_feedback","split":"train"}
```

主动句、被动句、换序和“里面/里边/里头”等表层差异先在 normalization 层归一，再压成同一类 `FRAME/ROLE` 模型。

```bash
uv run struct-compile-statement \
  --statement-data data/statement_examples.jsonl \
  --output data/statement_model.json
```

需要检查效果时，跑这条：

```bash
uv run struct-eval-statement \
  --statement-data data/statement_examples.jsonl \
  --statement-model data/statement_model.json
```

`struct-ask` 默认使用 `data/statement_model.json`。调试时也可以显式指定模型：

```bash
uv run struct-ask \
  --statement-model data/statement_model.json \
  "小王把芯片从托盘里面拿出来。芯片在哪里？"
```

一句话：先喂样本，再编译成模型，运行时只加载模型。

也可以直接用：

```bash
make model
make check
```

### 交互式自学习

日常测试直接用：

```bash
make ask TEXT="你擅长什么"
```

运行时会按同一套模型相似度做不确定性决策：

| 置信度 | 行为 |
| --- | --- |
| `>= 0.90` | 认为结构足够确定，直接回答。 |
| `0.50 - 0.90` | 不直接猜答案，先问用户“是不是这个意思”。确认后写回 JSONL、重新编译模型，再重试回答。 |
| `< 0.50` | 不再追问用户，写入 `data/unrecognized_examples.jsonl`，对用户只说“暂时无法识别。” |

所以如果一句话没有直接命中现有模型，它会先拿模型产物找相似含义，例如问你“它是不是在询问我能做什么”。你确认后，它会把原句按相同结构写入 JSONL，并重新编译模型；如果置信度低，或你否认了这个相似含义，系统只把原句记录到 `data/unrecognized_examples.jsonl`，留给后续离线整理。

如果待整理样本后来被你补成新的聊天能力，需要先把它迁移成 `data/query_examples.jsonl` 里的 Query 样本，再运行 `make model`。回答不会从运行时交互里直接生成；只有 `training`、`teacher`、`self_model`、`knowledge`、`curated`、`human_verified` 这些可信来源的回答样本会编译进 `data/dialog_answer_model.json`。没有可信回答时，系统会承认“已经理解问题，但还没有经过验证的相关回答”。

### 喂意图数据

意图分析不要继续靠“匹配用户怎么问”。推荐喂的是“行为观察 -> 心智假设”数据，让系统逐步学习 `Goal + Belief + Strategy`：

```json
{"observation":"妈妈在找眼镜","intention":{"subject":"妈妈","goal":"找到眼镜","belief":"妈妈不知道眼镜在哪里","strategy":"在可能的位置寻找眼镜","evidence":"妈妈在找眼镜","confidence":0.75,"source":"human_feedback"}}
```

完整 JSONL schema 可以带训练上下文和评估目标：

```json
{"observation":"孩子伸手去拿杯子","context":["杯子在桌上"],"world_state":["at(杯子,桌上)"],"belief_state":["believes(孩子,visible(杯子))"],"answer":"孩子想拿到杯子。","source":"human_feedback","split":"train","intention":{"subject":"孩子","goal":"拿到杯子","belief":"孩子认为杯子在眼前","strategy":"伸手抓取杯子","evidence":"孩子伸手去拿杯子","confidence":0.85,"source":"human_feedback"}}
```

也可以直接用 CLI 追加反馈样本：

```bash
uv run struct-add-intent-example "孩子伸手去拿杯子" \
  --subject "孩子" \
  --goal "拿到杯子" \
  --belief "孩子认为杯子在眼前" \
  --strategy "伸手抓取杯子" \
  --context "杯子在桌上" \
  --world-state "at(杯子,桌上)" \
  --answer "孩子想拿到杯子。"
```

追加后可以跑一个最小评估闭环：

```bash
uv run struct-eval-intent --train-data data/intent_examples.jsonl
```

保存为 JSONL 后，可以在代码里注入：

```python
from struct_llm.comprehension.intent import InMemoryIntentAnalyzer
from struct_llm.kernel import default_capabilities, predict

analyzer = InMemoryIntentAnalyzer.from_jsonl("data/intent_examples.jsonl")
capabilities = default_capabilities().with_intent_analyzers(analyzer)
prediction = predict("妈妈在找眼镜。你是谁？", capabilities)
print(prediction.structure.linearize())
```

也可以通过 CLI 直接喂：

```bash
uv run struct-ask --intent-data data/intent_examples.jsonl "妈妈在找眼镜。你是谁？"
```

当前 `InMemoryIntentAnalyzer` 是冷启动原型：默认没有样本就不会猜意图；有样本时会输出 `INTENT` 中间结构。后续可以把这个插槽替换成检索、训练好的分类/生成模型、在线反馈写回或多智能体强化学习模块，而不需要扩展文案匹配。

## 项目结构与文件配置

```text
.
  AGENTS.md            # 项目内 agent 约束：结构智能原则和模块边界
  README.md            # 项目说明、设计思想、使用方式
  Makefile             # 常用命令入口
  pyproject.toml       # Python 包元数据、依赖、CLI entry points
  uv.toml              # uv 配置
  uv.lock              # uv 锁文件
  data/
    query_examples.jsonl  # Query 解析训练样本
    query_model.json      # Query 编译模型，默认运行时加载
    statement_examples.jsonl  # 陈述解析训练样本
    statement_model.json      # 陈述编译模型，默认运行时加载
    intent_examples.jsonl # 可选：意图分析训练/反馈样本
src/struct_llm/
  capabilities.py       # 认知内核可插拔能力接口和组合方式
  kernel.py             # 认知循环编排：感知 -> 理解 -> 世界状态 -> 推理 -> 输出
  structure.py          # 结构表示：实体、关系、事件、规则、线性化格式
  perception/
    lexer.py            # 切句和查询候选保留
    normalizer.py       # 语气词、外层话术、槽位边界归一化
    reference.py        # 指代和指示词消解
  comprehension/
    statement.py        # 编译陈述样本并加载 statement_model.json
    query.py            # 编译 Query 样本并加载 query_model.json
    intent_dataset.py   # 意图训练/反馈 JSONL schema、校验、追加写入
    intent.py           # 观察样本 -> INTENT，可替换为训练模型
    structure_helpers.py # 纯结构构造和实体去重工具
  world/
    event_schema.py     # 事件 schema：角色别名、状态效果、事件查询匹配
    causal.py           # 条件规则展开和因果状态投影
    state.py            # FRAME -> 当前 STATE
  reasoning/
    core.py             # QUERY + FRAME/STATE -> 规则和答案
    pipeline.py         # 稳定门面：转发 core / rules / answers
    rules/              # 规则推导注册出口
    answers/            # 答案生成注册出口
  metacognition/
    confidence.py       # 置信度和不确定性策略
  memory/
    working.py          # 最小工作记忆：焦点实体、近期事件、当前状态
  motor/
    dialogue.py         # 已验证对话回答能力
    feedback.py         # 反馈学习服务
    learning_queue.py   # 待整理低置信样本队列
scripts/
  run_symbolic_demo.py  # 运行结构推理 demo
tests/
  test_reasoner.py      # 标准库测试
```

### 关键文件职责

`AGENTS.md` 是项目约束文件。它规定新增能力必须走“观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证”，并明确每种能力应该放在哪个模块。

`pyproject.toml` 定义包名、依赖和 CLI 命令。当前命令入口包括：

```text
struct-demo = struct_llm.cli:run_symbolic_demo
struct-ask = struct_llm.cli:ask_symbolic
struct-add-intent-example = struct_llm.cli:add_intent_example
struct-eval-intent = struct_llm.cli:eval_intent_examples
struct-eval-query = struct_llm.cli:eval_query_examples
struct-eval-statement = struct_llm.cli:eval_statement_examples
struct-compile-query = struct_llm.cli:compile_query_model
struct-compile-statement = struct_llm.cli:compile_statement_model
struct-compile-dialog-answer = struct_llm.cli:compile_dialog_answer_model
```

`Makefile` 是日常入口。`make ask` 会调用 `uv run struct-ask --learn-on-fail "$(TEXT)"`；`make test` 会调用标准库 unittest。

`structure.py` 定义所有中间结构。优先扩展这里的结构模型，而不是把语义塞进字符串。

`comprehension/intent_dataset.py` 负责意图训练/反馈样本的 JSONL schema、校验和追加写入。

`comprehension/intent.py` 负责可学习意图分析。它把行为观察映射到 `Intention(subject, goal, belief, strategy, evidence, confidence)`；默认实现只消费训练/反馈样本，不内置文案规则。

`world/event_schema.py` 定义事件的角色别名和状态效果。例如 `put_in` 投影为 `in(theme, goal)`，`move` 投影为 `at(theme, goal)`，`open/close` 投影为 `access(theme, result)`，`create/destroy` 投影为 `exists(theme, result)`。事件查询匹配和反事实事件排除也复用这里的角色别名。

`perception/normalizer.py` 负责剥离表层因素。例如当前会把 `放到/放入/放进` 归一为同一类容器放入动作，并把 `盒子里面/盒子里` 归一为 `盒子`。

`comprehension/statement.py` 负责默认陈述学习路径。它把 `data/statement_examples.jsonl` 编译为 `data/statement_model.json`，运行时从模型产物加载结构模板并通过实体槽位实例化 `FRAME/ROLE`；后续可以替换为分类器、生成模型或在线学习模型。

`world/state.py` 负责把历史事件投影成当前世界状态。新增事件如果会改变世界状态，应在这里新增 state projector 或 state reducer。

`comprehension/query.py` 负责默认 Query 学习路径。它把 `data/query_examples.jsonl` 编译为 `data/query_model.json`，运行时从模型产物加载抽象问题模式和 `QUERY` 结构模板。后续可以替换为分类器、生成模型或在线学习模型。

`reasoning/core.py` 负责规则推导和答案生成。`reasoning/pipeline.py` 现在只是稳定门面，`reasoning/rules/` 和 `reasoning/answers/` 暂时承接默认注册出口。

`kernel.py` 只做认知循环编排：切句、调用能力、组装结构、返回答案。业务规则不应该写回这个文件。

### 新增能力落点

遇到新失败样例时，先判断它属于哪个层级：

- 标点、尾句、寒暄问题没有被保留：改 `perception/lexer.py`。
- 语气词、同义动作、槽位边界污染：改 `perception/normalizer.py`。
- 陈述表达没有映射到已有 `FRAME/ROLE`：先追加 `data/statement_examples.jsonl`，再跑 `struct-compile-statement` 和 `struct-eval-statement --statement-model data/statement_model.json`。
- 新事件会改变当前世界，例如取出后不再在容器里：改 `world/state.py`。
- 用户换了问法但语义相同：先追加 `data/query_examples.jsonl`，再跑 `struct-compile-query` 和 `struct-eval-query --query-model data/query_model.json`。
- 已经有 `QUERY` 和 `FRAME/STATE`，但没有命中规则：改 `reasoning/core.py` 的 rule inferer。
- 已经命中规则，但答案表达不对：改 `reasoning/core.py` 的 answerer。

每次修改都要在 `tests/test_reasoner.py` 加端到端测试。对于表层差异，测试重点不是“某句能答”，而是多个不同表述能落到同一个 `FRAME/ROLE/STATE/QUERY`。

### 当前已支持的能力

当前 symbolic baseline 已经覆盖这些结构能力：

- 当前状态追踪：放入、移动、交接、涂色会更新当前 `STATE`。
- 属性状态追踪：支持打开/关闭这类对象状态，例如“盒子是什么状态？”。
- 存在性追踪：支持制造/创建/销毁/删除，销毁会清理对象的位置、持有、颜色和开关等当前状态。
- 状态失效：取出、否定陈述会删除当前包含关系，例如“托盘里没有芯片”。
- 状态修正：一句话可以先否定旧状态再写入新状态，例如“芯片不在托盘里而在盒子里”。
- 历史事件保留：当前状态改变后，仍可查询“谁曾经把芯片放进托盘”。
- 时间查询：支持“最开始在哪里”“某人操作之前在哪里”“某事件之后发生了什么”。
- 条件推理：支持“如果...那么...”“只要...就...”这类规则触发。
- 因果解释：支持“因为...所以...”和“为什么...”的结构化回答。
- 嵌套闭包：支持多层容器和移动传播，例如“芯片在实验室的大盒子里的小盒子里”。
- 集合查询：支持“某处有什么”“除了 X 还有什么”“经过了哪些地方”。
- 数量查询：支持“某处有几个东西”“数量是多少”，基于当前结构闭包回答已知下界，并过滤已销毁对象。
- 数量比较：支持“实验室和办公室哪个东西更多”“两处是否一样多”，比较当前已知内容闭包。
- 是非判断：支持“芯片存在吗”“芯片在托盘里吗”“实验室里有芯片吗”这类 polar query。
- 共位判断：支持“芯片和药瓶在同一个地方吗”，比较两个对象当前的位置键。
- 角色查询：支持“最后谁处理过 X”“小郭和小王分别做了什么”“每个人手里有什么”。
- 指代解析：支持“这个芯片”“这里”“这里有什么”这类基于上下文的追问入口。
- 来源跟踪：支持“谁说芯片在托盘里？”并将说法与事实分离。
- 信念世界：支持“小王认为芯片在哪里？”“谁认为芯片在盒子里？”这类个人视图查询，并且信念不会改写事实世界。
- 可学习意图假设：支持通过 `intent_examples.jsonl` 或 `.with_intent_analyzers(...)` 注入观察样本，输出 `INTENT` 中间结构。
- 矛盾检测：支持“有没有矛盾？”“哪里有冲突？”并比较说法/信念与当前事实状态。
- 反事实重放：支持“如果某人没有做某事件，X 会在哪里？”通过排除目标事件后重放 `FRAME -> STATE` 来回答。

## 下载后使用

如果还没有安装 `uv`，先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

进入项目目录后，同步环境：

```bash
uv sync
```

## 常用命令

最便捷方式：

```bash
make demo
make ask
make test
```

输入自己的问题：

```bash
make ask TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
```

进入连续输入模式：

```bash
make chat
```

## 当前最小任务

例子：

```text
小明把钥匙放进盒子。盒子被带到厨房。钥匙在哪里？
```

结构：

```text
ENTITY person=小明
ENTITY item=钥匙
ENTITY container=盒子
ENTITY place=厨房
REL in(钥匙,盒子)
REL at(盒子,厨房)
EVENT put_in(小明,钥匙) WITH holder=盒子
EVENT handle(小明,钥匙)
EVENT move(盒子,厨房)
FRAME f1 type=put_in time=1
ROLE f1 actor=小明
ROLE f1 theme=钥匙
ROLE f1 goal=盒子
FRAME f2 type=handle time=2
ROLE f2 actor=小明
ROLE f2 theme=钥匙
FRAME f3 type=move time=3
ROLE f3 theme=盒子
ROLE f3 goal=厨房
RULE container_moves_contents
QUERY location(钥匙)
```

答案：

```text
钥匙在厨房的盒子里。
```

这就是结构智能的最小形式：语言输入不是直接映射到答案，而是先映射到可组合、可检查、可迁移的中间结构。当前 symbolic baseline 会逐句抽取历史事件和当前状态，再根据问题抽取 `QUERY` 并推导规则；它不要求整段文本命中一个完整枚举模板。

它现在也支持更开放的“内容查询”，例如“实验室里至少有什么？”会沿着移动和包含关系做闭包推理。

### 抽象建模规则

实现遵循一条固定链条：观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证。

以“谁把芯片放进托盘？”、“芯片是谁放进托盘的？”、“芯片被谁放进托盘的？”为例，主动、换序、被动都是次要表层差异。它们会先归一到同一个事件角色模型：

```text
FRAME type=put_in
ROLE actor=?
ROLE theme=芯片
ROLE goal=托盘
```

再表达成可推理查询：

```text
QUERY actor_for_event(put_in,item=芯片,holder=托盘)
```

最后用已有事件和关系验证：

```text
EVENT put_in(小郭,芯片) WITH holder=托盘
REL in(芯片,托盘)
```
