# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

这是一个最小的“结构智能型 LLM”实验骨架。它不从堆大模型开始，而是先验证一个更小的问题：

> 模型能否从句子里抽取实体、关系、事件和规则，再基于这些结构推理出答案？

第一版包含两个层次：

- `symbolic`：标准库即可运行的显式结构推理 baseline。
- `neural`：预留 PyTorch tiny Transformer 训练入口，用来学习生成结构 token 和答案 token。

## 设计思想

这个项目要验证的是“结构智能”，不是用大段正则去穷举自然语言问法。每次新增能力时，都遵循这条链：

```text
观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证
```

它对应到代码里的含义是：

- 观察现象：先看失败输入、当前结构输出、期望答案，判断问题发生在切句、归一化、陈述解析、Query 解析、状态更新、规则推导还是答案生成。
- 剥离次要因素：把“放进/放到/放入”、“盒子里/盒子里面”、主动句、被动句、语气词、寒暄话术等先归一化，不让它们污染实体槽位。
- 构建理想模型：把自然语言变成 `ENTITY`、`FRAME/ROLE`、`STATE`、`QUERY` 这些中间结构。
- 数学表达：推理只依赖 frame 角色匹配、当前状态查询、关系闭包、状态覆盖等可计算规则。
- 实验验证：新增能力必须配测试，证明不同表层表达会落到同一个结构模型。

### 核心结构

`FRAME/ROLE/STATE/QUERY` 是当前 symbolic baseline 的核心语义层：

- `ENTITY`：识别出的对象，例如人、物品、容器、地点。
- `FRAME`：按原文顺序保留的历史事件，例如 `put_in`、`move`、`give`、`paint`。
- `ROLE`：事件角色，例如 `actor`、`theme`、`goal`、`recipient`。
- `STATE`：由事件投影出的当前世界状态，例如 `in(芯片,盒子)`、`at(托盘,实验室)`。
- `QUERY`：把用户问题归一成可计算查询，例如 `actor_for_event(put_in,item=芯片,holder=盒子)`。
- `RULE`：推理阶段命中的规则，例如 `event_actor_matches`、`container_moves_contents`。

`REL` 和 `EVENT` 仍会在 CLI 输出里显示，但它们只是兼容和阅读视图：`REL` 来自当前 `STATE`，`EVENT` 来自历史 `FRAME`。新增推理能力应该优先基于 `FRAME/ROLE/STATE/QUERY`，不要把 `REL/EVENT` 当主模型。

### 数据流

```text
原始文本
  -> text_processing.split_sentences
  -> normalization 表层剥离和槽位归一
  -> frame_parser 陈述句抽取 Entity + FRAME/ROLE
  -> state_engine FRAME 投影为当前 STATE
  -> query_parser 查询候选抽象为 QUERY
  -> inference 根据 QUERY + FRAME/STATE 推导 RULE 和答案
  -> reasoner 编排并返回 Prediction
```

这个顺序很重要。陈述句按原文顺序处理，后发生的放入、移动、交接、涂色会覆盖同一对象的旧当前状态；历史事件仍保留在 `FRAME` 里，所以“现在在哪里”和“谁曾经把 X 放进 Y”可以同时成立。

### 能力组合

为了避免 `reasoner.py` 重新膨胀成大 if/regex 文件，当前 pipeline 用 `StructuralCapabilities` 组合能力：

```python
capabilities = default_capabilities().with_query_parsers(parse_keeper_query)
prediction = predict(text, capabilities)
```

可插拔能力分成六类：

- `statement_parsers`：陈述解析能力，输入句子，输出 `Entity + Frame`。
- `state_projectors`：状态投影能力，输入 `Frame`，输出 `State`。
- `state_reducers`：状态覆盖能力，决定新状态如何更新当前世界。
- `query_parsers`：问题抽象能力，输入归一后的候选问题，输出 `Query`。
- `rule_inferers`：规则推导能力，输入完整 `Structure`，输出命中的规则名。
- `answerers`：答案生成能力，输入带规则的 `Structure`，输出自然语言答案。

新增能力时，先判断它属于哪一层，再在对应模块新增小函数并注册到默认能力列表。只有当外部调用想临时扩展某种表达时，才通过 `.with_query_parsers(...)` 这类方法注入。

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
    train.jsonl        # symbolic/neural 训练数据
    test.jsonl         # 测试数据
    tiny_model.pt      # tiny Transformer 训练产物，如果已训练
src/struct_llm/
  world.py              # 微型世界：人物、物品、容器、地点、任务模板
  structure.py          # 结构表示：实体、关系、事件、规则、线性化格式
  event_schema.py       # 事件 schema：角色别名、状态效果、事件查询匹配
  dataset.py            # 数据生成：训练/测试组合泛化切分
  text_processing.py    # 切句和查询候选保留
  normalization.py      # 语气词、外层话术、槽位边界归一化
  capabilities.py       # 可插拔能力接口：parser/projector/inferer/answerer
  frame_parser.py       # 陈述句 -> Entity + FRAME/ROLE
  state_engine.py       # FRAME -> 当前 STATE
  query_parser.py       # 查询候选 -> QUERY
  inference.py          # QUERY + FRAME/STATE -> 规则和答案
  reasoner.py           # 轻量编排层
  vocab.py              # 神经模型用的字符级词表
  model.py              # PyTorch tiny Transformer，可选依赖
scripts/
  make_dataset.py       # 生成 JSONL 数据
  run_symbolic_demo.py  # 运行结构推理 demo
  train_tiny_model.py   # 训练 tiny Transformer，需要安装 torch
tests/
  test_reasoner.py      # 标准库测试
```

### 关键文件职责

`AGENTS.md` 是项目约束文件。它规定新增能力必须走“观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证”，并明确每种能力应该放在哪个模块。

`pyproject.toml` 定义包名、依赖、可选神经网络依赖和 CLI 命令。当前命令入口包括：

```text
struct-demo = struct_llm.cli:run_symbolic_demo
struct-ask = struct_llm.cli:ask_symbolic
struct-ask-neural = struct_llm.cli:ask_neural
struct-make-dataset = struct_llm.cli:make_dataset
struct-train-tiny = struct_llm.cli:train_tiny_model
```

`Makefile` 是日常入口。`make ask` 会调用 `uv run struct-ask "$(TEXT)"`；`make test` 会调用标准库 unittest。

`structure.py` 定义所有中间结构。优先扩展这里的结构模型，而不是把语义塞进字符串。

`event_schema.py` 定义事件的角色别名和状态效果。例如 `put_in` 投影为 `in(theme, goal)`，`move` 投影为 `at(theme, goal)`，`open/close` 投影为 `access(theme, result)`，`create/destroy` 投影为 `exists(theme, result)`。事件查询匹配和反事实事件排除也复用这里的角色别名。

`normalization.py` 负责剥离表层因素。例如当前会把 `放到/放入/放进` 归一为同一类容器放入动作，并把 `盒子里面/盒子里` 归一为 `盒子`。

`frame_parser.py` 负责陈述句到 `FRAME/ROLE` 的抽取。新增事件类型，例如“取出”“打开”“关闭”，应在这里新增 statement parser。

`state_engine.py` 负责把历史事件投影成当前世界状态。新增事件如果会改变世界状态，应在这里新增 state projector 或 state reducer。

`query_parser.py` 负责把问题抽象成 `QUERY`。新增问法时，不要直接回答；先归一成类似 `location(芯片)` 或 `actor_for_event(...)` 的结构。

`inference.py` 负责规则推导和答案生成。新增规则时，通常要成对新增 `rule_inferer` 和 `answerer`。

`reasoner.py` 只做编排：切句、调用能力、组装结构、返回答案。业务规则不应该写回这个文件。

### 新增能力落点

遇到新失败样例时，先判断它属于哪个层级：

- 标点、尾句、寒暄问题没有被保留：改 `text_processing.py`。
- 语气词、同义动作、槽位边界污染：改 `normalization.py`。
- 新事件没有抽出来，例如“取出”“打开”“关闭”：改 `frame_parser.py`。
- 新事件会改变当前世界，例如取出后不再在容器里：改 `state_engine.py`。
- 用户换了问法但语义相同：改 `query_parser.py`。
- 已经有 `QUERY` 和 `FRAME/STATE`，但没有命中规则：改 `inference.py` 的 rule inferer。
- 已经命中规则，但答案表达不对：改 `inference.py` 的 answerer。

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

如果后面要训练 tiny Transformer，同步神经网络依赖：

```bash
uv sync --extra neural
```

## 常用命令

最便捷方式：

```bash
make demo
make ask
make data
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

训练 tiny Transformer：

```bash
make train
make ask-neural TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
make chat-neural
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
