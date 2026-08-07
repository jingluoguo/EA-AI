# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

这是一个最小的“结构智能型 LLM”实验骨架。它不从堆大模型开始，而是先验证一个更小的问题：

> 模型能否从句子里抽取实体、事件、状态、规则和问题，再基于这些结构推理出答案？

当前主路径是“神经输入理解 + 显式结构推理”：Query 和 Statement 由 PyTorch 字符级双向 GRU 模型解析，世界状态、规则和答案仍走可检查的结构推理。

## 设计原则

项目以训练数据、结构中间层和可替换学习能力为主路径。新增能力遵循：

```text
观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证
```

也就是说：先把失败样例沉淀为 JSONL 训练/反馈数据，再评估缺的是结构标签、槽位、状态投影、规则还是答案。只有数据证明结构表达不足时，才改 parser、state、query 或 inference。

核心中间结构是：

- `ENTITY`：识别出的对象，例如人、物品、容器、地点。
- `FRAME/ROLE`：按原文顺序保留的历史事件和事件角色。
- `STATE`：由事件投影出的当前世界状态，后发生状态覆盖旧状态。
- `QUERY`：把问题归一为可计算查询。
- `INTENT`：从行为观察推断出的心智假设。
- `RULE`：推理阶段命中的规则。

`REL` 和 `EVENT` 仍会在 CLI 输出中显示，但只是兼容阅读视图；新增推理能力应优先基于 `FRAME/ROLE/STATE/QUERY`。

## 数据流

```text
原始文本
  -> perception/lexer.py 切句和查询候选保留
  -> perception/normalizer.py 表层剥离和槽位归一
  -> my_neural.py 加载 Statement 神经模型，解析 Entity + FRAME/ROLE
  -> world/state.py 把 FRAME 投影为当前 STATE
  -> my_neural.py 加载 Query 神经模型，解析 QUERY
  -> memory/long_term.py 注入已确认长期 STATE
  -> intent_analyzers 推断 INTENT
  -> reasoning/rules/* 根据 QUERY + FRAME/STATE 推导 RULE
  -> reasoning/answers/* 生成答案
  -> kernel.py 返回 Prediction
```

## 快速开始

```bash
uv sync
make train
make check
make ask TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
```

常用命令：

```bash
make ask TEXT="芯片在哪里？"
make remember TEXT="我叫小王"
make knowledge QUESTION="为什么天是蓝的？" ANSWER="因为短波长蓝光更容易被大气散射。"
make train
make test
make check
```

统一 CLI 入口是：

```bash
uv run struct <command> [args...]
```

当前子命令包括 `ask`、`train`、`eval-query`、`eval-statement`、`eval-intent`、`add-memory`、`add-knowledge`、`compile-*`。

## 项目结构

```text
data/                         # JSONL 训练/反馈数据和神经/记忆产物
docs/
  development_workflow.md      # 训练优先开发流程和样本格式
  symbolic_baseline_capabilities.md
src/struct_llm/
  capabilities.py              # 可插拔认知能力接口和组合
  kernel.py                    # 公开预测入口和默认能力装配
  kernel_flow.py               # 感知 -> 理解 -> 状态 -> 推理 -> 输出编排
  dataset_io.py                # JSONL 行级读写和文件指纹公共工具
  structure.py                 # Entity/Frame/Role/State/Query/Intention
  cli.py                       # struct 子命令分发入口
  cli_commands/                # ask、learning、memory 等 CLI 命令实现
  perception/                  # 切句、归一化、指代消解
  comprehension/               # Query/Statement/Intent 样本 schema 和评估
  neural/                      # Query/Statement 神经模型和训练公共工具
  world/                       # 事件 schema、因果展开、状态投影
  reasoning/
    selection/                 # 共享结构选择器
    rules/                     # 分域规则推导
    answers/                   # 分域答案生成
  memory/                      # 长期记忆和知识模型
  motor/                       # 反馈、自学习、对话回答
tests/
  test_learning.py
  test_state.py
  test_queries.py
  test_dialogue.py
  support.py
```

## 开发入口

新增能力时，先读 [AGENTS.md](AGENTS.md) 和 [docs/development_workflow.md](docs/development_workflow.md)。能力分类清单见 [docs/symbolic_baseline_capabilities.md](docs/symbolic_baseline_capabilities.md)。

测试分布：

- `tests/test_learning.py`：数据 loader、反馈写入、神经训练和运行时加载。
- `tests/test_state.py`：事件投影、状态覆盖、否定和修正。
- `tests/test_queries.py`：位置、内容、数量、历史事件和指代查询。
- `tests/test_dialogue.py`：对话、记忆、因果、信念、矛盾和反事实。

每次修改都要加对应端到端测试。对于表层差异，测试重点不是“某句能答”，而是多个不同表达能落到同一个 `FRAME/ROLE/STATE/QUERY`。
