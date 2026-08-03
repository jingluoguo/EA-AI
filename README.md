# Structural LLM Minimal Lab

这是一个最小的“结构智能型 LLM”实验骨架。它不从堆大模型开始，而是先验证一个更小的问题：

> 模型能否从句子里抽取实体、关系、事件和规则，再基于这些结构推理出答案？

第一版包含两个层次：

- `symbolic`：标准库即可运行的显式结构推理 baseline。
- `neural`：预留 PyTorch tiny Transformer 训练入口，用来学习生成结构 token 和答案 token。

## 项目结构

```text
src/struct_llm/
  world.py              # 微型世界：人物、物品、容器、地点、任务模板
  structure.py          # 结构表示：实体、关系、事件、规则、线性化格式
  dataset.py            # 数据生成：训练/测试组合泛化切分
  reasoner.py           # 显式结构解析 + 规则推理 baseline
  vocab.py              # 神经模型用的字符级词表
  model.py              # PyTorch tiny Transformer，可选依赖
scripts/
  make_dataset.py       # 生成 JSONL 数据
  run_symbolic_demo.py  # 运行结构推理 demo
  train_tiny_model.py   # 训练 tiny Transformer，需要安装 torch
tests/
  test_reasoner.py      # 标准库测试
```

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
EVENT put_in(小明,钥匙)
EVENT handle(小明,钥匙)
EVENT move(盒子,厨房)
RULE container_moves_contents
QUERY location(钥匙)
```

答案：

```text
钥匙在厨房的盒子里。
```

这就是结构智能的最小形式：语言输入不是直接映射到答案，而是先映射到可组合、可检查、可迁移的中间结构。
当前 symbolic baseline 会先逐句抽取事件和关系，再根据问题抽取 `QUERY` 并推导规则；它不再要求整段文本命中一个完整枚举模板。

它现在也支持更开放的“内容查询”，例如“实验室里至少有什么？”会沿着移动和包含关系做闭包推理。

### 抽象建模规则

实现遵循一条固定链条：观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证。

以“谁把芯片放进托盘？”、“芯片是谁放进托盘的？”、“芯片被谁放进托盘的？”为例，主动、换序、被动都是次要表层差异。它们会先归一到同一个事件角色模型：

```text
put_in(actor=?, item=芯片, holder=托盘)
```

再表达成可推理查询：

```text
QUERY actor_for_event(put_in,item=芯片,holder=托盘)
```

最后用已有事件和关系验证：

```text
EVENT put_in(小郭,芯片)
REL in(芯片,托盘)
```
