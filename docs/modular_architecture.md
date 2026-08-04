# 模块化架构草案

## 核心原则

模块先有边界，再有能力；先能插拔，再谈增强。

这意味着新系统模块不需要一开始就有完整算法。规划、具身、情感、自我认知、持续学习都可以先提供空实现，只要它们有稳定接口、可替换默认实现和独立测试入口。后续增强单个模块时，不应该牵动整条结构推理管线。

## 分层目标

当前 `struct_llm` 已经有一条内部认知管线：

```text
text_processing
  -> normalization
  -> frame_parser
  -> state_engine
  -> query_parser
  -> inference
  -> reasoner
```

这条管线应保留为“认知内核”的第一版实现。外层再增加一个模块注册层，用来承载更完整的 agent 系统。

建议分为两层：

- 认知内核：理解、分析、推理、组织、记忆、知识、生成、对齐。
- 代理扩展：规划、具身、情感、自我认知、持续学习。

其中，当前代码已经主要覆盖“理解、分析、推理、组织”的一部分。记忆、知识、生成、对齐以及代理扩展模块可以先作为空槽位注册。

## 当前代码结构

当前已经新增模块协议和默认空实现，并保持现有推理结果不变：

```text
src/struct_llm/modules/
  __init__.py
  base.py              # ModuleContext、ModuleResult、StructuralModule 协议
  registry.py          # ModuleRegistry，组合各类系统模块
  cognitive.py         # 外层认知模块，挂载 struct_llm/cognitive 内核
  memory.py            # NoOpMemoryModule
  knowledge.py         # NoOpKnowledgeModule
  generation.py        # NoOpGenerationModule
  alignment.py         # NoOpAlignmentModule
  planning.py          # NoOpPlanningModule
  embodiment.py        # NoOpEmbodimentModule
  emotion.py           # NoOpEmotionModule
  self_model.py        # NoOpSelfModelModule
  learning.py          # NoOpLearningModule
```

同时，现有结构推理能力已经归入 `src/struct_llm/cognitive/`：

```text
src/struct_llm/cognitive/
  __init__.py
  capabilities.py      # CognitiveCapabilities，注册陈述、状态、查询、规则、答案能力
  kernel.py            # 唯一认知内核流水线，调用现有解析、状态、查询、推理模块
  text_processing.py   # 切句和查询候选保留
  normalization.py     # 表层剥离和槽位归一化
  frame_parser.py      # 陈述句 -> Entity + FRAME/ROLE
  state_engine.py      # FRAME -> 当前 STATE
  query_parser.py      # 查询候选 -> QUERY
  inference.py         # QUERY + FRAME/STATE -> 规则和答案
```

`cognitive/capabilities.py` 负责内部结构能力的可插拔组合；`modules/registry.py` 负责更高层系统模块的组合。两者不要混成一个概念：

- `CognitiveCapabilities`：一句话怎么解析成 `FRAME/STATE/QUERY`，以及怎么推理出答案。
- `ModuleRegistry`：一次 agent 运行有哪些外围系统参与，例如记忆、规划、对齐、学习。

现有的 `text_processing.py`、`normalization.py`、`frame_parser.py`、`state_engine.py`、`query_parser.py`、`inference.py` 已经移动到 `struct_llm/cognitive/` 并通过 `cognitive/kernel.py` 融入认知内核。`modules/cognitive.py` 只是把认知内核挂载到外层 `ModuleRegistry`，不能另起一套 query/parser 规则。

## 接口方向

第一版接口保持轻量，先满足可替换和可测试：

```python
@dataclass(frozen=True)
class ModuleContext:
    text: str
    structure: Structure | None = None
    answer: str | None = None


@dataclass(frozen=True)
class ModuleResult:
    context: ModuleContext
    notes: tuple[str, ...] = ()


class StructuralModule(Protocol):
    name: str

    def run(self, context: ModuleContext) -> ModuleResult:
        ...
```

空模块只返回原始 `context`。真实模块后续可以在不改变调用方的情况下追加 `notes`、补充结构、拦截请求或更新外部状态。

## 落代码路线

### 阶段一：建立外层模块壳

- 已新增 `src/struct_llm/modules/`。
- 已定义 `ModuleContext`、`ModuleResult`、`StructuralModule`。
- 已定义 `ModuleRegistry`，默认注册所有 `NoOp` 模块。
- 已增加测试，证明默认空模块不改变当前认知内核行为。

### 阶段二：把现有认知管线包成模块

- 已新增 `cognitive.py`，把当前 `parse_text()`、`predict()` 的核心逻辑包装成 `CognitiveKernelModule`。
- 已让 `reasoner.py` 回到编排层，默认通过模块注册表运行。
- 已保留当前 `predict(text, capabilities=None)` API，避免破坏已有调用。

### 阶段三：逐个激活空槽位

- `memory.py`：先支持短程上下文摘要或最近实体追踪，再考虑长期检索。
- `knowledge.py`：先定义外部知识查询接口，不急着接具体知识库。
- `alignment.py`：先做指令边界和拒答策略的结构化结果，不直接散落在回答函数里。
- `planning.py`：先把复杂任务拆成步骤结构，不接真实工具执行。
- `embodiment.py`：先定义感知输入和行动输出的数据结构。
- `emotion.py`：先输出情绪/语气标签，不直接改变事实推理。
- `self_model.py`：先记录系统能力边界、置信度、自检结果。
- `learning.py`：先记录可学习样例和失败原因，不自动改写核心规则。

### 阶段四：建立模块级评测

- 每个模块至少有一个“不启用时行为不变”的回归测试。
- 每个真实模块至少有一组“同义表达映射到同一结构”的测试。
- 外层模块测试不要越过模块边界直接依赖 `reasoner.py` 内部细节。

## 约束

- 新能力优先注册到对应模块或 `CognitiveCapabilities` 对应层，不写端到端特殊分支。
- `reasoner.py` 只做编排，不承载业务规则。
- 空实现是正式架构的一部分，不是临时代码。
- 模块增强必须能单独测试、单独替换、单独回滚。
