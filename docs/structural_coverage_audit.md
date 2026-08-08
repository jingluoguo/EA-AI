# 结构覆盖审计

本轮目标不是扩大样本规模，而是把训练集和测试集按结构现象重新梳理。审计遵循 `观察现象 -> 剥离次要因素 -> 构建理想模型 -> 数学表达 -> 实验验证`。

## 观察现象

- Query 数据覆盖了位置、历史事件、反事实、信念、来源、复合查询、存在性和基础聊天行为，但部分结构族存在大量近义重复，尤其是 `why`、`counterfactual_location`、`belief_location`。
- 集合 Query 虽然已有 `count/compare_count/contents_except`，但“是否一样多”和“除...之外”构式缺少独立 train/test 结构模式，轻微实体或句法变化可能从正确结构退化为拒绝解析。
- Statement 数据覆盖主动、被动、换序、状态覆盖、条件、因果、信念和来源，但“同一话轮同时包含来源陈述和信念陈述”的组合样本偏少。
- 因果状态投影中曾存在同句标签冲突：`因为 X 把容器带到地点，所以容器在地点` 旧样本只标 `because + move`，而结构目标应包含结果状态 `be_in`。这会让学习路径无法稳定泛化到 `because + move + STATE`。
- 条件句 consequent 与知识型 why query 存在边界风险：`小王就把托盘带到实验室` 应作为可展开的 `move` 陈述片段，而 `苹果会掉到地上的原因是什么` 应被陈述解析拒绝并交给 Query/knowledge 路径。
- 条件句还存在槽位泛化风险：`如果小郭把芯片放进托盘...` 曾被近邻样本解析成 `小王把芯片放进托盘`，说明 `if_then` 的 antecedent/consequent 需要带可替换实体槽位，而不能只保留整句常量。
- 条件 consequent 的状态句也有边界风险：`芯片就在托盘里` 中的 `就` 是构式标记，不应进入 `$item` 槽位，否则会产生 `芯片就` 这样的伪实体并污染后续状态展开。
- 时间并列陈述存在篇章切分风险：`先/起初...，后来/随后...` 若在逗号处拆开，后半句可能失去共享主体，被 Query 模型误吸，导致历史 frame 缺失或产生伪查询。
- 工作记忆已经保存 `focus_query_intent`，但原先没有进入 Query 神经输入；同一句省略表达如 `盒子呢` 会因缺少上一轮查询类型而在 `location`、`contents`、`object_state` 之间漂移。
- Episode 数据只有少量语用监督，已覆盖不完整表达、歧义指代、确认共识、角色纠错、动作结果报告和上一轮回忆；本轮继续把角色纠错和动作结果报告从 train 单例提升为 train/test 结构模式。
- Intent 数据已有大量带 `context`、`belief_state`、`strategy` 的人类反馈样本，但缺少专门面向简单聊天和复杂结构查询的结构模式标签，尤其是寒暄接任务、澄清、确认、继续、回忆上一轮、事实/信念区分、来源区分、篇章指代和反事实查询。
- Intent 结构样本需要 train/test 对照，否则只能证明结构族存在，不能证明轻微表层变化能归到同一意图结构。
- 测试集中已有端到端答案回归，但缺少一个稳定的结构层覆盖审计，来约束每次补样本必须贡献新的结构模式。

## 剥离次要因素

本轮不把“咋、为啥、请问、你知道的话、现在、到底”等表层词当成独立能力。它们只作为候选保留、槽位清理或训练样本里的可归一化因素。新增样本优先表达结构差异：

- 事实世界位置 vs 某人的信念位置。
- 谁说过某命题 vs 谁相信某命题。
- 当前状态覆盖 vs 历史 frame 保留。
- 同一主体的信念修订 vs 陈述来源：`先以为...后来认为...` 应保留两个 `believe` frame，而不是把前半句误归成 `say`。
- 单纯 `not_in` 否定旧位置 vs `not_in + be_in` 同句修正新位置；后者必须写入新的当前 `STATE in`，不能只清除旧位置。
- 普通存在状态查询 `existence` vs yes/no 极性存在查询 `polar_existence`。
- 连续交接中的当前 `owner` 覆盖 vs 历史 `give` frame 保留。
- 属性状态中的当前 `color/access` 覆盖 vs 历史 `paint/open/close` frame 保留。
- 完整可计算查询 vs 不完整、澄清、确认、继续等语用行为。
- 个人资料陈述 vs 未完成表达：`我想说`、`我想了解` 这类片段不能被陈述层误吸成 `profile_name/profile_like`，应留给 episode 的 incomplete act。
- 可计算 Query vs 缺少上下文的篇章指代表达；后者应被 query 层拒绝并交给 episode/pragmatic 层处理。
- 单一查询 vs 复合查询。
- 普通内容查询 `contents` vs 排除指定成员后的 `contents_except`；数量查询 `count` vs 两个持有者之间的 `compare_count`。
- 可由上文实体顺序解析的 `前者/后者` vs 没有上下文的未解析指代。
- 明说请求 vs 隐含意图；简单寒暄 vs 寒暄后接任务；事实查询 vs 对某人信念或信息来源的查询。
- 条件规则的主体、物体和容器属于结构槽位，不能从最近训练样本复用旧实体；`如果 A 把 B 放进 C，B 就在 C 里` 应保留为 `if_then(antecedent=A put_in B C, consequent=B be_in C)`。
- `先/起初...后来/随后...` 是一个共享主体或共享焦点的时间并列骨架，应在 statement 学习前保持为同一个候选，而不是把后半句当作独立查询。
- 省略 Query 的目标实体可以来自当前文本，查询类型来自上一轮焦点；二者应作为独立结构槽位组合，而不是把 `X呢` 固定映射成某一个 intent。
- 同一意图结构的 train/test 变体只保留轻微构式差异，避免把无关主题变化误当成泛化能力。

## 理想模型

新增覆盖仍落到现有中间层：

- `FRAME/ROLE`: `say`、`believe`、`be_in`、`put_in`、`take_out`、`move`、`create`、`destroy`、`if_then`、`because`。
- `FRAME/ROLE` 追加约束：信念修订中的两个命题都归入 `believe(person, proposition)`，历史信念保留，当前 belief 查询选择后发生的 frame。
- `FRAME/ROLE` 追加约束：条件句 `if_then` 的 antecedent/consequent 内部也必须使用实体槽位实例化，触发条件后 consequent 可重新解析为普通事件或状态 frame。
- `QUERY` 追加约束：`query_intent` 是上下文槽位，只参与候选分类和结构消歧，不应被错误写入目标实体或查询结果槽位。
- `STATE`: `in` 当前状态由后发生状态覆盖，`owner` 由后续交接覆盖，`exists` 由 create/destroy 更新，`color/access` 由后续属性事件覆盖，历史 frame 保留。
- `STATE` 追加约束：`not_in` 只移除旧位置，若同句还有 `be_in` 修正，则当前 `REL in` 必须落到修正后的新位置。
- `QUERY`: `location`、`owner`、`color`、`object_state`、`existence`、`polar_existence`、`belief_location`、`claim_source`、`belief_source`、`contents`、`contents_except`、`count`、`compare_count`、`compound`。
- `PRAGMATIC_ACT`: `clarification_request`、`confirmation_check`、`continuation_request`、`ambiguous_reference`、`repair_previous_understanding`、`action_result_report`。
- `PRAGMATIC_ACT` 的 `action_result_report` 需要保留 `status=success/failure` 这类结果状态，不能只退化成“助手报告做过动作”。
- `INTENT`: `dialog_opening`、`dialog_then_task`、`clarification_request`、`confirmation_check`、`continuation_request`、`recall_previous_turn`、`belief_location_query`、`source_contrast_query`、`counterfactual_location_query`、`incomplete_intention`。
- `INTENT` 追加覆盖 `repair_previous_understanding` 和 `action_result_report`，分别表示用户纠正上一轮结构绑定、助手报告动作执行结果。
- `INTENT` 追加覆盖 `resolved_reference_query`，表示 `前者/后者` 先被解析成篇章实体，再进入复合查询。

## 数学表达

评估只检查结构签名，不依赖答案模板：

- Query intent、target、qualifier 和 subquery 签名。
- Frame type 与 role 槽位。
- Pragmatic act、target、qualifier。
- Intent 的 subject、goal、belief、strategy，以及 `belief_state` 中的结构谓词。
- Intent 结构谓词必须同时出现在 train/test，且 train 侧 analyzer 要能命中 test 侧同结构样本。
- `structural_pattern_*` 样本必须能被对应学习路径直接评估命中，避免样本只存在于数据文件却没有被 parser/analyzer 消费。
- Query 层的空结构样本同样纳入消费评估，用来约束“那个呢”这类上下文依赖表达不能被误吸成 farewell/apology 等纯对话行为。
- 知识型 why query 要通过 statement reject + query sample 共同约束边界，防止被陈述模型误吸成不相关的条件/因果 frame。
- 未完成表达要通过 statement reject + episode incomplete supervision 共同约束边界，防止被陈述模型误吸成个人资料状态。
- test split 中结构族是否存在。
- 每个 `structural_pattern_*` 源标签都必须同时存在 train/test 对照，避免把“结构泛化”退化成只有一侧样本的人工评估点。

## 实验验证

新增 `tests/test_structural_coverage.py`，直接读取 JSONL 并执行结构层评估：

- 训练/测试集中必须包含关键结构族。
- 新增的结构模式样本必须被 Query、Statement、Episode 解析器命中。
- 新增 `structural_pattern_causal_eval` train/test 对照，统一 `because + move + be_in` 标注，并修正旧训练样本的因果状态投影标签冲突。
- 新增 `structural_pattern_conditional_consequent` 与 `structural_pattern_statement_query_boundary`，分别覆盖条件 consequent 的 `就把...带到...` 归一和知识 why query 的 statement/query 边界。
- Intent 结构样本必须保留上下文、信念状态、目标、信念和策略，并能由 intent analyzer 在结构层匹配。
- 为 10 个 `structural_pattern_intent_context` 谓词补齐 train/test 对照，并新增 heldout 结构评估，证明同一意图结构能跨轻微表达变化匹配。
- 为 `repair_previous_understanding` 与 `action_result_report` 增加 episode 和 intent train/test 对照；episode 测试要求对应 `expected_frames` 或 `action_result` 元数据存在，并通过 train-only pragmatic analyzer 命中 test 侧样本。
- 为 `action_result_report` 增加失败状态 train/test 对照；结构测试要求 episode 的 `action_result.status` 与 pragmatic qualifiers 保留 `status=failure`，intent 的 `belief_state` 也保留同一状态谓词。
- 为 `resolved_reference_query` 增加 intent train/test 对照；结构测试要求有上下文的 `前者/后者` 解析为真实实体子查询，同时没有上下文的 `前者在哪里` 继续被拒绝。
- 为 `structural_pattern_existence_state_eval` 增加 query 与 statement train/test 对照；结构测试要求 create/destroy 写入 `exists` 状态，并区分 `existence` 与 `polar_existence` 查询。
- 为 `structural_pattern_ownership_state_eval` 增加 statement train/test 对照，并复用已有 owner 查询；结构测试要求连续 `give` 保留历史 frame，但当前 `REL owner` 只保留最后接收者。
- 为 `structural_pattern_attribute_state_eval` 增加 statement train/test 对照，并复用已有 color/object_state 查询；结构测试要求连续 `paint/open/close` 保留历史 frame，但当前 `REL color/access` 只保留最后值。
- 为 `structural_pattern_negation_correction_state_eval` 增加 statement train/test 对照；结构测试要求 `not_in + be_in` 保留两个 frame，同时当前状态只保留修正后的新位置。
- 为 `structural_pattern_belief_revision_state_eval` 增加 statement train/test 对照；结构测试要求两个信念 frame 都被保留，且 belief-location 查询使用后发生的信念。
- 为 `structural_pattern_statement_incomplete_boundary` 增加 statement train/test 对照；结构测试和 dialogue 回归共同约束不完整表达不写入错误 profile 状态。
- 为 `structural_pattern_consequent_slot_eval` 增加 statement train/test 对照，约束 `就在...里` 的构式标记不进入实体槽位。
- 为 `structural_pattern_conditional_slot_eval` 增加 statement train/test 对照；结构测试要求 `if_then` 的 antecedent/consequent 使用当前句实体，并在条件满足后展开为新的 `be_in` frame。
- 为 `structural_pattern_collection_query_eval` 增加 query train/test 对照；结构测试要求“是否一样多”归一为带 `left/right` 的 `compare_count`，“除...之外”归一为带 `exclude` 的 `contents_except`，并与普通 `contents` 保持区分。
- 在 lexer 增加时间并列骨架合并，并用结构测试约束 `先/起初...后来/随后...` 在 statement 学习前保持完整，防止共享主体丢失和 Query 负迁移。
- 为 `structural_pattern_contextual_ellipsis_eval` 增加同句不同上下文的 query train/test 对照；结构测试要求 `盒子呢` 在 `focus_query_intent=location` 时归一为 `location(盒子)`，在 `focus_query_intent=contents` 时归一为 `contents(盒子)`，并验证 working memory 到 Query 输入的实际链路。
- 端到端只检查 `Structure.linearize()` 中的结构行，用来确认事实/信念/来源/复合查询不会被表层问法合并。
- 为 query 与 statement 的全部 `structural_pattern_*` 源标签补齐 train/test 对照，并在测试中逐源检查，保证后续新增结构模式必须自带训练输入和 heldout 验证面。
