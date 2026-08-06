# Symbolic Baseline 能力目录

本文件是已交付 symbolic baseline 能力集合的单一事实来源。它按任务族分组，而不是把目录塞回 README。

## 1. 状态与事实跟踪

- 当前状态关系：`in`、`at`、`owner`、`color`、`access`、`exists`、`name`、`likes`、`dislikes`
- 状态投影与覆盖：`put_in`、`take_out`、`move`、`give`、`paint`、`open`、`close`、`create`、`destroy`、`be_in`、`not_in`、`profile_name`、`profile_like`、`profile_dislike`
- 基本状态行为：当前状态覆盖、否定删除、修正覆盖、销毁清理

## 2. 事件与角色查询

- 复合查询组合：`compound`
- 事件执行者查询：`actor_for_event`、`latest_actor_for_event`、`earliest_actor_for_event`
- 物品处理者查询：`actor_for_item`、`latest_actor_for_item`
- 按执行者追踪历史：`actions_by_actors`、`inventories`
- 事件匹配使用的角色别名：`item`、`holder`、`source`、`actor`、`theme`、`goal`、`recipient`
- 物品使用历史 frame：`handle`

## 3. 位置、包含关系与数量

- 位置查询：`location`、`initial_location`、`location_before_actor_action`、`location_before_event`、`location_after_event`
- 包含关系查询：`contents`、`contents_except`、`contents_before_event`、`contents_after_event`、`places_visited`
- 比较查询：`count`、`compare_count`、`same_location`
- 极性变体：`polar_location`、`polar_contents`、`polar_existence`
- 嵌套闭包：多层容器与地点传播

## 4. 存在与属性跟踪

- 存在查询：`existence`
- 属性查询：`owner`、`color`、`object_state`
- 极性存在与销毁后的存在清理

## 5. 因果、信念与矛盾推理

- 因果查询：`why`、`counterfactual_location`
- 因果 frame：`if_then`、`because`
- 转述与信念 frame：`say`、`believe`
- 来源查询：`claim_source`、`belief_source`
- 个人视角查询：`belief_location`
- 一致性检查：`contradictions`

## 6. 对话与档案

- 对话意图：`dialog_act`
- 默认对话目标：`greeting`、`thanks`、`farewell`、`identity`、`capabilities`、`summary`
- 档案查询意图：`profile`
- 续问片段的指代消解：`this chip`、`here` 以及类似的上下文指针

## 7. 学习输入

- 陈述训练集：`data/statement_examples.jsonl`
- Query 训练集：`data/query_examples.jsonl`
- 意图训练 / 反馈集：`data/intent_examples.jsonl`
- 运行时产物：`data/statement_model.json`、`data/query_model.json`、`data/dialog_answer_model.json`
