# Tau-Bench 上下游 Agent 通信协议改造计划

## 1. 目标

在 Tau-Bench 原始任务基础上，构建一个用于评测上下游 Agent 信息传输协议的 benchmark。核心目标不是重新设计任务，而是在原始单 Agent 任务中加入一个 handoff 过程，评估 Agent A 执行部分任务后，是否能通过压缩上下文包将任务状态准确传递给 Agent B，使 Agent B 能继续完成剩余任务。

测评对象：

```text
Agent A → Context Compression + Semantic Alignment → Agent B
```

需要评估的问题：

```text
1. 压缩上下文是否保留了关键任务状态；
2. 语义槽位和用户约束是否被正确传递；
3. Agent B 是否能基于压缩上下文继续完成任务；
4. 最终任务成功率和稳定性是否接近完整上下文传输；
5. 是否减少 token 成本和错误工具调用。
```

## 2. 基本原则

本改造应遵循以下原则：

```text
1. 保留 Tau-Bench 原始任务，不重新构造大量新任务；
2. 保留原始 user simulator、policy、tools、database 和 evaluator；
3. 保留最终状态自动判分，避免使用大模型主观评审；
4. 只在任务中间增加 Agent A 到 Agent B 的 handoff 机制；
5. Agent A 和 Agent B 都可以执行任务、调用工具；
6. Agent B 不直接获得完整历史，只接收 handoff package；
7. 主指标仍使用任务成功率和稳定性，协议指标只作为分析项。
```

## 3. 改造对象

优先选择：

```text
τ-retail
τ-airline
```

暂不优先使用：

```text
τ² telecom：涉及 dual-control，用户也会改变环境，复杂度较高；
τ³ voice / knowledge：引入语音和知识检索变量，不适合作为第一阶段主线。
```

第一阶段只在 retail / airline 上完成可运行验证。

## 4. 原始 Tau-Bench 流程

原始流程为：

```text
User Simulator
      ↓
Single Agent
      ↓
Tools / Database
      ↓
Final State Evaluation
```

Single Agent 同时负责：

```text
1. 与用户交互；
2. 理解任务目标；
3. 查询必要信息；
4. 判断 policy；
5. 调用工具；
6. 修改数据库状态；
7. 完成最终任务。
```

## 5. 改造后流程

改造后流程为：

```text
User Simulator
      ↓
Agent A：执行前半段任务
      ↓
Handoff Package：上下文压缩 + 语义对齐
      ↓
Agent B：接续执行后半段任务
      ↓
Tools / Database
      ↓
Final State Evaluation
```

Agent A 和 Agent B 是两个独立 Agent。Agent A 先执行一部分任务，然后停止并生成 handoff package。Agent B 读取 handoff package，继续完成剩余任务。

## 6. Agent A 设计

Agent A 是上游任务执行 Agent，不只是收集信息，也要实际完成部分任务。

Agent A 负责：

```text
1. 接收用户请求；
2. 与用户进行必要对话；
3. 调用部分工具；
4. 查询用户、订单、商品、航班、预订等信息；
5. 判断任务是否满足 policy；
6. 获取用户选择或确认；
7. 记录已完成步骤和工具结果；
8. 生成 handoff package。
```

Agent A 输出的核心不是最终答案，而是：

```text
已完成任务状态 + 剩余任务目标 + 关键工具结果 + 用户约束 + 执行边界
```

## 7. Agent B 设计

Agent B 是下游任务接续 Agent。它不能直接访问完整历史，只能读取 Agent A 传来的 handoff package。

Agent B 负责：

```text
1. 解析 handoff package；
2. 判断 Agent A 已完成哪些子任务；
3. 判断自己还需要完成哪些子任务；
4. 继承 Agent A 的工具调用结果；
5. 检查用户确认状态和业务约束；
6. 必要时通过 recover_hint 请求恢复局部上下文；
7. 调用后续工具完成任务；
8. 使最终数据库状态达到原始 Tau 任务目标。
```

Agent B 需要避免：

```text
1. 重复执行 Agent A 已完成的工具调用；
2. 忘记用户约束；
3. 修改错误对象；
4. 在未确认时执行高风险操作；
5. 违反 policy。
```

## 8. 工具拆分方式

工具不必简单分成“只读”和“写入”。按照任务阶段拆分即可。

### Agent A 工具

Agent A 优先持有前置阶段工具，例如：

```text
retail:
- 查询用户信息
- 查询订单信息
- 查询商品信息
- 查询订单状态
- 判断是否符合取消 / 退货 / 换货规则

airline:
- 查询用户信息
- 查询预订信息
- 查询航班信息
- 查询票务规则
- 搜索可替代航班
- 计算费用或改签条件
```

### Agent B 工具

Agent B 优先持有后续执行工具，例如：

```text
retail:
- 取消订单
- 修改订单
- 发起退货
- 发起换货
- 更新退款或处理状态

airline:
- 执行改签
- 取消预订
- 更新乘客信息
- 处理退款
- 更改座位
```

允许少量基础查询工具共享，但 Agent B 不应直接拿到完整对话历史。

## 9. Handoff Package 设计

Agent A 传给 Agent B 的 handoff package 应采用结构化 JSON，而不是纯自然语言摘要。

最小字段如下：

```json
{
  "context_id": "ctx-001",
  "task_id": "retail-001",
  "domain": "retail",
  "task_goal": "帮助用户取消订单 order-123",

  "completed_subtasks": [
    "retrieved_order_status",
    "checked_cancellation_policy",
    "obtained_user_confirmation"
  ],

  "remaining_subtasks": [
    "cancel_order",
    "notify_user_result"
  ],

  "tool_trace_summary": [
    {
      "tool": "get_order",
      "result": "order-123 exists and has not been shipped",
      "ref": "tool-result-03"
    }
  ],

  "intermediate_state": {
    "order_id": "order-123",
    "order_status": "not_shipped",
    "policy_eligible": true,
    "user_confirmed": true
  },

  "semantic_frame": {
    "intent": "cancel_order",
    "slots": {
      "order_id": "order-123"
    }
  },

  "kept_constraints": [
    "只能取消尚未发货的订单",
    "必须获得用户确认后才能取消订单"
  ],

  "execution_boundary": {
    "allowed_actions": ["cancel_order"],
    "forbidden_actions": ["refund_without_policy_check"],
    "confirmation_status": "confirmed"
  },

  "refs": ["msg-03", "tool-result-03"],
  "recover_hint": "get_context(ctx-001, refs)",

  "token_full": 3800,
  "token_count": 620,
  "compression_ratio": 0.163
}
```

其中必须保留的关键字段是：

```text
completed_subtasks
remaining_subtasks
tool_trace_summary
intermediate_state
semantic_frame
kept_constraints
execution_boundary
refs
recover_hint
```

## 10. Handoff 点设计

每个原始 Tau 任务只设置一个 handoff 点。

handoff 点可以选择在：

```text
1. Agent A 已经完成必要信息查询之后；
2. Agent A 已经完成 policy 判断之后；
3. Agent A 已经获得用户确认之后；
4. Agent A 已经生成候选方案之后；
5. 需要 Agent B 执行最终状态修改之前。
```

示例：

```text
retail 取消订单任务：
Agent A：查询订单 → 判断是否可取消 → 获取用户确认
Handoff
Agent B：执行取消订单 → 通知用户结果

airline 改签任务：
Agent A：查询预订 → 搜索候选航班 → 判断改签规则 → 获取用户选择
Handoff
Agent B：执行改签 → 更新预订状态 → 通知用户结果
```

## 11. 对比方法

第一阶段只做最小必要对比，避免平行实验过多。

### Baseline 1：Single Agent

原始 Tau-Bench 单 Agent 设置。

```text
User → Single Agent → Tools
```

作用：

```text
作为原始任务参考上界。
```

### Baseline 2：Full Handoff

Agent A 执行前半段任务后，把完整对话历史和完整工具轨迹传给 Agent B。

```text
Agent A → Full Context → Agent B
```

作用：

```text
判断任务拆成两个 Agent 后，在没有压缩损失时能达到什么效果。
```

### Baseline 3：Summary Handoff

Agent A 只生成普通自然语言摘要传给 Agent B。

```text
Agent A → Natural Language Summary → Agent B
```

作用：

```text
判断普通摘要是否足以支持任务接续。
```

### Proposed：Semantic-aware Handoff

Agent A 生成结构化 handoff package，包含上下文压缩和语义对齐字段。

```text
Agent A → Structured Handoff Package → Agent B
```

作用：

```text
验证本文提出的上下文压缩 + 语义对齐协议是否有效。
```

## 12. 测评指标

主指标复用 Tau-Bench 原始评测方式。

### 主指标

```text
1. Task Success Rate / Pass^1
   单次运行任务成功率。

2. Pass^k
   同一任务多次运行是否稳定成功。

3. Final State Match
   最终数据库状态是否与原始目标状态一致。
```

### 辅助分析指标

```text
1. Compression Ratio
   压缩后 token 数 / 完整上下文 token 数。

2. Token Saving Rate
   1 - Compression Ratio。

3. Completed Subtask Recall
   Agent A 已完成子任务是否被正确传递。

4. Remaining Subtask Accuracy
   Agent B 是否正确识别剩余任务。

5. Slot Accuracy
   关键槽位是否正确传递。

6. Constraint Recall
   用户约束和业务规则是否被保留。

7. Duplicate Action Rate
   Agent B 是否重复执行 Agent A 已完成的操作。

8. Boundary Violation Rate
   Agent B 是否违反确认状态、授权范围或禁止操作。
```

主结论以 Task Success Rate 和 Pass^k 为准，辅助指标用于错误分析。

## 13. 实现步骤

### Step 1：跑通原始 Tau-Bench

先选择 retail 或 airline 的少量任务，确认原始 Single Agent 能正常运行，并能输出 trajectory、tool calls、final state 和 evaluation result。

输出：

```text
original_run_results.json
original_trajectories/
```

### Step 2：实现双 Agent wrapper

在原始 Agent 外层增加 Agent A / Agent B wrapper。

需要实现：

```text
1. Agent A 运行到 handoff 点后停止；
2. 保存 Agent A 的 messages、tool calls、tool results；
3. Agent A 生成 handoff package；
4. Agent B 接收 handoff package；
5. Agent B 继续和环境交互；
6. 最终仍调用 Tau 原始 evaluator。
```

输出：

```text
two_agent_wrapper.py
handoff_runtime.py
```

### Step 3：设计 handoff 点

为每个任务记录 handoff 点。

可以先人工设置少量任务的 handoff 点，例如 10-20 条任务。

字段格式：

```json
{
  "task_id": "retail-001",
  "handoff_after": "checked_policy_and_obtained_confirmation",
  "agent_a_stop_condition": "before_final_write_tool",
  "agent_b_start_condition": "execute_remaining_task"
}
```

输出：

```text
handoff_config.json
```

### Step 4：实现三种 handoff 方法

需要实现：

```text
1. Full Handoff
   直接传完整 conversation + tool trace。

2. Summary Handoff
   用普通自然语言摘要压缩上下文。

3. Structured Handoff
   生成包含 semantic_frame、kept_constraints、execution_boundary 的结构化 JSON。
```

输出：

```text
handoff_full.py
handoff_summary.py
handoff_structured.py
```

### Step 5：实现 Agent B 解析器

Agent B 需要能够读取 handoff package，并将其转化为可执行上下文。

Agent B prompt 中必须明确：

```text
1. 已完成子任务不要重复执行；
2. 剩余子任务需要继续完成；
3. 必须遵守 kept_constraints；
4. 高风险工具调用前检查 execution_boundary；
5. 信息不足时使用 recover_hint；
6. 最终目标仍是完成原始 Tau 任务。
```

输出：

```text
agent_b_prompt_template.md
handoff_parser.py
```

### Step 6：接入 Tau 原始 evaluator

最终任务完成后，不新增主观评审，直接使用 Tau 原始 evaluator 判断 final state 是否正确。

需要记录：

```text
task_success
final_state_match
pass_k
tool_calls
token_usage
handoff_package
```

输出：

```text
eval_results.json
```

### Step 7：记录分析指标

从 trajectory 和 handoff package 中自动统计：

```text
compression_ratio
token_saving_rate
duplicate_action_rate
boundary_violation_rate
tool_call_count
recovery_count
```

对于 Slot Accuracy、Constraint Recall 等指标，第一阶段可以先用小规模人工标注的 gold_handoff 做验证。

输出：

```text
analysis_metrics.json
```

## 14. 第一阶段最小可行目标

第一阶段不要做全量 benchmark，只做最小闭环。

建议范围：

```text
domain: retail
task number: 10-20
methods: Single Agent / Full Handoff / Summary Handoff / Structured Handoff
trials: 每个任务运行 3-4 次
metrics: task success, pass^k, compression ratio, duplicate action rate
```

完成标准：

```text
1. 原始 Single Agent 可以正常跑通；
2. 双 Agent wrapper 可以正常完成任务；
3. Full Handoff 能接近 Single Agent 表现；
4. Structured Handoff 的成功率高于 Summary Handoff；
5. Structured Handoff 的 token 使用低于 Full Handoff；
6. 能输出完整 trajectory 和 handoff package。
```

## 15. 最终交付物

需要交付：

```text
1. 双 Agent 改造代码；
2. handoff_config.json；
3. handoff package 生成模块；
4. Agent B 解析与执行模块；
5. 对比实验脚本；
6. 运行结果表；
7. 失败案例分析；
8. 简短实验报告。
```

## 16. 核心判断标准

该 benchmark 是否成功，不看摘要是否写得流畅，而看：

```text
Agent B 能不能基于 Agent A 传来的压缩上下文继续做任务；
上下游 Agent 合作后最终状态是否正确；
同一任务多次运行是否稳定；
相比完整上下文传输，是否减少 token；
相比普通摘要，是否减少重复执行、槽位错误和约束违反。
```

最终结论应围绕：

```text
语义感知 handoff package 是否能作为上下游 Agent 之间可靠的信息传输协议。
```
