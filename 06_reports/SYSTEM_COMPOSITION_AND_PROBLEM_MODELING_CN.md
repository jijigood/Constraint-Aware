# 面向论文写作的系统组成与问题建模说明

**题目建议：** RAG 编译可验证约束与确定性安全屏蔽的安全 DRL O-RAN 切片控制  
**日期：** 2026-06-25  
**项目目录：** `/home/huangxiaolin/safe_drl_oran/`  
**定位：** 当前主线研究方向；RCA / Route B′-Train 作为支撑性经验研究，不再是主论文主线。

---

## 摘要式概述

本研究关注动态流量、异构 SLA 与分布漂移下的 O-RAN 网络切片安全控制。系统中，DRL 控制器负责资源分配，安全屏蔽层负责保证 URLLC 等关键切片不被不安全动作破坏。前期实验已经表明：动态、负载感知的 safety shield 能在静态 reservation 失败的高 URLLC 场景下接近 oracle 安全性；DRL 能提升 reward 和学习效率；但让通用 LLM 直接生成 `urllc_min_prb` 这类安全关键数值，即使从 Qwen3-1.7B 扩展到 32B，也无法接近 oracle，主要表现为系统性 under-reservation。

因此，当前系统不再把 LLM 作为端到端控制器或安全数值生成器，而是将 LLM/RAG 定位为 **符号约束编译器**：它读取 SLA、意图、标准文本和策略证据，输出类型化、可验证的约束规格；随后由确定性 verifier 和 shield 计算并执行安全关键数值。核心思想是：**LLM 负责语义解释和约束编译，确定性模块负责数值计算和动作投影。**

---

## 1. 研究定位

### 1.1 研究动机

O-RAN 切片控制需要在多个业务切片之间分配有限无线资源，例如：

| 切片 | 主要目标 | 典型风险 |
|---|---|---|
| eMBB | 高吞吐 | 过度占用 PRB，挤压低时延业务 |
| URLLC | 低时延、高可靠 | 未及时服务导致 SLA violation |
| mMTC | 大规模接入 | 服务比例下降、接入覆盖不足 |

DRL 适合在动态环境中学习长期 reward，但在训练探索和分布漂移时可能产生不安全动作。安全屏蔽层可以把 DRL 的动作投影到安全可行集合中，但固定静态 reservation 无法同时适配 high-eMBB 和 high-URLLC 等不同流量 regime。

因此，本研究的核心问题是：

> 如何利用 RAG/LLM 从 SLA、标准和运营意图中提取动态安全约束，同时避免让 LLM 直接承担安全关键数值计算？

### 1.2 已建立的实验证据

当前系统设计建立在以下实验证据之上：

1. **动态 shield 的空间是真实的。** Phase 0 证明：high-URLLC 下静态 reservation 无法达到安全，load/SLA-aware oracle shield 可以接近安全。
2. **shield 对 DRL 有价值。** Phase 1 证明：shield 显著降低训练期 unsafe exploration，并提升 cross-regime robustness。
3. **直接 LLM 数值 shield 失败。** Phase 2a/2a-v2 证明：Qwen3-1.7B/4B/14B/32B 均无法产生接近 oracle 的 `urllc_min_prb`，主要是 under-reservation。
4. **LLM 的格式和引用能力可靠。** Phase 2a-v2 中 schema/citation 可达到 1.00，说明 LLM 擅长结构化输出和证据引用，但不擅长安全关键数值决策。
5. **RCA 支撑线给出相同教训。** Route B′-Train v1 显示 LLM 可通过训练提升格式、引用和抗干扰能力，但 source-aware ordering 与 scalar reward/DPO 不能未经验证地作为主机制。

综合结论：

> LLM 可以解释和编译规则，但不应直接输出安全关键控制数值。

---

## 2. 系统组成

当前论文式系统可以抽象为七层：

```text
SLA / intent / standards / policy evidence
        ↓
RAG retrieval
        ↓
LLM symbolic constraint compiler
        ↓
typed constraint specification
        ↓
verifier + fail-closed policy
        ↓
deterministic safety shield
        ↓
DRL slicing controller → safe PRB allocation
        ↓
environment + evaluator + honest gate
```

### 2.1 O-RAN 切片环境

环境代码位于：

- `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_env.py`
- `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_gym_env.py`

环境模拟三个切片共享固定 PRB 资源池：

- eMBB：吞吐收益高；
- URLLC：当前 slot 必须服务 offered load + backlog；
- mMTC：服务比例代表接入质量。

环境包含：

- 离散 PRB 分配动作；
- high-eMBB 与 high-URLLC 流量 regime；
- 昼夜变化；
- burst；
- channel variation；
- backlog；
- per-slice SLA 指标；
- reward 与 component metrics。

### 2.2 DRL 控制器

DRL 控制器输出候选 PRB allocation：

\[
a_t = (p_t^{embb}, p_t^{urllc}, p_t^{mmtc})
\]

Phase 1 已包含：

- PPO；
- DQN；
- proportional-fair rule baseline；
- no shield / static shield / oracle shield 变体。

相关代码：

- `/home/huangxiaolin/safe_drl_oran/01_code/drl/train_baselines.py`
- `/home/huangxiaolin/safe_drl_oran/01_code/drl/eval_baselines.py`
- `/home/huangxiaolin/safe_drl_oran/01_code/drl/drl_common.py`

在论文中应明确：**DRL 是控制器，LLM 不替代 DRL。**

### 2.3 确定性安全屏蔽层

安全屏蔽层接收 DRL 提出的动作 \(a_t\)，以及由约束规格和状态计算出的安全约束 \(\mathcal{C}_t\)，输出安全动作：

\[
a_t^{safe} = \Pi_{\mathcal{C}_t}(a_t)
\]

其中：

- \(a_t\)：DRL 原始动作；
- \(\mathcal{C}_t\)：当前时刻安全可行集合；
- \(\Pi_{\mathcal{C}_t}\)：确定性投影算子。

当前实现分布在：

- `_project_to_min_urllc` 与 oracle shield：`01_code/env/slicing_env.py`
- Phase 2a producer / parser：`01_code/rag/constraint_producers.py`
- counterfactual safety scoring：`01_code/rag/counterfactual.py`
- oracle reservation / scoring credibility：`01_code/rag/scoring_credibility.py`

后续工程上可以抽成 `01_code/shield/`，但论文建模中已经可以将其表述为独立 deterministic shield layer。

### 2.4 RAG 证据层

RAG 层负责检索：

- SLA 文本；
- O-RAN / 3GPP 标准；
- operator intent；
- policy template；
- 可靠性、低时延、切片优先级相关文本。

它复用已有电信 RAG 资产：

- 2,070 文档 O-RAN/3GPP KB；
- BGE-M3 embedding cache；
- 本地 vLLM / Qwen 推理栈。

RAG 的作用不是输出 PRB 数字，而是提供：

- SLA metric；
- reliability target；
- latency semantics；
- slice priority；
- applicable policy；
- citation provenance。

### 2.5 LLM 符号约束编译器

LLM 读取检索证据、运营意图和可选的网络摘要，输出 **typed constraint specification**，而不是输出 `urllc_min_prb`。

示例：

```json
{
  "constraint_type": "urllc_latency_reliability",
  "slice": "urllc",
  "metric": "slot_level_latency_service",
  "service_rule": "serve_offered_load_plus_backlog_in_current_slot",
  "reliability_target": 0.99,
  "channel_margin_policy": "pessimistic_quantile",
  "formula_id": "load_backlog_over_spectral_efficiency",
  "units": {
    "load": "Mbps",
    "capacity": "Mbps_per_PRB",
    "reservation": "PRB"
  },
  "applicability": {
    "traffic_regime": ["high_urllc", "bursty", "mixed"],
    "slice_priority": "URLLC before eMBB throughput"
  },
  "citations": ["sla_doc_3", "oran_policy_7"]
}
```

核心设计约束：

> LLM 可以选择约束类型、可靠性目标、公式 ID、适用条件和引用证据，但不能直接填写安全关键 reservation 数值。

### 2.6 Verifier

Verifier 负责在约束进入 shield 前进行机械检查。

| 检查项 | 目的 |
|---|---|
| JSON/schema validity | 输出可解析、字段完整 |
| type validity | 字段类型正确 |
| unit validity | 单位一致 |
| citation validity | 引用 ID 存在于检索证据 |
| range validity | reliability / threshold 合法 |
| formula validity | `formula_id` 属于白名单 |
| feasibility | 约束在 PRB budget 下可执行 |
| monotonicity sanity | 更高可靠性不应导致更低 reservation |

若验证失败：

```text
invalid spec → fail-closed fallback → conservative shield
```

这使得 LLM 错误变成可检测的规格错误，而不是直接变成 unsafe action。

---

## 3. 问题建模

### 3.1 Constrained MDP

O-RAN slicing 被建模为约束马尔可夫决策过程：

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \mathcal{C}, \gamma)
\]

其中：

- \(\mathcal{S}\)：网络状态空间；
- \(\mathcal{A}\)：PRB allocation 动作空间；
- \(P\)：流量、信道、backlog 转移；
- \(r\)：reward；
- \(\mathcal{C}\)：SLA 诱导的安全约束集合；
- \(\gamma\)：折扣因子。

### 3.2 状态空间

状态可写为：

\[
s_t =
\{
d_t^{embb}, d_t^{urllc}, d_t^{mmtc},
b_t^{embb}, b_t^{urllc}, b_t^{mmtc},
g_t,
\tau_t
\}
\]

其中：

- \(d_t^i\)：slice \(i\) 的 offered load；
- \(b_t^i\)：slice \(i\) 的 backlog；
- \(g_t\)：channel gain；
- \(\tau_t\)：episode 内归一化时间。

当前实现中，该状态对应 8 维 observation vector。

### 3.3 动作空间

动作是三个切片的 PRB 分配：

\[
a_t = (p_t^{embb}, p_t^{urllc}, p_t^{mmtc})
\]

满足：

\[
p_t^{embb} + p_t^{urllc} + p_t^{mmtc} = P_{\max}
\]

以及离散粒度：

\[
p_t^i \in \{0, \Delta p, 2\Delta p, \dots, P_{\max}\}
\]

当前环境中：

- \(P_{\max}=100\)；
- \(\Delta p=10\)。

### 3.4 Reward

reward 同时考虑吞吐、mMTC 服务比例、URLLC 安全奖励和 URLLC 违规惩罚：

\[
r_t =
w_e \cdot \text{served}^{embb}_t
+ w_m \cdot \text{served}^{mmtc}_t
+ w_u \cdot \mathbb{1}[\text{URLLC safe}]
- \beta \cdot \mathbb{1}[\text{URLLC violation}]
\]

论文中不应只报告 scalar reward，还要报告：

- URLLC violation rate；
- eMBB SLA rate；
- mMTC SLA rate；
- Jain fairness；
- PRB utilization / reservation；
- training-time unsafe exploration；
- reward at matched safety。

### 3.5 URLLC 安全约束

URLLC 安全要求当前 slot 服务 offered load + backlog：

\[
\text{capacity}^{urllc}_t
\ge
d_t^{urllc} + b_t^{urllc}
\]

其中：

\[
\text{capacity}^{urllc}_t
=
p_t^{urllc} \cdot se^{urllc} \cdot g_t
\]

为满足可靠性目标，引入 pessimistic channel：

\[
g_t^{pess} = f(g_t, \rho)
\]

其中 \(\rho\) 是由 SLA / intent 编译得到的 reliability target。

确定性 solver 计算：

\[
p_{min,t}^{urllc}
=
\left\lceil
\frac{
d_t^{urllc} + b_t^{urllc}
}{
se^{urllc} \cdot g_t^{pess}
}
\right\rceil_{\Delta p}
\]

这里 \(\lceil \cdot \rceil_{\Delta p}\) 表示向上取整到 PRB 粒度。

这是安全关键数值，必须由确定性 solver 计算，而不是由 LLM 直接生成。

### 3.6 Shield Projection

给定 DRL 原始动作 \(a_t\)，shield 解：

\[
a_t^{safe}
=
\arg\min_{a \in \mathcal{A}}
\|a-a_t\|_1
\]

约束为：

\[
p^{urllc} \ge p_{min,t}^{urllc}
\]

也就是说，shield 尽量少改 DRL 动作，同时保证 URLLC reservation floor。

### 3.7 RAG-Compiled Constraint

令：

- \(D\)：SLA / 标准 / 策略文档集合；
- \(I\)：operator intent；
- \(q\)：由 intent 和任务生成的检索 query；
- \(E_q = \text{Retrieve}(q, D)\)：检索证据；
- \(x_t\)：当前网络摘要。

LLM compiler 输出：

\[
z_t = \text{Compile}_{LLM}(I, E_q, x_t)
\]

其中 \(z_t\) 是符号约束规格，而不是动作或 PRB 数字。

Verifier 判断：

\[
V(z_t) \in \{\text{valid}, \text{invalid}\}
\]

若 valid：

\[
p_{min,t}^{urllc} = F(z_t, s_t)
\]

若 invalid：

\[
p_{min,t}^{urllc} = F_{fallback}(s_t)
\]

整个安全路径是：

```text
LLM semantic compile → verifier → deterministic formula → shield projection
```

而不是：

```text
LLM directly emits safety-critical PRB value
```

### 3.8 优化目标

优化目标是：

\[
\max_{\pi}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T}
\gamma^t r(s_t, a_t^{safe})
\right]
\]

满足：

\[
\Pr[\text{URLLC violation}] \le \epsilon
\]

并且：

\[
a_t^{safe}
=
\Pi_{\mathcal{C}(z_t,s_t)}(\pi(s_t))
\]

Phase 2b 的目标不是证明 LLM 会控制网络，而是证明：

> LLM/RAG 编译出的约束规格经过 verifier 和 deterministic shield 后，可以接近 oracle safety，同时避免 Phase 2a 中直接数值生成的 calibration failure。

---

## 4. 研究问题

### Q1：动态 shield 是否必要？

已由 Phase 0 支持：

- static reservation 在 high-URLLC 下无法达到安全；
- load/SLA-aware oracle shield 接近安全；
- static reservation 无法跨 regime 泛化。

### Q2：shield 对 DRL 是否有价值？

已由 Phase 1 支持：

- shield 显著降低 unsafe exploration；
- shield 提升 distribution-shift robustness；
- DQN high-URLLC 收敛安全性明显改善；
- PPO 单 regime 下可部分 internalize safety，因此主 claim 应聚焦训练安全和 cross-regime。

### Q3：off-the-shelf LLM 能否直接输出安全数值？

已由 Phase 2a/2a-v2 否定：

- 1.7B 到 32B 单调变好但仍远离 oracle；
- persistent under-reservation；
- RAG 提升 citation，不显著提升控制决策；
- schema/citation success 不等于安全控制成功。

### Q4：LLM 能否编译可验证符号约束，并由确定性 shield 执行？

这是下一阶段 Phase 2b 的核心研究问题。

推荐 arms：

| Arm | 描述 |
|---|---|
| static shield | 固定 URLLC reservation |
| direct LLM numeric | Phase 2a 中失败的直接 `urllc_min_prb` |
| LLM symbolic compiler + solver | 无 RAG，仅 LLM 编译符号规格 |
| RAG-LLM symbolic compiler + solver | 主方法 |
| oracle_margin | 安全上界参考 |

---

## 5. Go / No-Go Gates

### G1 — Safety and Reward

主方法应接近 oracle safety frontier：

```text
URLLC violation <= oracle_margin + epsilon
AND reward >= oracle_margin - delta
AND safer than static under cross-regime shift
```

注意：不要在 unmatched safety 下用 reward 和 unsafe static 直接比较。

### G2 — Verifiable Soundness

要求：

```text
typed_spec_validity high
zero unsafe specs pass verifier
invalid specs fail closed
```

Verifier 拒绝率高本身不是失败；只要 fallback 安全且透明报告即可。

### G3 — RAG Sensitivity

RAG 必须正确影响约束规格：

```text
changing SLA / reliability / intent text changes the compiled spec correctly
```

不能重复 Phase 2a 的问题：RAG 只改善 citation，但 action/spec 不变。

### G4 — Shift Robustness

主方法应在以下变化中保持安全：

- high-eMBB；
- high-URLLC；
- bursty regime；
- reliability target 变化；
- noisy / ambiguous SLA text；
- missing or conflicting evidence。

---

## 6. 评价指标

| 指标 | 含义 |
|---|---|
| URLLC violation rate | 核心安全指标 |
| mean reward | 控制效用 |
| reward at matched safety | 公平比较不同 shield |
| eMBB SLA rate | 吞吐服务保持 |
| mMTC SLA rate | 接入服务保持 |
| Jain fairness | 资源分配均衡 |
| mean / p95 reservation | shield 保守程度 |
| typed spec validity | compiler 结构正确性 |
| citation validity | RAG grounding 纪律 |
| verifier rejection rate | 规格错误识别能力 |
| unsafe-spec pass rate | 应为 0 |
| fallback rate | fail-closed 使用频率 |
| RAG sensitivity score | SLA/intent 变化是否正确反映到规格 |

---

## 7. 可写 Claim 与不可写 Claim

### 已有证据支持的 claim

1. Static shield 在 traffic-regime shift 下不足。
2. Shield 对 DRL 的 safe exploration 和 cross-regime robustness 有价值。
3. 直接 LLM numeric safety generation 在本地模型尺度内失败。
4. schema/citation 成功不等于安全控制成功。

### Phase 2b 要检验的 claim

1. RAG-LLM symbolic compiler + deterministic shield 能接近 oracle safety。
2. Verifier 能阻止错误规格进入 safety-critical path。
3. RAG 的价值体现在正确改变 typed constraints，而不仅是 citation。

### 应避免的 claim

- LLM 直接控制网络；
- LLM 直接可靠计算安全 reservation；
- RAG 自动改善控制决策；
- 当前系统可直接部署到真实 O-RAN；
- RCA/source-aware 线是主贡献。

---

## 8. 与 RCA 支撑研究的关系

Track A → Route B → Route B′-Train 并不是废弃线，而是支撑主线的经验基础：

- Track A 证明 confidence / retrieve-or-stop 不可靠；
- Route B 证明 inference-time sufficiency selection 也不可靠；
- Route B′-Train 证明训练可提升抗干扰，但 source-aware 与 scalar reward/DPO 不能无门控地信任；
- 这些结果共同支持当前结论：LLM 不应进入安全关键数值路径。

因此 RCA 线在论文中应作为 motivation / supporting study，而不是主方法。

---

## 9. 复现实验资产

| 资产 | 路径 |
|---|---|
| Phase 0 计划 | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase0.md` |
| Phase 1 计划 | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase1.md` |
| Phase 2a 计划 | `/home/huangxiaolin/safe_drl_oran/06_reports/PLAN_phase2a.md` |
| 切片环境 | `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_env.py` |
| Gym wrapper | `/home/huangxiaolin/safe_drl_oran/01_code/env/slicing_gym_env.py` |
| DRL 训练/评估 | `/home/huangxiaolin/safe_drl_oran/01_code/drl/` |
| Phase 2a producer | `/home/huangxiaolin/safe_drl_oran/01_code/rag/constraint_producers.py` |
| counterfactual scorer | `/home/huangxiaolin/safe_drl_oran/01_code/rag/counterfactual.py` |
| scoring credibility | `/home/huangxiaolin/safe_drl_oran/01_code/rag/scoring_credibility.py` |
| Phase 0 结果 | `/home/huangxiaolin/safe_drl_oran/04_results/phase0_headroom.json` |
| Phase 1 结果 | `/home/huangxiaolin/safe_drl_oran/04_results/phase1/` |
| Phase 2a 结果 | `/home/huangxiaolin/safe_drl_oran/04_results/phase2a/` |

---

## 10. 论文结构建议

```text
1. Introduction
   - O-RAN slicing safe DRL
   - why direct LLM numeric control is unsafe

2. Empirical Motivation
   - Phase 0 dynamic shield headroom
   - Phase 1 DRL + shield value
   - Phase 2a direct LLM numeric no-go

3. System Architecture
   - RAG evidence layer
   - LLM symbolic constraint compiler
   - verifier
   - deterministic shield
   - DRL controller

4. Problem Formulation
   - constrained MDP
   - URLLC reliability/latency constraint
   - shield projection
   - compile -> verify -> solve formalization

5. Experiments
   - static / direct LLM numeric / symbolic compiler / RAG-symbolic compiler / oracle
   - SLA variation
   - traffic shift
   - verifier/fallback analysis

6. Results
   - safety-reward Pareto
   - verifier soundness
   - RAG sensitivity
   - shift robustness

7. Discussion
   - LLM reliability boundary
   - deterministic safety enforcement
   - limitations and real O-RAN path

8. Conclusion
```

---

## 11. 一句话主论点

> RAG-LLM 适合把 SLA 与运营意图编译成类型化、可验证的约束规格；但安全 O-RAN DRL 需要确定性数值执行。LLM 应解释和编译规则，不应直接计算安全关键 reservation。

