# =============================================================================
# strategy_policy.py — 确定性策略层（Deterministic Policy Layer）
# =============================================================================
# 本文件是系统中所有规则化、不需要 LLM 的决策逻辑的集中地。
#
# 与 LLM Agent 的关系：
#   - LLM（llm_decision.py）输出"柔性决策"，本文件输出"确定性决策"
#   - LLM 的 fallback 参数全部由本文件的函数产生
#   - 当 USE_LLM_AGENTS=False 时，系统完全依靠本文件的规则运行
#   - 当 LLM 输出不合法时，本文件的规则充当安全阀兜底
#
# 包含五大决策域：
#   1. 评估 Gate 判定     — decide_evaluation_gates()
#   2. 教师搜索策略       — decide_teacher_search()
#   3. 策略巡检决策       — decide_inspection()
#   4. 参数大师 Action 选择 — decide_parameter_master_action()
#   5. DAG 更新 & Replay   — update_search_dag_policy() / build_replay_entries_policy()
#
# 辅助能力：
#   - MCTS UCB 节点/Action 选择
#   - 数据窗口偏移管理（避免重复使用同批次数据）
#   - 候选模型评分（reward 函数）
#   - MCTS 边历史查询与摘要
# =============================================================================

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import (
    DATASET_SHARD_COUNT,              # 数据集分片数量
    DATASET_CACHE_MODE,               # 数据缓存运行模式
    DATASET_OFFSET_CACHE_MODE,        # 是否强制 offset cache 窗口模式
    DATASET_SHARD_SELECTION_POLICY,   # 数据窗口选择策略
    DATASET_SHARD_SIZE,               # 单个分片大小
    DATA_HARD_RATIO_MERGE_THRESHOLD,  # hard 比例超过此值 → 触发 merge/replace 策略
    DATA_MIN_TRAIN_QUESTIONS_PER_ROUND, # 每轮最少训练题数，低于此值触发低训练量信号
    EVAL_NEW_SKILL_GATE,              # 新能力 gate 阈值
    EVAL_OLD_SKILL_GATE,              # 旧能力 gate 阈值
    EVAL_PROBE_ACC_GATE,              # 探针准确率 gate 阈值
    FROZEN_DEGRADE_TOLERANCE,         # frozen probe 退化容忍度
    TEST_FORGETTING_TOLERANCE,        # 遗忘容忍度
    HOLDOUT_ERROR_TOLERANCE,          # holdout 误差容忍度
    MAX_ROUNDS,                       # 最大训练轮次
    MCTS_ACTION_SPACE_PATH,           # MCTS action space 配置文件路径
    MCTS_EXPLORATION_C,               # MCTS UCB 探索系数（通用）
    MCTS_EXPLORATION_C_ACTION,        # MCTS UCB 探索系数（action 选择）
    MCTS_EXPLORATION_C_NODE,          # MCTS UCB 探索系数（节点选择）
    MCTS_QUERY_ACTION_BIAS_MAX,       # DAG 检索证据对 action UCB 的最大加权
    MCTS_QUERY_CANDIDATE_TOP_K,       # DAG 检索保留候选节点数量
    MCTS_QUERY_COLD_START_EDGES,      # DAG 检索冷启动最少边数
    PARAMETER_MASTER_CARD_TOP_EDGES_PER_NODE, # 参数大师 evidence card 每节点边数
    PARAMETER_MASTER_CARD_TOP_NODES,  # 参数大师 evidence card 节点数
    MCTS_QUERY_HIGH_FORGETTING_THRESHOLD, # 高遗忘节点检索阈值
    MCTS_QUERY_LOW_COTEST_THRESHOLD,  # 低 cotest 节点检索阈值
    MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD, # 低新能力节点检索阈值
    MCTS_QUERY_LOW_PROBE_THRESHOLD,   # 低 probe 节点检索阈值
    MCTS_QUERY_ROLLBACK_STREAK_THRESHOLD, # 连续 rollback 检索阈值
    MCTS_CONTINUOUS_ACTION_ENABLED,     # 是否启用连续参数 surrogate-UCB
    MCTS_CONTINUOUS_BANDWIDTH,          # 连续参数 RBF surrogate 带宽
    MCTS_CONTINUOUS_CANDIDATES,         # 每轮连续搜索候选数
    MCTS_CONTINUOUS_EPOCH_COST_WEIGHT,  # epochs 训练成本惩罚权重
    MCTS_CONTINUOUS_EXPLORATION_BETA,   # 连续参数探索 bonus 系数
    MCTS_MUTATION_SCALE_MAX,          # 参数变异缩放上限
    MCTS_MUTATION_SCALE_MIN,          # 参数变异缩放下限
    MCTS_TUNER_COLD_START_EDGES,      # MCTS 冷启动最少边数（不足时不调参）
    MIN_NEW_SKILL_GAIN,               # 最小新能力增益
    PROMOTION_TOTAL_RELATIVE_GAIN,    # 晋升所需 frozen/old/new 总相对收益
    REPLAY_BUFFER_MAX_SIZE,           # 回放池最大容量
    REWARD_FORGETTING_PENALTY_WEIGHT, # reward 中遗忘惩罚权重
    REWARD_GATE_BONUS,                # 通过 gate 的额外奖励
    REWARD_STOP_BONUS,                # 达到停止条件的额外奖励
    SCREENING_MIN_NEXT_WINDOW_REMAINDER, # 数据窗口最小剩余量（不够则跳到下一个窗口）
)
from src.models.messages import EvalResultPayload
from src.tools.replay_buffer import create_replay_entry
from src.tools.search_query import fallback_query_from_goal
from src.tools.search_dag import add_search_edge, add_search_node, backpropagate_value

# ─────────────────────────────────────────────────────────────────────────────
# 新 Agent 提示词的延迟导入（避免循环依赖）
# 实际使用时从 agent_prompts 中按 key 取，配合 prompt_for_agent() 使用
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Frozen Dataclass 定义（5 个决策结果类型）
# 全部使用 frozen=True 以确保决策不可变 —— 一旦创建就不能修改，
# 防止下游代码意外篡改策略决策。
# ─────────────────────────────────────────────────────────────────────────────


# ── InspectionPolicyDecision ─────────────────────────────────────────────────
# 策略巡检的最终决策结果。
# 由 decide_inspection() 产生，strategy_inspector node 消费。
#
# 决策三态：
#   - "promote"  → 候选模型晋升为新的 champion
#   - "rollback" → 候选模型性能退化，回退到上一个 checkpoint
#   - "prune"    → 剪枝，换一个 DAG 父节点重新探索
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InspectionPolicyDecision:
    decision: str                       # promote / rollback / prune
    reason: str                         # 决策理由（一句话说明）
    confidence: float                   # 置信度 0~1
    should_store_to_replay_buffer: bool # 是否将本轮题目存入回放池
    should_update_checkpoint: bool      # 是否更新 checkpoint
    should_terminate: bool              # 是否终止整个训练循环
    achieved_target: bool               # 是否达到目标（probe 准确率达标）
    budget_exhausted: bool              # 是否预算耗尽（达到 MAX_ROUNDS）
    termination_reason: str             # 终止原因；非终止 prune 时留空（target_achieved / budget_exhausted）


# ── DAGPolicyResult ──────────────────────────────────────────────────────────
# DAG 更新策略结果。
# 由 update_search_dag_policy() 产生，strategy_inspector node 消费。
# 包含更新后的完整 DAG（节点+边），以及下一轮应该从哪个父节点分支。
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DAGPolicyResult:
    nodes: list[dict]                   # 更新后的所有 DAG 节点
    edges: list[dict]                   # 更新后的所有 DAG 边
    new_node_id: str                    # 本轮新创建的节点 ID
    next_parent_node_id: str            # 下一轮的分支父节点 ID（UCB 选出）
    action_type: str                    # 本轮 action 类型
    action_summary: str                 # 本轮 action 摘要


# ── TeacherSearchPolicy ──────────────────────────────────────────────────────
# 教师搜索策略结果。
# 由 decide_teacher_search() 产生，teacher node 消费（作为 LLM 的 fallback）。
# 决定：下一轮搜什么关键词、聚焦什么难度、用什么采样计划。
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TeacherSearchPolicy:
    search_query: str                   # HF 搜索用的英文查询词
    target_difficulty: str              # 目标难度（easy / medium / hard / ""）
    decision_summary: str               # 决策摘要
    sampling_plan: dict                 # 采样计划（difficulty_weights + module_weights）

    @property
    def target_bucket(self) -> str:
        """target_bucket 是 target_difficulty 的别名，用于兼容旧接口。"""
        return self.target_difficulty


# ── EvaluationGatePolicy ─────────────────────────────────────────────────────
# 评估 Gate 判定结果。
# 由 decide_evaluation_gates() 产生，evaluator node 和 strategy_inspector 消费。
# 判断候选模型是否通过了四道 gate。
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvaluationGatePolicy:
    pass_old_skill_gate: bool           # 旧能力保持 gate（遗忘是否在容忍范围内）
    pass_new_skill_gate: bool           # 新能力提升 gate（候选相对训练前是否达到最小增益）
    pass_probe_gate: bool               # 探针准确率 gate（是否超过阈值）
    pass_frozen_gate: bool              # frozen probe gate（是否退化超过容忍度）
    should_stop: bool                   # 是否应该停止（promote + probe 达标）
    should_promote_candidate: bool      # 是否应该晋升候选模型
    probe_acc_for_gate: float = 0.0     # 实际用于 gate 的 probe 信号（低准确率时可用 multi-shot）
    frozen_degrade_tolerance_used: float = FROZEN_DEGRADE_TOLERANCE # 本轮 frozen 容忍度


EARLY_PROBE_ACC_THRESHOLD = 0.40
LOW_PROBE_TOLERANCE_WEIGHT = 0.05
PROVISIONAL_NEW_ABILITY_DROP_TOLERANCE = 0.10
PROVISIONAL_SMALL_SAMPLE_MAX_NEW_ABILITY_EVAL_COUNT = 50
PROVISIONAL_LARGE_SAMPLE_DROP_TOLERANCE = 0.02
PROMISING_OLD_ABILITY_GAIN = 0.05
PROMISING_COTEST_GAIN = 0.10
PROMISING_COTEST_ACC = 0.55
PROMISING_PROBE_ACC = 0.30
PROMISING_RELATIVE_PROBE_GAIN = 0.03
KEEP_BRANCH_MIN_RAW_REWARD = 0.24
INSIGNIFICANT_DROP_Z = 1.64
PRUNE_TREND_PROBE_DELTA = 0.03
PRUNE_TREND_COTEST_DELTA = 0.10
PRUNE_TREND_REWARD_DELTA = 0.03
ADVANCE_DECISIONS = {"promote", "provisional_promote", "keep_branch"}
INSPECTION_DECISIONS = {*ADVANCE_DECISIONS, "rollback", "prune"}
BRANCH_RETENTION_DECISIONS = {"provisional_promote", "keep_branch"}


# ── ParameterMasterDecision ──────────────────────────────────────────────────
# 参数大师决策结果。
# 由 decide_parameter_master_action() 产生，parameter_master node 消费。
# 决定：选哪个训练 action、从哪个 DAG 节点分支、用哪个数据窗口、多少回放比例。
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ParameterMasterDecision:
    action_key: str                     # action 名称（如 "lora_conservative"）
    action_type: str                    # action 类型（当前固定 "mcts_action"）
    decision_summary: str               # 决策摘要
    training_hyperparams: dict          # 训练超参数（lr, epochs, batch_size 等）
    replay_sample_ratio: float          # 回放池采样比例（0~1）
    branch_parent_node_id: str          # DAG 分支父节点 ID
    action_metadata: dict               # action 元数据（含窗口偏移、MCTS 统计等）


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  一、评估 Gate 判定 — decide_evaluation_gates()                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 这是系统最关键的"安检"环节：评估完成后，判定候选模型是否通过了所有质量门。
#
# 四道 Gate：
#   pass_frozen_gate:    候选模型在 frozen probe 上的错误率不能比 champion 差
#                        超过 FROZEN_DEGRADE_TOLERANCE。这是最硬的门 ——
#                        frozen probe 退化意味着模型丢了之前掌握的能力。
#   pass_old_skill_gate: 旧能力测试集的遗忘幅度不能超过 TEST_FORGETTING_TOLERANCE。
#                        但如果旧能力评估样本 < 2，则跳过此检查（样本太少无统计意义）。
#   pass_new_skill_gate: 晋升收益 gate，候选在 frozen/old/new 三项相对收益之和必须达标。
#   pass_probe_gate:     probe 准确率是否超过 EVAL_PROBE_ACC_GATE。
#
# 晋升条件：pass_frozen_gate AND pass_old_skill_gate AND pass_new_skill_gate 同时满足。
# 停止条件：晋升 + probe 准确率达到目标阈值。
# ════════════════════════════════════════════════════════════════════════════════


def decide_evaluation_gates(
    old_error_rate: float,                      # 旧能力测试的错误率
    champion_old_error_rate: float | None,      # champion 模型在旧能力上的错误率
    new_skill_acc_before: float,                # 训练前新能力准确率
    new_skill_acc_after: float,                 # 训练后新能力准确率
    cotest_acc_before: float,                   # 训练前 cotest 准确率
    cotest_acc_after: float,                    # 训练后 cotest 准确率
    probe_acc_after: float,                     # 训练后 probe 准确率
    champion_probe_error_rate: float | None = None, # champion 在 frozen probe 上的错误率
    old_skill_eval_count: int = 0,              # 旧能力评估样本数（< 2 时跳过 old_skill_gate）
    base_frozen_error_rate: float | None = None,# base 模型在 frozen probe 上的错误率（冷启动 fallback）
    old_ability_acc_before: float = 0.0,        # 训练前旧能力准确率
    old_ability_acc_after: float = 0.0,         # 训练后旧能力准确率
    forgetting_delta: float = 0.0,              # 遗忘幅度
    new_ability_acc_before: float | None = None,# 训练前真实新能力准确率（若有 role split）
    new_ability_acc_after: float | None = None, # 训练后真实新能力准确率（若有 role split）
    probe_acc_for_gate: float | None = None,    # 可选：低准确率时使用 multi-shot probe 信号
) -> EvaluationGatePolicy:
    """Gate decision: probe_frozen degradation + old_ability forgetting.

    Promote when candidate does NOT degrade frozen_probe beyond tolerance
    AND does NOT trigger excessive forgetting on old_ability test questions.
    """
    effective_probe_acc = probe_acc_after
    used_probe_gate_override = False
    if (
        probe_acc_for_gate is not None
        and probe_acc_after < EARLY_PROBE_ACC_THRESHOLD
    ):
        effective_probe_acc = max(probe_acc_after, probe_acc_for_gate)
        used_probe_gate_override = effective_probe_acc > probe_acc_after

    adaptive_frozen_tolerance = FROZEN_DEGRADE_TOLERANCE
    if used_probe_gate_override and effective_probe_acc < EARLY_PROBE_ACC_THRESHOLD:
        adaptive_frozen_tolerance = min(
            0.15,
            FROZEN_DEGRADE_TOLERANCE
            + LOW_PROBE_TOLERANCE_WEIGHT * (1.0 - effective_probe_acc),
        )

    # 候选模型在 frozen probe 上的错误率
    candidate_probe_error_rate = 1.0 - effective_probe_acc

    # frozen gate: 候选 vs champion 对比
    # 如果 champion 的错误率未知，用 base 模型兜底（首次运行时 champion 还不存在）
    pass_frozen_gate = (
        champion_probe_error_rate is None
        or (candidate_probe_error_rate - champion_probe_error_rate) <= adaptive_frozen_tolerance
    )

    # 冷启动 fallback：如果 champion 没有 frozen probe 数据，用 base 模型代替
    if base_frozen_error_rate is not None and champion_probe_error_rate is None:
        pass_frozen_gate = (
            (candidate_probe_error_rate - base_frozen_error_rate) <= adaptive_frozen_tolerance
        )

    # old_skill gate: 样本太少不检查，否则遗忘必须在容忍范围内
    pass_old_skill_gate = (
        old_skill_eval_count < 2
        or forgetting_delta <= TEST_FORGETTING_TOLERANCE
    )
    # new_skill gate: 优先使用 role split 后的真实新能力，而不是混合 old/new 的总 test acc。
    effective_new_before = new_skill_acc_before
    effective_new_after = new_skill_acc_after
    if new_ability_acc_before is not None and new_ability_acc_after is not None:
        effective_new_before = new_ability_acc_before
        effective_new_after = new_ability_acc_after

    new_skill_gain = effective_new_after - effective_new_before
    old_skill_gain = old_ability_acc_after - old_ability_acc_before
    frozen_probe_gain = 0.0
    if champion_probe_error_rate is not None:
        frozen_probe_gain = champion_probe_error_rate - candidate_probe_error_rate
    elif base_frozen_error_rate is not None:
        frozen_probe_gain = base_frozen_error_rate - candidate_probe_error_rate
    total_relative_gain = frozen_probe_gain + old_skill_gain + new_skill_gain
    pass_new_skill_gate = total_relative_gain > PROMOTION_TOTAL_RELATIVE_GAIN

    # 晋升条件 = safety gates 通过 + 新能力确有提升。
    # 旧能力恢复 / cotest 改善只进入 provisional_promote / keep_branch，避免直接替换 champion。
    should_promote_candidate = pass_frozen_gate and pass_old_skill_gate and pass_new_skill_gate
    # 停止条件 = 可以晋升 + probe 准确率已经达标
    should_stop = should_promote_candidate and effective_probe_acc >= EVAL_PROBE_ACC_GATE

    return EvaluationGatePolicy(
        pass_old_skill_gate=pass_old_skill_gate,
        pass_new_skill_gate=pass_new_skill_gate,
        pass_probe_gate=probe_acc_after >= EVAL_PROBE_ACC_GATE,
        pass_frozen_gate=pass_frozen_gate,
        should_stop=should_stop,
        should_promote_candidate=should_promote_candidate,
        probe_acc_for_gate=effective_probe_acc,
        frozen_degrade_tolerance_used=adaptive_frozen_tolerance,
    )


# ── 一.5、Gate Controller Agent ───────────────────────────────────────────────
# 门控动态调整方案。
# 消费方：evaluator node 在评估完成后调用此函数，决定是否调整门控阈值。
# 返回的 gate_adjustment 会写入 evaluator 的返回消息，供 MCTS 边查询。
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GateAdjustmentResult:
    adjust: bool                            # 是否调整门控
    frozen_degrade_tolerance: float         # frozen probe 退化容忍度
    test_forgetting_tolerance: float        # 遗忘容忍度
    eval_probe_acc_gate: float              # 探针准确率门控
    reason: str                             # 调整原因


def decide_gate_adjustment(
    round_id: int,                          # 当前轮次
    rollback_streak: int,                   # 连续回退次数
    per_difficulty_acc: dict,               # 各难度探针准确率
    old_skill_gate_passed: bool,            # 上一轮旧能力门控是否通过
    frozen_gate_passed: bool,               # 上一轮 frozen gate 是否通过
    should_promote: bool,                   # 上一轮是否应晋升
    data_pressure: dict | None = None,      # 数据压力信号
) -> GateAdjustmentResult:
    """确定性门控调整 fallback。

    提供保守的默认行为：正常情况下不调整，只在明确需要时才调整。
    LLM agent 可以通过 decide_json() 覆盖此结果。
    """
    data = data_pressure or {}

    # 默认：不调整
    adjust = False
    frozen_tol = FROZEN_DEGRADE_TOLERANCE
    forget_tol = TEST_FORGETTING_TOLERANCE
    probe_gate = EVAL_PROBE_ACC_GATE
    reason = "no adjustment needed"

    # 规则 1: 连续回退 >= 3 → 放宽门控
    if rollback_streak >= 3:
        adjust = True
        frozen_tol = min(0.15, FROZEN_DEGRADE_TOLERANCE + 0.03)
        forget_tol = min(0.20, TEST_FORGETTING_TOLERANCE + 0.05)
        reason = f"rollback_streak={rollback_streak}: relax gates to allow promotion"

    # 规则 2: 连续 promote → 可以考虑收紧门控（提高标准）
    elif should_promote and rollback_streak == 0 and round_id > 2:
        # 如果各难度准确率都高，适当收紧
        acc_values = [v for k, v in per_difficulty_acc.items() if k in ("easy", "medium", "hard")]
        if acc_values and sum(acc_values) / len(acc_values) > 0.80:
            adjust = True
            probe_gate = min(0.95, EVAL_PROBE_ACC_GATE + 0.05)
            reason = "high accuracy across difficulties: tighten probe gate"

    return GateAdjustmentResult(
        adjust=adjust,
        frozen_degrade_tolerance=frozen_tol,
        test_forgetting_tolerance=forget_tol,
        eval_probe_acc_gate=probe_gate,
        reason=reason,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  二、教师搜索策略 — decide_teacher_search() + 辅助函数                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 教师的核心职责：决定下一轮要搜什么数据、练什么难度。
# 这套确定性函数是 teacher node 中 LLM 决策的 fallback。
# ════════════════════════════════════════════════════════════════════════════════


# ── pick_weakest_bucket ──────────────────────────────────────────────────────
# 从各难度的探针准确率中选出准确率最低的难度档。
# 训练策略是"哪里弱练哪里" —— 永远聚焦模型当前最差的难度。
# ──────────────────────────────────────────────────────────────────────────────
def pick_weakest_bucket(metrics_after: dict, default_bucket: str) -> str:
    # 优先用 frozen probe 的分难度准确率（更稳定），其次用普通 probe 的
    per_difficulty_acc = metrics_after.get("per_difficulty_acc_frozen") or metrics_after.get("per_difficulty_acc", {})
    valid_difficulties = {
        k: v for k, v in per_difficulty_acc.items()
        if k in ("easy", "medium", "hard")
    }
    if valid_difficulties:
        return min(valid_difficulties, key=lambda k: valid_difficulties[k])
    return default_bucket if default_bucket in ("easy", "medium", "hard", "unknown") else "hard"


# ── _normalize_weights ───────────────────────────────────────────────────────
# 权重归一化工具函数。
# 将任意正数权重归一化为总和 = 1 的分布，自动去除负数和非正条目。
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items() if v > 0}


def _clamp_probability_distribution(
    weights: dict[str, float],
    minimum: float = 0.05,
    maximum: float = 0.80,
) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        try:
            cleaned[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            cleaned[key] = 0.0
    if not cleaned:
        return {}

    bucket_count = len(cleaned)
    min_bound = max(0.0, min(float(minimum), 1.0))
    max_bound = max(min_bound, min(float(maximum), 1.0))
    if bucket_count * min_bound > 1.0:
        min_bound = 1.0 / bucket_count
    if bucket_count * max_bound < 1.0:
        max_bound = 1.0 / bucket_count

    allocation = {key: min_bound for key in cleaned}
    remaining = max(0.0, 1.0 - min_bound * bucket_count)
    free_keys = set(cleaned)
    while free_keys and remaining > 1e-12:
        raw_total = sum(cleaned[key] for key in free_keys)
        if raw_total <= 0.0:
            shares = {key: remaining / len(free_keys) for key in free_keys}
        else:
            shares = {key: remaining * cleaned[key] / raw_total for key in free_keys}

        capped = [
            key for key, share in shares.items()
            if allocation[key] + share > max_bound
        ]
        if not capped:
            for key, share in shares.items():
                allocation[key] += share
            remaining = 0.0
            break

        for key in capped:
            remaining -= max(0.0, max_bound - allocation[key])
            allocation[key] = max_bound
            free_keys.remove(key)

    total = sum(allocation.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in allocation.items()}


def difficulty_weights_from_accuracy(
    per_difficulty_acc: dict[str, float] | None,
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert evaluator per-difficulty accuracy into sampling weights.

    Low-accuracy buckets receive more probability mass, but min/max clamping
    prevents starvation and single-bucket collapse.
    """

    fallback_weights = fallback or {"easy": 0.15, "medium": 0.70, "hard": 0.15}
    if not isinstance(per_difficulty_acc, dict) or not per_difficulty_acc:
        return _clamp_probability_distribution(fallback_weights)
    valid_acc: dict[str, float] = {}
    for bucket in ("easy", "medium", "hard"):
        if bucket not in per_difficulty_acc:
            continue
        try:
            valid_acc[bucket] = min(max(float(per_difficulty_acc[bucket]), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
    if not valid_acc:
        return _clamp_probability_distribution(fallback_weights)

    deficits: dict[str, float] = {}
    for bucket in ("easy", "medium", "hard"):
        if bucket in valid_acc:
            deficits[bucket] = max(0.0, 1.0 - valid_acc[bucket])
        else:
            deficits[bucket] = max(0.0, float(fallback_weights.get(bucket, 0.0)))
    if not any(value > 0.0 for value in deficits.values()):
        return _clamp_probability_distribution(fallback_weights)
    return _clamp_probability_distribution(deficits)


# ── build_sampling_plan ──────────────────────────────────────────────────────
# 组装采样计划。
# 采样计划定义了 filter 如何从候选题目池中抽样：
#   - primary_axis:   主抽样维度（当前固定为 dynamic_difficulty）
#   - secondary_axis: 次抽样维度（module，如几何/代数分类）
#   - difficulty_weights: easy/medium/hard 的抽样比例
#   - module_weights:      各模块的抽样比例
#   - target_difficulty:   目标难度档
# ──────────────────────────────────────────────────────────────────────────────
def build_sampling_plan(
    target_difficulty: str = "",
    focus_modules: dict[str, int] | None = None,
    difficulty_weights: dict[str, float] | None = None,
    source: str = "teacher_policy",
) -> dict:
    # 默认难度配比：主攻 medium（70%），easy 和 hard 各 15%
    selected_difficulty_weights = difficulty_weights or {
        "easy": 0.15,
        "medium": 0.70,
        "hard": 0.15,
    }

    module_weights = {}
    if focus_modules:
        module_weights = _normalize_weights(
            {
                module: float(count)
                for module, count in focus_modules.items()
                if module and module != "unknown"
            }
        )

    return {
        "version": 1,
        "source": source,
        "primary_axis": "dynamic_difficulty",
        "secondary_axis": "module",
        "difficulty_weights": _normalize_weights(selected_difficulty_weights),
        "module_weights": module_weights,
        "target_difficulty": target_difficulty,
    }


# ── build_curriculum_sampling_plan ───────────────────────────────────────────
# 课程学习式的采样计划。
# 根据 probe 各难度的当前准确率，自适应调整训练配比：
#   - medium 准确率 < 30%  → 主攻 medium（60%），少量 easy 巩固基础
#   - medium 30~70%        → medium 70%，开始加 hard
#   - hard < 30%           → 加大 hard 比例到 30%
#   - 都还行               → easy/medium/hard = 10/45/45，均衡练习
# 这种设计的直觉：像人类学习一样，先打好基础再上难度。
# ──────────────────────────────────────────────────────────────────────────────
def build_curriculum_sampling_plan(
    probe_easy_acc: float | None = None,
    probe_medium_acc: float | None = None,
    probe_hard_acc: float | None = None,
    focus_modules: dict[str, int] | None = None,
) -> dict:
    weights = difficulty_weights_from_accuracy(
        {
            "easy": 0.0 if probe_easy_acc is None else float(probe_easy_acc),
            "medium": 0.0 if probe_medium_acc is None else float(probe_medium_acc),
            "hard": 0.0 if probe_hard_acc is None else float(probe_hard_acc),
        }
    )

    return build_sampling_plan(
        focus_modules=focus_modules,
        difficulty_weights=weights,
        source="parameter_master_curriculum",
    )


def _difficulty_search_query(goal: str, target_difficulty: str) -> str:
    """Build a content-domain search query from the user's goal.

    The system must not hard-code a specific subject such as math.  Difficulty
    words are policy metadata; the searcher attaches them only after it has a
    content-domain anchor.
    """
    _ = target_difficulty
    return fallback_query_from_goal(goal)


# ── decide_teacher_search ────────────────────────────────────────────────────
# 确定性教师搜索策略（fallback）。
# 当 USE_LLM_AGENTS=False 或 LLM 解析失败时使用。
# 注意：此函数仅提供 fallback 行为，真正的搜索策略由 LLM agent 决定。
# 首轮直接使用原始 goal（不做任何翻译），后续轮次由 LLM 根据上下文生成。
#
# 四种 fallback 决策分支：
#   round=0：     使用原始 user goal 作为搜索词
#   hard 比例过高：降低 hard 权重，允许 merge/replace
#   训练池太小：   merge shards 扩充数据量
#   正常模式：     主攻准确率最低的难度档
# ──────────────────────────────────────────────────────────────────────────────
def decide_teacher_search(
    goal: str,                                  # 用户目标（原始文本，不翻译）
    round_id: int,                              # 当前轮次
    metrics_after: dict,                        # 上轮评估后的探针分难度准确率
    default_bucket: str,                        # 默认难度桶
    round_data_stats: dict | None = None,       # 本轮数据统计（hard_ratio, train_count 等）
) -> TeacherSearchPolicy:
    # ── 首轮：保留原始 goal，由 searcher agent 负责改写为 HF 查询 ──
    if round_id <= 0:
        return TeacherSearchPolicy(
            search_query=goal.strip(),
            target_difficulty="",
            decision_summary=f"initial semantic request from user goal",
            sampling_plan=build_sampling_plan(""),
        )

    # ── 数据压力检测 ──
    data_stats = round_data_stats if isinstance(round_data_stats, dict) else {}
    hard_ratio = float(data_stats.get("hard_ratio", 0.0) or 0.0)
    low_train_signal = bool(data_stats.get("low_train_signal", False))
    hard_dominated_signal = bool(data_stats.get("hard_dominated_signal", False)) or hard_ratio >= DATA_HARD_RATIO_MERGE_THRESHOLD

    # 情况 1: hard 题目占比过高 → 降低 hard 权重，允许 merge/replace
    if hard_dominated_signal:
        plan = build_sampling_plan(
            "medium",
            difficulty_weights={"easy": 0.35, "medium": 0.55, "hard": 0.10},
            source="teacher_data_pressure",
        )
        plan["dataset_policy_hint"] = "merge_or_replace"
        plan["data_pressure"] = {
            "hard_ratio": hard_ratio,
            "train_count": data_stats.get("train_count", 0),
            "reason": "hard_dominated",
        }
        return TeacherSearchPolicy(
            search_query=_difficulty_search_query(goal, "medium"),
            target_difficulty="medium",
            decision_summary=(
                f"hard_ratio={hard_ratio:.3f}; search easier/medium data and allow merge/replace"
            ),
            sampling_plan=plan,
        )

    # 情况 2: 训练池题量不足 → merge shards 扩充
    if low_train_signal:
        plan = build_sampling_plan(
            "medium",
            difficulty_weights={"easy": 0.25, "medium": 0.60, "hard": 0.15},
            source="teacher_data_pressure",
        )
        plan["dataset_policy_hint"] = "merge_shards"
        plan["data_pressure"] = {
            "hard_ratio": hard_ratio,
            "train_count": data_stats.get("train_count", 0),
            "reason": "low_train_count",
        }
        return TeacherSearchPolicy(
            search_query=_difficulty_search_query(goal, "medium"),
            target_difficulty="medium",
            decision_summary="training pool too small; search broader data and merge shards",
            sampling_plan=plan,
        )

    # 情况 3 (正常): 找出最弱难度档，主攻它
    target_difficulty = pick_weakest_bucket(metrics_after, default_bucket)
    evaluator_weights = difficulty_weights_from_accuracy(
        metrics_after.get("per_difficulty_acc_frozen") or metrics_after.get("per_difficulty_acc") or {},
    )
    return TeacherSearchPolicy(
        search_query=goal.strip(),
        target_difficulty=target_difficulty,
        decision_summary=f"focus on {target_difficulty} difficulty",
        sampling_plan=build_sampling_plan(
            target_difficulty,
            difficulty_weights=evaluator_weights,
            source="teacher_evaluator_accuracy",
        ),
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  三、候选模型评分 — score_candidate_components() / score_candidate()        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 评分函数用于 MCTS DAG 的 value backpropagation（回传 reward 值到搜索树）。
#
# 核心指标权重分配：
#   0.45 × frozen_probe_acc   ← 最重要的指标：frozen probe 上的泛化能力
#   0.25 × new_skill_acc       ← 新能力测试集的准确率
#   0.15 × cotest_acc          ← cotest（补考）准确率
#   0.10 × new_gain            ← 新能力提升幅度
#   0.05 × cotest_gain         ← cotest 提升幅度
#
# 额外的奖励加成（bonus）：
#   + REWARD_GATE_BONUS        ← 通过了所有 gate
#   + REWARD_STOP_BONUS        ← 达到了停止条件（probe 准确率达标）
# ════════════════════════════════════════════════════════════════════════════════


def score_candidate_components(metrics: EvalResultPayload) -> dict[str, float]:
    """Reward focused on frozen probe accuracy and test gains."""
    frozen_acc = metrics.probe_acc_frozen
    if frozen_acc is None:
        frozen_acc = metrics.probe_acc_after
    new_gain = metrics.new_skill_acc_after - metrics.new_skill_acc_before
    cotest_gain = metrics.cotest_acc_after - metrics.cotest_acc_before
    gate_bonus = REWARD_GATE_BONUS if metrics.should_promote_candidate else 0.0
    stop_bonus = REWARD_STOP_BONUS if metrics.should_stop else 0.0
    raw_reward = (
        0.45 * frozen_acc
        + 0.25 * metrics.new_skill_acc_after
        + 0.15 * metrics.cotest_acc_after
        + 0.10 * new_gain
        + 0.05 * cotest_gain
    )
    reward = raw_reward + gate_bonus + stop_bonus
    return {
        "probe_reward": float(frozen_acc),
        "new_skill_acc_after": float(metrics.new_skill_acc_after),
        "cotest_acc_after": float(metrics.cotest_acc_after),
        "new_gain": float(new_gain),
        "cotest_gain": float(cotest_gain),
        "old_skill_penalty": 0.0,
        "forgetting_penalty": 0.0,
        "gate_bonus": float(gate_bonus),
        "stop_bonus": float(stop_bonus),
        "raw_reward": float(raw_reward),
        "reward": float(reward),
    }


def score_candidate(metrics: EvalResultPayload) -> float:
    """Policy reward used by bandit/MCTS-style DAG value backprop."""
    return score_candidate_components(metrics)["reward"]


def _ability_delta(metrics: EvalResultPayload, before_key: str, after_key: str) -> float:
    before = getattr(metrics, before_key)
    after = getattr(metrics, after_key)
    return float(after - before)


def _probe_signal(metrics: EvalResultPayload) -> float:
    return float(metrics.probe_acc_frozen if metrics.probe_acc_frozen is not None else metrics.probe_acc_after)


def _new_ability_eval_count(metrics: EvalResultPayload) -> int:
    count = int(getattr(metrics, "new_ability_eval_count", 0) or 0)
    if count > 0:
        return count
    return int(getattr(metrics, "new_skill_eval_count", 0) or 0)


def _new_ability_drop_is_statistically_small(metrics: EvalResultPayload, before: float, after: float) -> bool:
    before_count = _new_ability_eval_count(metrics)
    after_count = before_count
    if before_count <= 1 or after_count <= 1:
        return False
    drop = before - after
    if drop <= 0:
        return True
    before = max(0.0, min(1.0, float(before)))
    after = max(0.0, min(1.0, float(after)))
    standard_error = math.sqrt(
        before * (1.0 - before) / before_count
        + after * (1.0 - after) / after_count
    )
    return standard_error > 0 and drop <= INSIGNIFICANT_DROP_Z * standard_error


def _new_ability_is_stable_enough_for_provisional(
    metrics: EvalResultPayload,
    before: float,
    after: float,
) -> bool:
    gain = after - before
    if gain >= -PROVISIONAL_LARGE_SAMPLE_DROP_TOLERANCE:
        return True

    eval_count = _new_ability_eval_count(metrics)
    if (
        0 < eval_count <= PROVISIONAL_SMALL_SAMPLE_MAX_NEW_ABILITY_EVAL_COUNT
        and gain >= -PROVISIONAL_NEW_ABILITY_DROP_TOLERANCE
    ):
        return True

    return _new_ability_drop_is_statistically_small(metrics, before, after)


def _is_mixed_but_promising(metrics: EvalResultPayload) -> bool:
    """Soft exploration signal for candidates worth keeping without hard promotion."""
    if not (metrics.pass_frozen_gate and metrics.pass_old_skill_gate):
        return False
    new_ability_gain = _ability_delta(metrics, "new_ability_acc_before", "new_ability_acc_after")
    new_before = metrics.new_ability_acc_before
    new_after = metrics.new_ability_acc_after
    if metrics.new_ability_eval_count <= 0:
        new_before = metrics.new_skill_acc_before
        new_after = metrics.new_skill_acc_after
        new_ability_gain = new_after - new_before
    if not _new_ability_is_stable_enough_for_provisional(metrics, new_before, new_after):
        return False

    old_ability_gain = _ability_delta(metrics, "old_ability_acc_before", "old_ability_acc_after")
    cotest_gain = metrics.cotest_acc_after - metrics.cotest_acc_before
    raw_reward = score_candidate_components(metrics)["raw_reward"]
    probe_acc = _probe_signal(metrics)
    champion_probe_acc = getattr(metrics, "probe_acc_champion", None)
    relative_probe_gain = None
    if champion_probe_acc is not None:
        relative_probe_gain = probe_acc - float(champion_probe_acc)
    supporting_signal = (
        old_ability_gain >= PROMISING_OLD_ABILITY_GAIN
        or metrics.forgetting_delta < 0.0
        or cotest_gain >= PROMISING_COTEST_GAIN
        or metrics.cotest_acc_after >= PROMISING_COTEST_ACC
        or probe_acc >= PROMISING_PROBE_ACC
        or (relative_probe_gain is not None and relative_probe_gain >= PROMISING_RELATIVE_PROBE_GAIN)
        or raw_reward >= KEEP_BRANCH_MIN_RAW_REWARD
    )
    return supporting_signal


def _has_recent_positive_trend(metrics: EvalResultPayload, edges_tail: list[dict] | None = None) -> bool:
    """Avoid pruning branches that are still improving on probe/cotest/reward."""
    previous: list[dict] = []
    for edge in edges_tail or []:
        if not isinstance(edge, dict):
            continue
        metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
        edge_metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
        previous.append({"edge": edge, "metadata": metadata, "metrics": edge_metrics})
    if not previous:
        return _is_mixed_but_promising(metrics)

    previous_probe_values = [
        item["metrics"].get("probe_acc_frozen", item["metrics"].get("probe_acc_after"))
        for item in previous
    ]
    previous_cotest_values = [item["metrics"].get("cotest_acc_after") for item in previous]
    previous_reward_values = []
    for item in previous:
        reward_components = item["metadata"].get("reward_components")
        if isinstance(reward_components, dict) and reward_components.get("raw_reward") is not None:
            previous_reward_values.append(reward_components.get("raw_reward"))
        elif item["edge"].get("reward") is not None:
            previous_reward_values.append(item["edge"].get("reward"))

    def _max_numeric(values: list[Any]) -> float | None:
        cleaned: list[float] = []
        for value in values:
            try:
                if value is not None:
                    cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        return max(cleaned) if cleaned else None

    current_probe = _probe_signal(metrics)
    current_cotest = float(metrics.cotest_acc_after)
    current_reward = score_candidate_components(metrics)["raw_reward"]
    previous_probe = _max_numeric(previous_probe_values)
    previous_cotest = _max_numeric(previous_cotest_values)
    previous_reward = _max_numeric(previous_reward_values)

    return (
        _is_mixed_but_promising(metrics)
        or (previous_probe is not None and current_probe >= previous_probe + PRUNE_TREND_PROBE_DELTA)
        or (previous_cotest is not None and current_cotest >= previous_cotest + PRUNE_TREND_COTEST_DELTA)
        or (previous_reward is not None and current_reward >= previous_reward + PRUNE_TREND_REWARD_DELTA)
    )


def inspection_gate_blocker_reason(metrics: EvalResultPayload) -> str:
    """Return a concise, accurate reason for a non-promoted candidate."""
    failed: list[str] = []
    if not metrics.pass_frozen_gate:
        failed.append(f"frozen probe gate (probe={_probe_signal(metrics):.3f})")
    if not metrics.pass_old_skill_gate:
        failed.append(f"old skill gate (forgetting_delta={metrics.forgetting_delta:+.3f})")
    if not metrics.pass_new_skill_gate:
        new_gain = _ability_delta(metrics, "new_ability_acc_before", "new_ability_acc_after")
        if new_gain == 0.0:
            new_gain = _ability_delta(metrics, "new_skill_acc_before", "new_skill_acc_after")
        failed.append(f"new skill gate (gain={new_gain:+.3f})")
    if not failed:
        return "candidate failed promotion quality gate"
    return "candidate failed " + ", ".join(failed)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  四、策略巡检决策 — decide_inspection()                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Agent 决策为主，gate 仅为 fallback 安全底线。
# 消费方：strategy_inspector node 调用此函数获取确定性 fallback，
#         然后通过 LLM agent（INSPECTION_AGENT_PROMPT）覆盖决策。
#
# fallback 决策逻辑（仅当 LLM 不可用时生效）：
#   1. rollback_streak >= 4  → prune（换父节点）
#   2. frozen_gate + old_skill_gate + new_skill_gate 通过 → promote
#   3. 否则 → rollback
# ════════════════════════════════════════════════════════════════════════════════


def decide_inspection(
    metrics: EvalResultPayload,
    round_id: int,
    rollback_streak: int = 0,
    search_dag_edges_tail: list[dict] | None = None,
) -> InspectionPolicyDecision:
    """Promotion policy: hard safety gates plus soft exploration states.

    Rollback is reserved for unsafe regressions. Mixed but promising candidates are
    kept as DAG branches without replacing the champion.
    """
    budget_exhausted = round_id >= MAX_ROUNDS

    if budget_exhausted:
        return InspectionPolicyDecision(
            decision="promote" if metrics.pass_frozen_gate and metrics.pass_old_skill_gate else "rollback",
            reason="round budget exhausted before probe target was reached",
            confidence=0.9,
            should_store_to_replay_buffer=True,
            should_update_checkpoint=bool(metrics.pass_frozen_gate and metrics.pass_old_skill_gate),
            should_terminate=True,
            achieved_target=False,
            budget_exhausted=True,
            termination_reason="budget_exhausted",
        )

    # ── prune：连续回退太多且没有任何改善趋势时，才剪掉当前分支 ──
    if rollback_streak >= 4 and not _has_recent_positive_trend(metrics, search_dag_edges_tail):
        return InspectionPolicyDecision(
            decision="prune",
            reason=f"rollback_streak={rollback_streak}: prune current branch, explore other nodes",
            confidence=1.0,
            should_store_to_replay_buffer=True,
            should_update_checkpoint=False,
            should_terminate=False,
            achieved_target=False,
            budget_exhausted=budget_exhausted,
            termination_reason="",
        )

    # ── promote：候选模型通过质量门 ──
    if metrics.pass_frozen_gate and metrics.pass_old_skill_gate and metrics.pass_new_skill_gate:
        achieved_target = metrics.should_stop
        should_terminate = achieved_target or budget_exhausted
        return InspectionPolicyDecision(
            decision="promote",
            reason=(
                "candidate passes frozen-probe gate"
                if not budget_exhausted or achieved_target
                else "candidate promoted, but round budget exhausted before probe target was reached"
            ),
            confidence=0.9,
            should_store_to_replay_buffer=True,
            should_update_checkpoint=True,
            should_terminate=should_terminate,
            achieved_target=achieved_target,
            budget_exhausted=budget_exhausted,
            termination_reason=(
                "target_achieved" if achieved_target
                else "budget_exhausted" if budget_exhausted
                else ""
            ),
        )

    if _is_mixed_but_promising(metrics):
        return InspectionPolicyDecision(
            decision="provisional_promote",
            reason="candidate is mixed but promising; keep branch without replacing champion",
            confidence=0.75,
            should_store_to_replay_buffer=True,
            should_update_checkpoint=False,
            should_terminate=False,
            achieved_target=False,
            budget_exhausted=budget_exhausted,
            termination_reason="",
        )

    if rollback_streak >= 4:
        return InspectionPolicyDecision(
            decision="keep_branch",
            reason=f"rollback_streak={rollback_streak}: trend check prevents prune; keep branch",
            confidence=0.65,
            should_store_to_replay_buffer=True,
            should_update_checkpoint=False,
            should_terminate=False,
            achieved_target=False,
            budget_exhausted=budget_exhausted,
            termination_reason="",
        )

    # ── rollback：候选模型质量不过关，回退 ──
    return InspectionPolicyDecision(
        decision="rollback",
        reason=inspection_gate_blocker_reason(metrics),
        confidence=0.9,
        should_store_to_replay_buffer=True,
        should_update_checkpoint=False,
        should_terminate=budget_exhausted,
        achieved_target=False,
        budget_exhausted=budget_exhausted,
        termination_reason="budget_exhausted" if budget_exhausted else "",
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  五、MCTS 节点选择 — _ucb_score() / select_next_parent_node()                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# MCTS（蒙特卡洛树搜索）的探索-利用平衡。
# UCB1 公式：score = value + C × sqrt(ln(total_visits) / visits)
#   - value:    该节点的平均 reward
#   - visits:   该节点被访问的次数
#   - C:        探索系数（exploration constant），越大越倾向于探索未访问的节点
#   - 未访问过的节点 score = +∞，保证每个节点至少被探索一次
# ════════════════════════════════════════════════════════════════════════════════


def _ucb_score(node: dict, total_visits: int, exploration: float = 1.4) -> float:
    """UCB1 公式：平衡探索（exploration）与利用（exploitation）。"""
    visits = max(0, int(node.get("visit_count", 0) or 0))
    value = float(node.get("value_estimate", 0.0) or 0.0)
    if visits == 0:
        return float("inf")  # 未访问过的节点优先探索
    return value + exploration * math.sqrt(math.log(max(total_visits, 1) + 1) / visits)


def select_next_parent_node(
    nodes: list[dict],
    fallback_node_id: str,
    candidate_node_ids: list[str] | None = None,
) -> str:
    """选择下一轮的分支父节点，以节点 value 为主，优先高 value 节点。

    训练模型成本高昂，不需要探索所有节点。
    策略：优先从 value_estimate 最高的节点（champion）开始训练，
    适度探索未访问过的节点（上限由 EXPLORATION_C_NODE 控制）。

    从未访问过的节点最多选 1 个，其余按 value 降序选。
    """
    if not nodes:
        return fallback_node_id

    allowed_ids = {str(node_id) for node_id in candidate_node_ids or [] if node_id}
    selectable_nodes = [
        node for node in nodes
        if not allowed_ids or str(node.get("node_id", "")) in allowed_ids
    ]
    if not selectable_nodes:
        selectable_nodes = nodes

    # 分离已访问和未访问节点
    visited = [n for n in selectable_nodes if int(n.get("visit_count", 0) or 0) > 0]
    unvisited = [n for n in selectable_nodes if int(n.get("visit_count", 0) or 0) == 0]

    # 按 value_estimate 降序排列已访问节点
    visited.sort(key=lambda n: float(n.get("value_estimate", 0.0) or 0.0), reverse=True)

    # 如果存在未访问节点且探索系数允许，偶尔选一个未访问的
    if unvisited and visited:
        # 计算 UCB 只用于在 visited 和 unvisited 之间做决择
        total_visits = sum(int(n.get("visit_count", 0) or 0) for n in visited) + 1
        best_visited_score = _ucb_score(visited[0], total_visits, exploration=MCTS_EXPLORATION_C_NODE)
        best_unvisited_score = _ucb_score(unvisited[0], total_visits, exploration=MCTS_EXPLORATION_C_NODE)

        # 如果未访问的 UCB 分数显著更高（>20%），探索一下
        if best_unvisited_score > best_visited_score * 1.2:
            return unvisited[0].get("node_id") or fallback_node_id

    # 默认：返回 value 最高的已访问节点
    if visited:
        return visited[0].get("node_id") or fallback_node_id
    # 只有未访问节点
    if unvisited:
        return unvisited[0].get("node_id") or fallback_node_id
    return fallback_node_id


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  六、Action Space — 训练策略模板库                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 定义了 12 种训练策略模板（action），每种包含完整的超参数组合。
# 分为两大族：
#   - Full Fine-Tuning（前 7 种）：适用于显存充足的场景
#   - LoRA Fine-Tuning（后 5 种）：适用于小显存 / 减少遗忘风险的场景
#
# Action 选择逻辑由 decide_parameter_master_action() 根据 rollback_streak
# 和 DAG 边历史（UCB）自动决定。
# ════════════════════════════════════════════════════════════════════════════════


def _default_candidate_action_space() -> dict[str, dict]:
    """返回 12 种内置训练策略模板的默认定义。"""
    return {
        # ── Full Fine-Tuning 系列 ────────────────────────────────────
        "balanced_default": {           # 均衡默认：中等学习率 + 中度 replay
            "learning_rate": 5.0e-6,
            "num_train_epochs": 3.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.30,
        },
        "conservative_anti_forget": {   # 保守抗遗忘：低学习率 + 高 replay + 更长 warmup
            "learning_rate": 2.5e-6,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 12,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.45,
        },
        "ultra_conservative_replay": {  # 超保守 replay：极低学习率 + 80% replay
            "learning_rate": 1.0e-6,
            "num_train_epochs": 1.0,
            "gradient_accumulation_steps": 16,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.08,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.80,
        },
        "low_lr_more_steps": {          # 低学习率多步：慢速稳定训练
            "learning_rate": 3.0e-6,
            "num_train_epochs": 3.0,
            "gradient_accumulation_steps": 10,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.35,
        },
        "faster_explore": {             # 快速探索：高学习率 + 低 replay，快速迭代
            "learning_rate": 8.0e-6,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.25,
        },
        "shift_window_balanced": {      # 切换数据窗口 + 均衡：默认参数但低 replay
            "learning_rate": 5.0e-6,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.20,
        },
        "probe_failure_refresh": {      # probe 失败刷新：与 conservative 相同参数
            "learning_rate": 2.5e-6,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 12,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.45,
        },
        # ── LoRA Fine-Tuning 系列 ────────────────────────────────────
        # LoRA 优势：参数量少，遗忘风险低，适合 0.6B 模型的全量微调不稳定问题
        "lora_conservative": {          # LoRA 保守：rank=8, alpha=16, 中等 replay
            "learning_rate": 1.0e-4,
            "num_train_epochs": 3.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.30,
            "finetuning_type": "lora",
            "lora_rank": 8,
            "lora_alpha": 16,
        },
        "lora_explore": {               # LoRA 探索：高 rank + 高学习率，低 replay
            "learning_rate": 3.0e-4,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.20,
            "finetuning_type": "lora",
            "lora_rank": 16,
            "lora_alpha": 32,
        },
        "lora_ultra_safe": {            # LoRa 超安全：低学习率 + 高 replay + 低 rank
            "learning_rate": 5.0e-5,
            "num_train_epochs": 2.0,
            "gradient_accumulation_steps": 12,
            "per_device_train_batch_size": 1,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "replay_sample_ratio": 0.45,
            "finetuning_type": "lora",
            "lora_rank": 8,
            "lora_alpha": 16,
        },
    }


def _candidate_action_space() -> dict[str, dict]:
    """加载 action space，优先从外部 JSON 配置文件合并。

    外部配置（MCTS_ACTION_SPACE_PATH）可以覆盖/新增 action 模板，
    允许用户在不修改代码的情况下调整训练策略。
    """
    action_space = _default_candidate_action_space()
    config_path = Path(MCTS_ACTION_SPACE_PATH)
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded:
                for key, value in loaded.items():
                    if not isinstance(value, dict):
                        continue
                    key = str(key)
                    # 外部配置与内置默认合并：外部值覆盖内置同名 key
                    merged = dict(action_space.get(key, {}))
                    merged.update(value)
                    action_space[key] = merged
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[strategy_policy] Could not load MCTS action space {config_path}: {exc}")
    return action_space


def get_parameter_action_space() -> dict[str, dict]:
    """公开接口：返回 deep copy 的 action space。"""
    return {key: dict(value) for key, value in _candidate_action_space().items()}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  六.5、Action Selection Agent fallback                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 确定性 fallback：根据 rollback_streak 和 edge_summary 选 action + 变异。
# LLM agent（ACTION_SELECTION_AGENT_PROMPT）通过 decide_json() 覆盖。
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ActionSelectionResult:
    action_key: str             # 选中的动作模板名
    lr_scale: float             # 学习率缩放 (0.9|1.0|1.1)
    replay_scale: float         # 回放比例缩放 (0.9|1.0|1.1)
    finetuning_type: str        # "full" | "lora"
    reason: str                 # 选择原因


@dataclass(frozen=True)
class ActionUCBSelection:
    action_key: str
    visits: int
    mean_reward: float
    reason: str
    diagnostics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ContinuousParamSample:
    source_action_key: str
    finetuning_type: str
    learning_rate: float
    replay_sample_ratio: float
    num_train_epochs: float
    reward: float
    normalized_vector: tuple[float, float, float]


@dataclass(frozen=True)
class ContinuousParamCandidate:
    source_action_key: str
    learning_rate: float
    replay_sample_ratio: float
    num_train_epochs: float
    normalized_vector: tuple[float, float, float]
    generation: str


@dataclass(frozen=True)
class ContinuousParamSelection:
    source_action_key: str
    learning_rate: float
    replay_sample_ratio: float
    num_train_epochs: float
    normalized_vector: tuple[float, float, float]
    predicted_reward: float
    exploration_bonus: float
    dag_context_bias: float
    cost_penalty: float
    score: float
    effective_samples: float
    reason: str
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DAGBranchCandidate:
    parent_node_id: str
    child_node_id: str
    action_key: str
    query_type: str
    score: float
    effect_metrics: dict[str, float]
    round_id: int | None
    decision: str


@dataclass(frozen=True)
class DAGQueryResult:
    enabled: bool
    query_type: str
    skipped_reason: str
    threshold: float | int | None
    matched_node_count: int
    candidate_branches: list[dict[str, Any]]
    top_candidate_node_ids: list[str]
    action_evidence: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MCTSHistoryLeafResult:
    enabled: bool
    query_type: str
    similar_nodes: list[dict[str, Any]]
    edge_filtered_evidence: dict[str, list[dict[str, Any]]]
    action_evidence: dict[str, dict[str, Any]]
    constraints: dict[str, Any]
    reason: str


_SAFE_ROLLBACK_ACTION_ORDER = (
    "conservative_anti_forget",
    "ultra_conservative_replay",
    "low_lr_more_steps",
    "probe_failure_refresh",
    "shift_window_balanced",
    "lora_conservative",
    "lora_ultra_safe",
)


def _finetuning_type_for_action(action_key: str, action_space: dict[str, dict]) -> str:
    template = action_space.get(action_key, {})
    return str(template.get("finetuning_type") or ("lora" if "lora" in action_key else "full")).lower()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _node_metrics(node: dict) -> dict[str, Any]:
    metrics = node.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _metric_value(node: dict, key: str, default: float = 0.0) -> float:
    metrics = _node_metrics(node)
    if key in metrics:
        return _safe_float(metrics.get(key), default)
    fallback_by_key = {
        "old_error_rate": node.get("forgetting_score"),
        "probe_acc_after": node.get("accuracy"),
        "probe_acc_frozen": node.get("accuracy"),
        "cotest_acc_after": node.get("stability_score"),
    }
    return _safe_float(fallback_by_key.get(key), default)


def _probe_metric(node: dict) -> float:
    metrics = _node_metrics(node)
    if metrics.get("probe_acc_frozen") is not None:
        return _safe_float(metrics.get("probe_acc_frozen"), 0.0)
    return _metric_value(node, "probe_acc_after", 0.0)


def _outgoing_edges(edges: list[dict], node_id: str) -> list[dict]:
    return [
        edge for edge in edges
        if isinstance(edge, dict) and str(edge.get("from_node_id", "")) == node_id
    ]


def _detect_dag_query_type(
    metrics_after: dict | None,
    previous_decision: str,
    rollback_streak: int,
) -> tuple[str, float | int | None, str]:
    metrics = metrics_after if isinstance(metrics_after, dict) else {}
    if rollback_streak >= MCTS_QUERY_ROLLBACK_STREAK_THRESHOLD or previous_decision == "prune":
        return "repeated_rollback", MCTS_QUERY_ROLLBACK_STREAK_THRESHOLD, "rollback_streak_or_prune"
    forgetting_delta = _safe_float(metrics.get("forgetting_delta"), 0.0)
    if forgetting_delta > MCTS_QUERY_HIGH_FORGETTING_THRESHOLD:
        return "high_forgetting", MCTS_QUERY_HIGH_FORGETTING_THRESHOLD, "forgetting_delta"
    probe = _safe_float(metrics.get("probe_acc_frozen", metrics.get("probe_acc")), 1.0)
    if probe < MCTS_QUERY_LOW_PROBE_THRESHOLD:
        return "low_probe", MCTS_QUERY_LOW_PROBE_THRESHOLD, "probe_acc_frozen"
    new_skill = _safe_float(metrics.get("new_skill_acc", metrics.get("new_skill_acc_after")), 1.0)
    if new_skill < MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD:
        return "low_new_skill", MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD, "new_skill_acc"
    cotest = _safe_float(metrics.get("cotest_acc", metrics.get("cotest_acc_after")), 1.0)
    if cotest < MCTS_QUERY_LOW_COTEST_THRESHOLD:
        return "low_cotest", MCTS_QUERY_LOW_COTEST_THRESHOLD, "cotest_acc"
    return "", None, "no_query_trigger"


def _node_matches_query_type(node: dict, query_type: str, threshold: float | int | None) -> bool:
    if query_type == "high_forgetting":
        return (
            _metric_value(node, "old_error_rate", 0.0) > _safe_float(threshold, 0.0)
            or _metric_value(node, "forgetting_delta", 0.0) > _safe_float(threshold, 0.0)
        )
    if query_type == "low_probe":
        return _probe_metric(node) < _safe_float(threshold, 1.0)
    if query_type == "low_new_skill":
        return _metric_value(node, "new_skill_acc_after", 1.0) < _safe_float(threshold, 1.0)
    if query_type == "low_cotest":
        return _metric_value(node, "cotest_acc_after", 1.0) < _safe_float(threshold, 1.0)
    if query_type == "repeated_rollback":
        return True
    return False


def _branch_effect_metrics(parent: dict, child: dict) -> dict[str, float]:
    parent_forget = _metric_value(parent, "old_error_rate", 0.0)
    child_forget = _metric_value(child, "old_error_rate", 0.0)
    parent_forgetting_delta = _metric_value(parent, "forgetting_delta", 0.0)
    child_forgetting_delta = _metric_value(child, "forgetting_delta", 0.0)
    parent_probe = _probe_metric(parent)
    child_probe = _probe_metric(child)
    parent_new = _metric_value(parent, "new_skill_acc_after", 0.0)
    child_new = _metric_value(child, "new_skill_acc_after", 0.0)
    parent_cotest = _metric_value(parent, "cotest_acc_after", 0.0)
    child_cotest = _metric_value(child, "cotest_acc_after", 0.0)
    return {
        "forget_recovery": parent_forget - child_forget,
        "forgetting_delta_recovery": parent_forgetting_delta - child_forgetting_delta,
        "probe_delta": child_probe - parent_probe,
        "new_skill_delta": child_new - parent_new,
        "cotest_delta": child_cotest - parent_cotest,
    }


def _decision_support_bonus(decision: str) -> float:
    if decision == "promote":
        return 0.15
    if decision in BRANCH_RETENTION_DECISIONS:
        return 0.08
    return 0.0


def _branch_score_for_query(
    query_type: str,
    effect: dict[str, float],
    edge: dict,
    child: dict,
) -> float:
    reward = _safe_float(edge.get("reward"), _safe_float(child.get("value_estimate"), 0.0))
    decision = str(edge.get("decision") or "")
    promote_bonus = _decision_support_bonus(decision)
    rollback_penalty = 0.10 if decision == "rollback" else 0.0
    if query_type == "high_forgetting":
        return (
            1.00 * effect["forget_recovery"]
            + 0.70 * effect["forgetting_delta_recovery"]
            + 0.25 * effect["probe_delta"]
            + 0.15 * effect["new_skill_delta"]
            + 0.10 * effect["cotest_delta"]
            + 0.20 * reward
            + promote_bonus
            - rollback_penalty
        )
    if query_type == "low_probe":
        return (
            1.00 * effect["probe_delta"]
            + 0.25 * effect["forget_recovery"]
            + 0.15 * effect["new_skill_delta"]
            + 0.10 * effect["cotest_delta"]
            + 0.20 * reward
            + promote_bonus
            - rollback_penalty
        )
    if query_type == "low_new_skill":
        return (
            1.00 * effect["new_skill_delta"]
            + 0.20 * effect["probe_delta"]
            + 0.20 * effect["forget_recovery"]
            + 0.10 * effect["cotest_delta"]
            + 0.20 * reward
            + promote_bonus
            - rollback_penalty
        )
    if query_type == "low_cotest":
        return (
            1.00 * effect["cotest_delta"]
            + 0.20 * effect["probe_delta"]
            + 0.20 * effect["forget_recovery"]
            + 0.10 * effect["new_skill_delta"]
            + 0.20 * reward
            + promote_bonus
            - rollback_penalty
        )
    if query_type == "repeated_rollback":
        recovery = max(
            effect["forget_recovery"],
            effect["probe_delta"],
            effect["new_skill_delta"],
            effect["cotest_delta"],
        )
        return recovery + 0.30 * reward + 2.0 * promote_bonus - rollback_penalty
    return 0.0


def _build_dag_query_result(
    search_dag_nodes: list[dict],
    search_dag_edges: list[dict],
    metrics_after: dict | None,
    previous_decision: str,
    rollback_streak: int,
) -> DAGQueryResult:
    query_type, threshold, trigger = _detect_dag_query_type(metrics_after, previous_decision, rollback_streak)
    if not query_type:
        return DAGQueryResult(False, "", trigger, threshold, 0, [], [], {})
    if len(search_dag_edges) < MCTS_QUERY_COLD_START_EDGES:
        return DAGQueryResult(
            False,
            query_type,
            f"cold_start_edges<{MCTS_QUERY_COLD_START_EDGES}",
            threshold,
            0,
            [],
            [],
            {},
        )

    nodes_by_id = _node_by_id(search_dag_nodes)
    matched_nodes = [
        node for node in search_dag_nodes
        if isinstance(node, dict) and _node_matches_query_type(node, query_type, threshold)
    ]
    candidates: list[DAGBranchCandidate] = []
    for parent in matched_nodes:
        parent_id = str(parent.get("node_id", ""))
        if not parent_id:
            continue
        for edge in _outgoing_edges(search_dag_edges, parent_id):
            child_id = str(edge.get("to_node_id", ""))
            child = nodes_by_id.get(child_id)
            if not isinstance(child, dict):
                continue
            metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
            action_key = str(metadata.get("action_key") or "")
            if not action_key:
                continue
            effect = _branch_effect_metrics(parent, child)
            score = _branch_score_for_query(query_type, effect, edge, child)
            candidates.append(DAGBranchCandidate(
                parent_node_id=parent_id,
                child_node_id=child_id,
                action_key=action_key,
                query_type=query_type,
                score=score,
                effect_metrics=effect,
                round_id=edge.get("round_id"),
                decision=str(edge.get("decision") or metadata.get("decision") or ""),
            ))

    if not candidates:
        return DAGQueryResult(False, query_type, "no_child_branches", threshold, len(matched_nodes), [], [], {})

    candidates.sort(key=lambda item: item.score, reverse=True)
    top_candidates = candidates[: max(1, MCTS_QUERY_CANDIDATE_TOP_K)]
    action_buckets: dict[str, list[DAGBranchCandidate]] = {}
    for candidate in top_candidates:
        action_buckets.setdefault(candidate.action_key, []).append(candidate)
    action_evidence: dict[str, dict[str, Any]] = {}
    for action_key, bucket in action_buckets.items():
        mean_score = sum(item.score for item in bucket) / max(1, len(bucket))
        bounded = max(-MCTS_QUERY_ACTION_BIAS_MAX, min(MCTS_QUERY_ACTION_BIAS_MAX, mean_score))
        action_evidence[action_key] = {
            "score": bounded,
            "raw_mean_score": mean_score,
            "samples": len(bucket),
            "best_child_node_id": bucket[0].child_node_id,
            "best_parent_node_id": bucket[0].parent_node_id,
            "best_effect_metrics": bucket[0].effect_metrics,
        }

    top_node_ids: list[str] = []
    for candidate in top_candidates:
        if candidate.child_node_id not in top_node_ids:
            top_node_ids.append(candidate.child_node_id)
    return DAGQueryResult(
        True,
        query_type,
        "",
        threshold,
        len(matched_nodes),
        [
            {
                "parent_node_id": item.parent_node_id,
                "child_node_id": item.child_node_id,
                "action_key": item.action_key,
                "score": item.score,
                "effect_metrics": item.effect_metrics,
                "round_id": item.round_id,
                "decision": item.decision,
            }
            for item in top_candidates
        ],
        top_node_ids,
        action_evidence,
    )


def _edge_training_summary(edge: dict) -> dict[str, Any]:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    summary = metadata.get("training_summary")
    return summary if isinstance(summary, dict) else {}


def _loss_diagnosis_from_edge(edge: dict) -> str:
    summary = _edge_training_summary(edge)
    diagnosis_raw = summary.get("diagnosis")
    diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}
    return str(diagnosis.get("loss_diagnosis") or "")


def _dataset_action_from_edge(edge: dict) -> str:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    return str(metadata.get("dataset_action") or metadata.get("dataset_selection_mode") or "")


def _current_situation_tags(
    metrics_after: dict | None,
    previous_decision: str,
    rollback_streak: int,
    round_data_stats: dict | None,
) -> set[str]:
    metrics = metrics_after if isinstance(metrics_after, dict) else {}
    data_stats = round_data_stats if isinstance(round_data_stats, dict) else {}
    tags: set[str] = set()
    if rollback_streak >= 2 or previous_decision in {"rollback", "prune"}:
        tags.add("rollback_pressure")
    if _safe_float(metrics.get("probe_acc_frozen", metrics.get("probe_acc")), 1.0) < MCTS_QUERY_LOW_PROBE_THRESHOLD:
        tags.add("low_probe")
    if _safe_float(metrics.get("forgetting_delta"), 0.0) > MCTS_QUERY_HIGH_FORGETTING_THRESHOLD:
        tags.add("high_forgetting")
    if _safe_float(metrics.get("new_skill_acc", metrics.get("new_skill_acc_after")), 1.0) < MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD:
        tags.add("low_new_skill")
    hard_ratio = _safe_float(data_stats.get("hard_ratio"), 0.0)
    if hard_ratio > 0.80 or bool(data_stats.get("hard_dominated_signal")):
        tags.add("hard_skew")
    summary_raw = metrics.get("training_summary")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    diagnosis_raw = summary.get("diagnosis")
    diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}
    loss_diagnosis = str(diagnosis.get("loss_diagnosis") or "")
    if loss_diagnosis:
        tags.add(f"loss_{loss_diagnosis}")
    eval_loss_raw = summary.get("eval_loss")
    eval_loss = eval_loss_raw if isinstance(eval_loss_raw, dict) else {}
    if _safe_float(eval_loss.get("eval_train_gap_last"), 0.0) > 0.10:
        tags.add("eval_train_gap")
    return tags


def _node_situation_tags(node: dict) -> set[str]:
    tags: set[str] = set()
    if _probe_metric(node) < MCTS_QUERY_LOW_PROBE_THRESHOLD:
        tags.add("low_probe")
    if _metric_value(node, "forgetting_delta", 0.0) > MCTS_QUERY_HIGH_FORGETTING_THRESHOLD:
        tags.add("high_forgetting")
    if _metric_value(node, "new_skill_acc_after", 1.0) < MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD:
        tags.add("low_new_skill")
    metrics = _node_metrics(node)
    summary_raw = metrics.get("training_summary")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    diagnosis_raw = summary.get("diagnosis")
    diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}
    loss_diagnosis = str(diagnosis.get("loss_diagnosis") or "")
    if loss_diagnosis:
        tags.add(f"loss_{loss_diagnosis}")
    eval_loss_raw = summary.get("eval_loss")
    eval_loss = eval_loss_raw if isinstance(eval_loss_raw, dict) else {}
    if _safe_float(eval_loss.get("eval_train_gap_last"), 0.0) > 0.10:
        tags.add("eval_train_gap")
    return tags


def build_mcts_history_leaf_result(
    *,
    search_dag_nodes: list[dict],
    search_dag_edges: list[dict],
    metrics_after: dict | None,
    previous_decision: str,
    rollback_streak: int,
    round_data_stats: dict | None = None,
    candidate_actions: dict[str, dict] | None = None,
) -> MCTSHistoryLeafResult:
    """Retrieve similar DAG situations and summarize outgoing-edge experience."""
    if not search_dag_nodes or not search_dag_edges:
        return MCTSHistoryLeafResult(False, "cold_start", [], {"successful_edges": [], "failed_edges": []}, {}, {}, "no DAG history")

    current_tags = _current_situation_tags(metrics_after, previous_decision, rollback_streak, round_data_stats)
    query_type, _threshold, trigger = _detect_dag_query_type(metrics_after, previous_decision, rollback_streak)
    if query_type:
        current_tags.add(query_type)
    nodes_by_id = _node_by_id(search_dag_nodes)
    similar_nodes: list[dict[str, Any]] = []
    for node in search_dag_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        node_tags = _node_situation_tags(node)
        matched = sorted(current_tags.intersection(node_tags))
        if query_type and _node_matches_query_type(node, query_type, _threshold):
            matched.append(query_type)
        if not matched and query_type != "repeated_rollback":
            continue
        denominator = max(1, len(current_tags.union(node_tags)))
        similarity = len(set(matched)) / denominator
        if query_type == "repeated_rollback" and not matched:
            similarity = max(similarity, 0.10)
            matched.append("repeated_rollback_context")
        if similarity <= 0:
            continue
        similar_nodes.append({
            "node_id": node_id,
            "similarity": round(similarity, 4),
            "matched_reasons": sorted(set(matched)),
        })
    similar_nodes.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
    similar_nodes = similar_nodes[: max(1, MCTS_QUERY_CANDIDATE_TOP_K)]
    if not similar_nodes:
        return MCTSHistoryLeafResult(False, query_type or trigger, [], {"successful_edges": [], "failed_edges": []}, {}, {}, "no similar nodes")

    successful_edges: list[dict[str, Any]] = []
    failed_edges: list[dict[str, Any]] = []
    per_action: dict[str, dict[str, float | int]] = {}
    for similar in similar_nodes:
        parent_id = str(similar.get("node_id") or "")
        parent = nodes_by_id.get(parent_id, {})
        for edge in _outgoing_edges(search_dag_edges, parent_id):
            child = nodes_by_id.get(str(edge.get("to_node_id") or ""))
            if not isinstance(child, dict):
                continue
            metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
            action_key = str(metadata.get("action_key") or "")
            if not action_key:
                continue
            effect = _branch_effect_metrics(parent, child)
            raw_reward = _edge_selection_reward(edge)
            numeric_raw_reward = _safe_float(raw_reward, 0.0)
            decision = str(edge.get("decision") or metadata.get("decision") or "")
            score = (
                numeric_raw_reward
                + 0.25 * max(effect.values())
                + _decision_support_bonus(decision)
                - (0.15 if decision in {"rollback", "prune"} else 0.0)
            )
            entry = {
                "from_node_id": parent_id,
                "to_node_id": edge.get("to_node_id"),
                "round_id": edge.get("round_id"),
                "action_key": action_key,
                "decision": decision,
                "reward": edge.get("reward"),
                "raw_reward": raw_reward,
                "similarity": similar.get("similarity"),
                "effect_metrics": effect,
                "loss_diagnosis": _loss_diagnosis_from_edge(edge),
                "dataset_action": _dataset_action_from_edge(edge),
                "score": score,
            }
            bucket = per_action.setdefault(action_key, {"support": 0.0, "failure": 0.0, "samples": 0})
            bucket["samples"] = int(bucket["samples"]) + 1
            if decision in ADVANCE_DECISIONS or numeric_raw_reward > 0:
                successful_edges.append(entry)
                bucket["support"] = float(bucket["support"]) + max(0.0, score)
            else:
                failed_edges.append(entry)
                bucket["failure"] = float(bucket["failure"]) + abs(min(0.0, score)) + 0.10

    successful_edges.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    failed_edges.sort(key=lambda item: float(item.get("score", 0.0)))
    action_evidence: dict[str, dict[str, Any]] = {}
    forbidden_action_keys: list[str] = []
    for action_key, values in per_action.items():
        samples = max(1, int(values.get("samples", 0)))
        support = float(values.get("support", 0.0)) / samples
        failure = float(values.get("failure", 0.0)) / samples
        net_score = support - failure
        bounded = max(-MCTS_QUERY_ACTION_BIAS_MAX, min(MCTS_QUERY_ACTION_BIAS_MAX, net_score))
        action_evidence[action_key] = {
            "score": bounded,
            "support_score": support,
            "failure_score": failure,
            "net_score": net_score,
            "evidence_count": samples,
        }
        if failure > support and samples >= 1 and rollback_streak >= 2:
            forbidden_action_keys.append(action_key)

    if previous_decision in {"rollback", "prune"} and rollback_streak >= 2:
        recent_edges = sorted(
            [edge for edge in search_dag_edges if isinstance(edge, dict)],
            key=lambda item: int(item.get("round_id", -1) if item.get("round_id") is not None else -1),
        )[-rollback_streak:]
        recent_actions = []
        for edge in recent_edges:
            metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
            if edge.get("decision") in {"rollback", "prune"} and metadata.get("action_key"):
                recent_actions.append(str(metadata.get("action_key")))
        if len(recent_actions) >= 2 and len(set(recent_actions)) == 1:
            forbidden_action_keys.append(recent_actions[0])

    allowed_candidate_keys = set(candidate_actions or {})
    forbidden_action_keys = sorted({key for key in forbidden_action_keys if not allowed_candidate_keys or key in allowed_candidate_keys})
    hard_ratio = _safe_float((round_data_stats or {}).get("hard_ratio"), 0.0) if isinstance(round_data_stats, dict) else 0.0
    forbid_retry_same_data = rollback_streak >= 2 or hard_ratio > 0.80
    constraints = {
        "forbidden_action_keys": forbidden_action_keys,
        "cooldown_action_keys": {key: 2 for key in forbidden_action_keys},
        "force_exploration": bool(rollback_streak >= 2 or forbidden_action_keys),
        "forbid_retry_same_data": forbid_retry_same_data,
        "preferred_dataset_actions": ["replace_dataset", "merge_shards"] if forbid_retry_same_data else ["shift_window", "merge_shards"],
        "preferred_update_types": ["lora", "conservative_full"] if rollback_streak >= 2 else [],
    }
    return MCTSHistoryLeafResult(
        True,
        query_type or trigger,
        similar_nodes,
        {"successful_edges": successful_edges[:MCTS_QUERY_CANDIDATE_TOP_K], "failed_edges": failed_edges[:MCTS_QUERY_CANDIDATE_TOP_K]},
        action_evidence,
        constraints,
        "similar DAG nodes retrieved and outgoing edges contrasted",
    )


def select_action_fallback(
    rollback_streak: int,
    previous_decision: str,
    edge_summary: dict | None = None,
) -> ActionSelectionResult:
    """确定性 action 选择 fallback。

    LLM agent 可以通过 decide_json() 覆盖，选择不同 action + 变异参数。
    """
    # 连续回退多 → 探索性/保守策略
    if rollback_streak >= 3:
        action_key = "probe_failure_refresh"
        lr_scale = 0.9
        replay_scale = 1.1
        reason = f"rollback_streak={rollback_streak}: refresh data conservatively"
    elif rollback_streak >= 2:
        action_key = "conservative_anti_forget"
        lr_scale = 0.9
        replay_scale = 1.1
        reason = f"rollback_streak={rollback_streak}: conservative"
    elif previous_decision == "rollback":
        action_key = "lora_conservative"
        lr_scale = 1.0
        replay_scale = 1.0
        reason = "previous rollback: cautious"
    else:
        action_key = "balanced_default"
        lr_scale = 1.0
        replay_scale = 1.0
        reason = "normal progression"

    finetuning_type = "lora" if "lora" in action_key else "full"

    return ActionSelectionResult(
        action_key=action_key,
        lr_scale=lr_scale,
        replay_scale=replay_scale,
        finetuning_type=finetuning_type,
        reason=reason,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  六.6、Data Window Agent fallback                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 确定性 fallback：根据数据压力和准确率趋势管理窗口。
# LLM agent（DATA_WINDOW_MANAGER_PROMPT）通过 decide_json() 覆盖。
# ════════════════════════════════════════════════════════════════════════════════




# ── _edge_selection_reward ────────────────────────────────────────────────────
# 从 DAG 边（edge）的 action_metadata 中提取 reward 值。
# 提取优先级：reward_components.raw_reward > action_metadata.raw_reward
#   > action_metadata.selection_reward > edge.reward
# ──────────────────────────────────────────────────────────────────────────────
def _edge_selection_reward(edge: dict) -> float | None:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    reward_components = metadata.get("reward_components", {})
    if isinstance(reward_components, dict):
        raw_reward = reward_components.get("raw_reward")
        if raw_reward is not None:
            try:
                return float(raw_reward)
            except (TypeError, ValueError):
                pass
    raw_reward = metadata.get("raw_reward", metadata.get("selection_reward"))
    if raw_reward is not None:
        try:
            return float(raw_reward)
        except (TypeError, ValueError):
            pass
    reward = edge.get("reward", None)
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _continuous_template_point(action_key: str, template: dict[str, Any]) -> tuple[float, float, float] | None:
    try:
        lr = float(template.get("learning_rate"))
        replay = float(template.get("replay_sample_ratio"))
        epochs = float(template.get("num_train_epochs"))
    except (TypeError, ValueError):
        return None
    if lr <= 0.0 or epochs <= 0.0:
        return None
    _ = action_key
    return lr, min(1.0, max(0.0, replay)), epochs


def _continuous_bounds(
    action_keys: list[str],
    action_space: dict[str, dict],
    default_action_key: str,
) -> dict[str, tuple[float, float]]:
    points: list[tuple[float, float, float]] = []
    for key in action_keys:
        point = _continuous_template_point(key, action_space.get(key, {}))
        if point is not None:
            points.append(point)
    if not points:
        point = _continuous_template_point(default_action_key, action_space.get(default_action_key, {}))
        if point is not None:
            points.append(point)
    if not points:
        points.append((5.0e-6, 0.30, 3.0))

    def _range(values: list[float], *, lower_floor: float, upper_cap: float | None = None) -> tuple[float, float]:
        low = min(values)
        high = max(values)
        if abs(high - low) <= 1e-12:
            low *= 0.8
            high *= 1.2
        low = max(lower_floor, low)
        if upper_cap is not None:
            high = min(upper_cap, high)
        if high < low:
            high = low
        return low, high

    lrs = [point[0] for point in points]
    replays = [point[1] for point in points]
    epochs = [point[2] for point in points]
    return {
        "learning_rate": _range(lrs, lower_floor=1e-12),
        "replay_sample_ratio": _range(replays, lower_floor=0.0, upper_cap=1.0),
        "num_train_epochs": _range(epochs, lower_floor=0.1),
    }


def _clamp_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return min(high, max(low, value))


def _normalize_continuous_vector(
    learning_rate: float,
    replay_sample_ratio: float,
    num_train_epochs: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    def _linear(value: float, key: str) -> float:
        low, high = bounds[key]
        if abs(high - low) <= 1e-12:
            return 0.5
        return min(1.0, max(0.0, (value - low) / (high - low)))

    lr_low, lr_high = bounds["learning_rate"]
    log_low = math.log(max(lr_low, 1e-12))
    log_high = math.log(max(lr_high, 1e-12))
    if abs(log_high - log_low) <= 1e-12:
        lr_norm = 0.5
    else:
        lr_norm = (math.log(max(learning_rate, 1e-12)) - log_low) / (log_high - log_low)
        lr_norm = min(1.0, max(0.0, lr_norm))
    return (
        lr_norm,
        _linear(replay_sample_ratio, "replay_sample_ratio"),
        _linear(num_train_epochs, "num_train_epochs"),
    )


def _edge_continuous_sample(
    edge: dict,
    *,
    bounds: dict[str, tuple[float, float]],
    action_space: dict[str, dict],
    finetuning_type: str,
) -> ContinuousParamSample | None:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    params = metadata.get("training_hyperparams") if isinstance(metadata.get("training_hyperparams"), dict) else {}
    continuous = metadata.get("continuous_params") if isinstance(metadata.get("continuous_params"), dict) else {}
    source_action_key = str(
        continuous.get("source_action_key")
        or metadata.get("source_action_key")
        or metadata.get("action_key")
        or ""
    )
    if not source_action_key:
        return None
    sample_finetuning = str(
        continuous.get("finetuning_type")
        or metadata.get("finetuning_type")
        or params.get("finetuning_type")
        or _finetuning_type_for_action(source_action_key, action_space)
    ).lower()
    if sample_finetuning != finetuning_type:
        return None
    template_point = _continuous_template_point(source_action_key, action_space.get(source_action_key, {}))
    try:
        lr = float(continuous.get("learning_rate", params.get("learning_rate")))
        replay = float(continuous.get("replay_sample_ratio", metadata.get("replay_sample_ratio")))
        epochs = float(continuous.get("num_train_epochs", params.get("num_train_epochs")))
    except (TypeError, ValueError):
        if template_point is None:
            return None
        try:
            lr = float(params.get("learning_rate", template_point[0]))
        except (TypeError, ValueError):
            lr = template_point[0]
        replay = float(metadata.get("replay_sample_ratio", template_point[1]) or template_point[1])
        epochs = float(params.get("num_train_epochs", template_point[2]) or template_point[2])
    reward = _edge_selection_reward(edge)
    if reward is None or lr <= 0.0 or epochs <= 0.0:
        return None
    replay = min(1.0, max(0.0, replay))
    return ContinuousParamSample(
        source_action_key=source_action_key,
        finetuning_type=sample_finetuning,
        learning_rate=lr,
        replay_sample_ratio=replay,
        num_train_epochs=epochs,
        reward=float(reward),
        normalized_vector=_normalize_continuous_vector(lr, replay, epochs, bounds),
    )


def _halton(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / float(base)
    value = max(1, int(index))
    while value > 0:
        result += factor * (value % base)
        value //= base
        factor /= float(base)
    return result


def _candidate_from_values(
    *,
    source_action_key: str,
    learning_rate: float,
    replay_sample_ratio: float,
    num_train_epochs: float,
    bounds: dict[str, tuple[float, float]],
    generation: str,
) -> ContinuousParamCandidate:
    lr = _clamp_to_bounds(learning_rate, bounds["learning_rate"])
    replay = _clamp_to_bounds(replay_sample_ratio, bounds["replay_sample_ratio"])
    epochs = _clamp_to_bounds(num_train_epochs, bounds["num_train_epochs"])
    return ContinuousParamCandidate(
        source_action_key=source_action_key,
        learning_rate=lr,
        replay_sample_ratio=replay,
        num_train_epochs=epochs,
        normalized_vector=_normalize_continuous_vector(lr, replay, epochs, bounds),
        generation=generation,
    )


def _nearest_continuous_action_key(
    *,
    learning_rate: float,
    replay_sample_ratio: float,
    num_train_epochs: float,
    action_keys: list[str],
    action_space: dict[str, dict],
    bounds: dict[str, tuple[float, float]],
    default_action_key: str,
) -> str:
    target = _normalize_continuous_vector(learning_rate, replay_sample_ratio, num_train_epochs, bounds)
    best_key = default_action_key
    best_distance = float("inf")
    for key in action_keys:
        point = _continuous_template_point(key, action_space.get(key, {}))
        if point is None:
            continue
        vector = _normalize_continuous_vector(point[0], point[1], point[2], bounds)
        distance = sum((target[index] - vector[index]) ** 2 for index in range(3))
        if distance < best_distance:
            best_key = key
            best_distance = distance
    return best_key


def _build_continuous_candidates(
    *,
    action_keys: list[str],
    action_space: dict[str, dict],
    default_action_key: str,
    bounds: dict[str, tuple[float, float]],
    samples: list[ContinuousParamSample],
    round_id: int,
) -> list[ContinuousParamCandidate]:
    candidates: list[ContinuousParamCandidate] = []

    for key in action_keys:
        point = _continuous_template_point(key, action_space.get(key, {}))
        if point is None:
            continue
        candidates.append(_candidate_from_values(
            source_action_key=key,
            learning_rate=point[0],
            replay_sample_ratio=point[1],
            num_train_epochs=point[2],
            bounds=bounds,
            generation="template_anchor",
        ))

    for sample in sorted(samples, key=lambda item: item.reward, reverse=True)[:5]:
        for lr_scale, replay_shift, epoch_shift, generation in (
            (1.0, 0.0, 0.0, "history_exact"),
            (0.9, 0.0, 0.0, "history_lr_down"),
            (1.1, 0.0, 0.0, "history_lr_up"),
            (1.0, 0.05, 0.0, "history_replay_up"),
            (1.0, -0.05, 0.0, "history_replay_down"),
            (1.0, 0.0, 0.25, "history_epochs_up"),
            (1.0, 0.0, -0.25, "history_epochs_down"),
        ):
            lr = sample.learning_rate * lr_scale
            replay = sample.replay_sample_ratio + replay_shift
            epochs = sample.num_train_epochs + epoch_shift
            source_key = _nearest_continuous_action_key(
                learning_rate=lr,
                replay_sample_ratio=replay,
                num_train_epochs=epochs,
                action_keys=action_keys,
                action_space=action_space,
                bounds=bounds,
                default_action_key=sample.source_action_key,
            )
            candidates.append(_candidate_from_values(
                source_action_key=source_key,
                learning_rate=lr,
                replay_sample_ratio=replay,
                num_train_epochs=epochs,
                bounds=bounds,
                generation=generation,
            ))

    candidate_target = max(1, int(MCTS_CONTINUOUS_CANDIDATES))
    lr_low, lr_high = bounds["learning_rate"]
    log_lr_low = math.log(max(lr_low, 1e-12))
    log_lr_span = math.log(max(lr_high, 1e-12)) - log_lr_low
    replay_low, replay_high = bounds["replay_sample_ratio"]
    epoch_low, epoch_high = bounds["num_train_epochs"]
    for index in range(1, candidate_target + 1):
        halton_index = index + max(0, int(round_id))
        lr = math.exp(log_lr_low + _halton(halton_index, 2) * log_lr_span)
        replay = replay_low + _halton(halton_index, 3) * (replay_high - replay_low)
        epochs = epoch_low + _halton(halton_index, 5) * (epoch_high - epoch_low)
        source_key = _nearest_continuous_action_key(
            learning_rate=lr,
            replay_sample_ratio=replay,
            num_train_epochs=epochs,
            action_keys=action_keys,
            action_space=action_space,
            bounds=bounds,
            default_action_key=default_action_key,
        )
        candidates.append(_candidate_from_values(
            source_action_key=source_key,
            learning_rate=lr,
            replay_sample_ratio=replay,
            num_train_epochs=epochs,
            bounds=bounds,
            generation="halton",
        ))

    unique: dict[tuple[str, float, float, float], ContinuousParamCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.source_action_key,
            round(candidate.learning_rate, 12),
            round(candidate.replay_sample_ratio, 6),
            round(candidate.num_train_epochs, 6),
        )
        unique.setdefault(key, candidate)
    return list(unique.values())


def _score_continuous_candidate(
    candidate: ContinuousParamCandidate,
    *,
    samples: list[ContinuousParamSample],
    action_evidence: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    weighted_reward = 0.0
    total_weight = 0.0
    nearest_adjusted_reward = 0.0
    bandwidth = max(1e-6, float(MCTS_CONTINUOUS_BANDWIDTH))
    for sample in samples:
        distance_sq = sum(
            (candidate.normalized_vector[index] - sample.normalized_vector[index]) ** 2
            for index in range(3)
        )
        distance = math.sqrt(distance_sq)
        weight = math.exp(-distance_sq / (2.0 * bandwidth * bandwidth))
        weighted_reward += weight * sample.reward
        total_weight += weight
        nearest_adjusted_reward = max(nearest_adjusted_reward, sample.reward - 0.25 * distance)
    weighted_mean_reward = weighted_reward / total_weight if total_weight > 1e-12 else 0.0
    predicted_reward = max(weighted_mean_reward, nearest_adjusted_reward)
    exploration_bonus = float(MCTS_CONTINUOUS_EXPLORATION_BETA) * math.sqrt(
        math.log(len(samples) + 2.0) / (total_weight + 1.0)
    )
    evidence = action_evidence.get(candidate.source_action_key, {}) if isinstance(action_evidence, dict) else {}
    dag_context_bias = _safe_float(evidence.get("score"), 0.0) if isinstance(evidence, dict) else 0.0
    cost_penalty = float(MCTS_CONTINUOUS_EPOCH_COST_WEIGHT) * candidate.normalized_vector[2]
    score = predicted_reward + exploration_bonus + dag_context_bias - cost_penalty
    return {
        "score": score,
        "predicted_reward": predicted_reward,
        "exploration_bonus": exploration_bonus,
        "dag_context_bias": dag_context_bias,
        "cost_penalty": cost_penalty,
        "effective_samples": total_weight,
    }


def _select_continuous_params_by_surrogate(
    *,
    search_dag_edges: list[dict],
    default_action_key: str,
    action_keys: list[str],
    action_space: dict[str, dict],
    action_evidence: dict[str, dict[str, Any]] | None = None,
    round_id: int = 0,
) -> ContinuousParamSelection:
    finetuning_type = _finetuning_type_for_action(default_action_key, action_space)
    family_action_keys = [
        key for key in action_keys
        if _finetuning_type_for_action(key, action_space) == finetuning_type
        and _continuous_template_point(key, action_space.get(key, {})) is not None
    ]
    if not family_action_keys:
        family_action_keys = [default_action_key]
    if default_action_key not in family_action_keys:
        family_action_keys.insert(0, default_action_key)

    bounds = _continuous_bounds(family_action_keys, action_space, default_action_key)
    samples = [
        sample for edge in search_dag_edges
        if (sample := _edge_continuous_sample(
            edge,
            bounds=bounds,
            action_space=action_space,
            finetuning_type=finetuning_type,
        )) is not None
    ]
    candidates = _build_continuous_candidates(
        action_keys=family_action_keys,
        action_space=action_space,
        default_action_key=default_action_key,
        bounds=bounds,
        samples=samples,
        round_id=round_id,
    )
    if not candidates:
        point = _continuous_template_point(default_action_key, action_space.get(default_action_key, {})) or (5.0e-6, 0.30, 3.0)
        candidates = [_candidate_from_values(
            source_action_key=default_action_key,
            learning_rate=point[0],
            replay_sample_ratio=point[1],
            num_train_epochs=point[2],
            bounds=bounds,
            generation="fallback_template",
        )]

    scored: list[tuple[ContinuousParamCandidate, dict[str, float]]] = [
        (
            candidate,
            _score_continuous_candidate(
                candidate,
                samples=samples,
                action_evidence=action_evidence,
            ),
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    cold_start = not samples
    if cold_start:
        best_candidate, best_score = next(
            (
                item for item in scored
                if item[0].source_action_key == default_action_key
                and item[0].generation == "template_anchor"
            ),
            scored[0],
        )
    else:
        best_candidate, best_score = scored[0]

    top_candidates = []
    for candidate, values in scored[:5]:
        top_candidates.append({
            "source_action_key": candidate.source_action_key,
            "learning_rate": candidate.learning_rate,
            "replay_sample_ratio": candidate.replay_sample_ratio,
            "num_train_epochs": candidate.num_train_epochs,
            "generation": candidate.generation,
            "score": values["score"],
            "predicted_reward": values["predicted_reward"],
            "exploration_bonus": values["exploration_bonus"],
            "dag_context_bias": values["dag_context_bias"],
            "cost_penalty": values["cost_penalty"],
            "effective_samples": values["effective_samples"],
            "raw_reward_basis": "raw_reward_without_gate_bonus",
        })

    bounds_json = {key: [value[0], value[1]] for key, value in bounds.items()}
    selected_point = {
        "source_action_key": best_candidate.source_action_key,
        "learning_rate": best_candidate.learning_rate,
        "replay_sample_ratio": best_candidate.replay_sample_ratio,
        "num_train_epochs": best_candidate.num_train_epochs,
        "normalized_vector": list(best_candidate.normalized_vector),
        "generation": best_candidate.generation,
    }
    diagnostics = {
        "enabled": True,
        "cold_start": cold_start,
        "finetuning_type": finetuning_type,
        "bounds": bounds_json,
        "candidate_count": len(candidates),
        "history_sample_count": len(samples),
        "selected_point": selected_point,
        "top_candidates": top_candidates,
        "bandwidth": MCTS_CONTINUOUS_BANDWIDTH,
        "exploration_beta": MCTS_CONTINUOUS_EXPLORATION_BETA,
        "epoch_cost_weight": MCTS_CONTINUOUS_EPOCH_COST_WEIGHT,
    }
    return ContinuousParamSelection(
        source_action_key=best_candidate.source_action_key,
        learning_rate=best_candidate.learning_rate,
        replay_sample_ratio=best_candidate.replay_sample_ratio,
        num_train_epochs=best_candidate.num_train_epochs,
        normalized_vector=best_candidate.normalized_vector,
        predicted_reward=best_score["predicted_reward"],
        exploration_bonus=best_score["exploration_bonus"],
        dag_context_bias=best_score["dag_context_bias"],
        cost_penalty=best_score["cost_penalty"],
        score=best_score["score"],
        effective_samples=best_score["effective_samples"],
        reason="cold_start_continuous_candidate" if cold_start else "surrogate_ucb_score",
        diagnostics=diagnostics,
    )


# ── _select_action_key_by_ucb ────────────────────────────────────────────────
# 用 MCTS UCB 从历史 DAG 边中选出最优的 action key。
#
# 工作方式：
#   1. 遍历所有 DAG 边，按 action_key 分组统计：每个 action 的访问次数和总 reward
#   2. 对每个 action 计算 UCB 分数
#   3. 返回 UCB 分数最高的 action key
#
# 这实现了"自动超参数调优"：reward 高的 action 会被更多使用，
# 但未充分探索的 action 也有机会被选中。
# ──────────────────────────────────────────────────────────────────────────────
def _select_action_key_by_ucb(
    search_dag_edges: list[dict],    # DAG 边列表
    default_action_key: str,        # 默认 action（当没有历史数据时使用）
    action_keys: list[str],         # 本轮允许的 action key 白名单
    action_evidence: dict[str, dict[str, Any]] | None = None,
) -> ActionUCBSelection:            # 返回选中的 action 及诊断信息
    stats: dict[str, dict[str, float]] = {}
    for edge in search_dag_edges:
        metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
        key = metadata.get("action_key", "")
        reward = _edge_selection_reward(edge)
        if not key or reward is None:
            continue
        item = stats.setdefault(key, {"visits": 0.0, "reward_sum": 0.0})
        item["visits"] += 1.0
        item["reward_sum"] += float(reward)

    allowed_keys = [key for key in action_keys if key]
    if not allowed_keys:
        allowed_keys = [default_action_key]
    if default_action_key not in allowed_keys:
        allowed_keys.insert(0, default_action_key)

    total_visits = sum(stats.get(key, {}).get("visits", 0.0) for key in allowed_keys) + 1.0
    best_key = allowed_keys[0]
    best_score = float("-inf")
    best_visits = 0
    best_mean = 0.0
    best_reason = "ucb_score"
    diagnostics: dict[str, dict[str, Any]] = {}

    for key in allowed_keys:
        item = stats.get(key, {"visits": 0.0, "reward_sum": 0.0})
        visits = int(item["visits"])
        mean_reward = item["reward_sum"] / max(item["visits"], 1.0)
        evidence = action_evidence.get(key, {}) if isinstance(action_evidence, dict) else {}
        evidence_score = _safe_float(evidence.get("score"), 0.0) if isinstance(evidence, dict) else 0.0
        if visits == 0:
            score = float("inf")  # 未尝试过的 action 优先
            reason = "forced_initial_trial"
        else:
            score = mean_reward + MCTS_EXPLORATION_C_ACTION * math.sqrt(math.log(total_visits + 1.0) / visits)
            if evidence_score:
                score += evidence_score
                reason = "ucb_score_with_dag_query_evidence"
            else:
                reason = "ucb_score"
        diagnostics[key] = {
            "visits": visits,
            "mean_reward": mean_reward,
            "ucb_score": "inf" if math.isinf(score) else score,
            "reason": reason,
            "dag_query_evidence_score": evidence_score,
        }
        if score > best_score:
            best_score = score
            best_key = key
            best_visits = visits
            best_mean = mean_reward
            best_reason = reason

    return ActionUCBSelection(
        action_key=best_key,
        visits=best_visits,
        mean_reward=best_mean,
        reason=best_reason,
        diagnostics=diagnostics,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  九、DAG 更新策略 — update_search_dag_policy()                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 每轮评估后更新搜索 DAG（Directed Acyclic Graph）：
#   1. 创建新节点，记录本轮评估指标
#   2. 创建新边，连接父节点 → 新节点，记录 action 和 reward
#   3. 回传 reward 值到整条路径（backpropagate）
#   4. 用 UCB 选出下一轮的分支父节点
#
# DAG 的作用：记录所有搜索路径，用于 MCTS 的探索-利用决策。
# ════════════════════════════════════════════════════════════════════════════════


def update_search_dag_policy(
    nodes: list[dict],              # 当前 DAG 节点列表
    edges: list[dict],              # 当前 DAG 边列表
    parent_node_id: str,            # 父节点 ID（本轮训练的起点）
    decision: str,                  # 本轮决策（promote/rollback/prune）
    metrics: EvalResultPayload,     # 本轮评估结果
    round_id: int,                  # 当前轮次
    replay_used: bool,              # 是否使用了回放数据
    action_metadata: dict | None = None, # action 元数据
) -> DAGPolicyResult:
    # 计算 reward
    reward_components = score_candidate_components(metrics)
    reward = reward_components["reward"]
    current_nodes = [dict(n) for n in nodes]
    current_edges = [dict(e) for e in edges]

    # 创建新 DAG 节点——记录本轮评估指标
    new_node = add_search_node(
        parent_node_id=parent_node_id,
        accuracy=metrics.probe_acc_after,
        forgetting_score=metrics.old_error_rate,
        stability_score=metrics.cotest_acc_after,
        value_estimate=reward,
        visit_count=0,
        metrics=metrics.model_dump(mode="json"),
    )
    current_nodes.append(new_node.model_dump())

    # 组装边元数据
    metadata = dict(action_metadata or {})
    action_type = metadata.get("action_type") or (
        "keep_branch" if decision in {"provisional_promote", "keep_branch"}
        else "reuse_buffer_data" if replay_used
        else "add_dataset"
    )
    metadata.setdefault("replay_used", replay_used)
    metadata.setdefault("decision", decision)
    metadata.setdefault("reward", reward)
    metadata.setdefault("raw_reward", reward_components["raw_reward"])
    metadata.setdefault("selection_reward", reward_components["raw_reward"])
    metadata.setdefault("reward_components", reward_components)
    metadata.setdefault("metrics", metrics.model_dump(mode="json"))
    action_summary = (
        f"round_{round_id} {decision} "
        f"(reward={reward:.3f}, probe_acc={metrics.probe_acc_after:.3f}, "
        f"probe_acc_frozen={metrics.probe_acc_frozen if metrics.probe_acc_frozen is not None else metrics.probe_acc_after:.3f}, "
        f"cotest_acc={metrics.cotest_acc_after:.3f})"
    )

    # 创建新边：连接父节点 → 新节点
    if parent_node_id:
        edge = add_search_edge(
            from_node_id=parent_node_id,
            to_node_id=new_node.node_id,
            action_type=action_type,
            action_summary=action_summary,
            round_id=round_id,
            decision=decision,
            reward=reward,
            action_metadata=metadata,
        )
        current_edges.append(edge.model_dump())

    # 回传 reward 到整条搜索路径
    current_nodes = backpropagate_value(current_nodes, current_edges, new_node.node_id, reward)
    # 用 UCB 选出下一轮的分支父节点
    next_parent = select_next_parent_node(current_nodes, new_node.node_id)

    return DAGPolicyResult(
        nodes=current_nodes,
        edges=current_edges,
        new_node_id=new_node.node_id,
        next_parent_node_id=next_parent,
        action_type=action_type,
        action_summary=action_summary,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  十、Replay Buffer 策略 — build_replay_entries_policy()                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 把本轮的题目存入回放池（replay buffer），供后续轮次复用。
#
# 设计动机：
#   - 每轮有价值的题目不应被丢弃，后续训练可以回放复习
#   - 按 reward 排序，只保留 top-K（REPLAY_BUFFER_MAX_SIZE）
#   - 去重：question_id 相同的题目只保留一条
# ════════════════════════════════════════════════════════════════════════════════


def build_replay_entries_policy(
    filtered_questions: list[dict],     # 本轮过滤后的题目
    metrics: EvalResultPayload,         # 本轮评估结果
    round_id: int,                      # 当前轮次
    decision: str,                      # 本轮决策
    existing_entries: list[dict],       # 已有的回放条目
) -> list[dict]:
    """Store reusable training examples from promoted champion branches only."""
    if decision != "promote":
        return list(existing_entries)

    reward = max(0.0, score_candidate(metrics))
    dataset_signature = f"round_{round_id}_{decision}"

    # 用 question_id 去重
    by_question_id = {
        e.get("question_id"): e for e in existing_entries if e.get("question_id")
    }
    updated = list(existing_entries)

    for q in filtered_questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("question_id", "")
        if not qid or qid in by_question_id:
            continue

        # 创建回放条目
        dynamic_difficulty = str(q.get("dynamic_difficulty") or q.get("bucket") or "unknown")
        entry = create_replay_entry(
            dataset_signature=dataset_signature,
            source_round=round_id,
            success_score=reward,
            question_id=qid,
            question_text=q.get("question_text", ""),
            gold_answer=q.get("gold_answer", ""),
            rollout_gold_answer=q.get("rollout_gold_answer", ""),
            train_output=q.get("train_output", ""),
            target_style=q.get("target_style", "answer"),
            evaluation_method=q.get("evaluation_method", "gold"),
            needs_judge=bool(q.get("needs_judge", False)),
            source_dataset_id=q.get("source_dataset_id"),
            source_dataset_row_id=q.get("source_dataset_row_id"),
            source_dataset_split=q.get("source_dataset_split"),
            source_dataset_subset=q.get("source_dataset_subset"),
            source_dataset_requested_split=q.get("source_dataset_requested_split"),
            source_dataset_split_names=q.get("source_dataset_split_names") or [],
            source_dataset_columns=q.get("source_dataset_columns") or [],
            source_dataset_first_row=q.get("source_dataset_first_row") or {},
            source_dataset_schema=q.get("source_dataset_schema") or {},
            dynamic_difficulty=dynamic_difficulty,
            origin_round_id=round_id,
            pass_count=q.get("pass_count"),
            rollout_count=q.get("rollout_count"),
            pass_rate=q.get("pass_rate"),
            bucket=dynamic_difficulty,
            module=str(q.get("module", "unknown") or "unknown"),
            used_in_rounds=[],
        )
        updated.append(entry.model_dump())

    # 按 reward 降序排序，保留 top-K
    updated.sort(key=lambda e: e.get("success_score", 0.0), reverse=True)
    return updated[:REPLAY_BUFFER_MAX_SIZE]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  十一、MCTS 边查询与摘要 — query_mcts_edges() / build_parameter_edge_summary() ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 这些函数用于从 DAG 历史中查询和摘要边信息，为 parameter_master 的 LLM
# 调用提供结构化的上下文数据。
# ════════════════════════════════════════════════════════════════════════════════


# ── 辅助：按 node_id 索引节点 ──
def _node_by_id(nodes: list[dict]) -> dict[str, dict]:
    return {
        str(node.get("node_id", "")): node
        for node in nodes
        if isinstance(node, dict) and node.get("node_id")
    }


# ── 从边或子节点中提取评估指标 ──
def _edge_metrics(edge: dict, nodes_by_id: dict[str, dict]) -> dict[str, Any]:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    metrics = metadata.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    child = nodes_by_id.get(str(edge.get("to_node_id", "")), {})
    if isinstance(child.get("metrics"), dict) and child["metrics"]:
        return child["metrics"]
    return {
        "probe_acc_after": child.get("accuracy"),
        "old_error_rate": child.get("forgetting_score"),
        "cotest_acc_after": child.get("stability_score"),
    }


# ── 从边的 action_metadata 中提取训练超参数 ──
def _edge_hyperparams(edge: dict) -> dict[str, Any]:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    params = metadata.get("training_hyperparams")
    return dict(params) if isinstance(params, dict) else {}


# ── summarize_mcts_edge ───────────────────────────────────────────────────────
# 把一条 DAG 边摘要为结构化的 dict，方便 LLM 和代码读取。
# 摘要字段包括：轮次、action template、学习率、replay 比例、遗忘、probe 准确率等。
# ──────────────────────────────────────────────────────────────────────────────
def summarize_mcts_edge(edge: dict, nodes_by_id: dict[str, dict]) -> dict[str, Any]:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    metrics = _edge_metrics(edge, nodes_by_id)
    hyperparams = _edge_hyperparams(edge)
    reward_components = metadata.get("reward_components", {})
    if not isinstance(reward_components, dict):
        reward_components = {}
    raw_reward = reward_components.get("raw_reward", metadata.get("raw_reward"))
    if raw_reward is None:
        raw_reward = _edge_selection_reward(edge)
    training_summary = metadata.get("training_summary")
    if not isinstance(training_summary, dict):
        metrics_training_summary = metrics.get("training_summary")
        training_summary = metrics_training_summary if isinstance(metrics_training_summary, dict) else {}
    training_summary_dict: dict[str, Any] = training_summary
    train_loss_raw = training_summary_dict.get("train_loss")
    eval_loss_raw = training_summary_dict.get("eval_loss")
    training_process_raw = training_summary_dict.get("training_process")
    diagnosis_raw = training_summary_dict.get("diagnosis")
    loss_phase_raw = training_summary_dict.get("loss_phase")
    train_loss = train_loss_raw if isinstance(train_loss_raw, dict) else {}
    eval_loss = eval_loss_raw if isinstance(eval_loss_raw, dict) else {}
    training_process = training_process_raw if isinstance(training_process_raw, dict) else {}
    diagnosis = diagnosis_raw if isinstance(diagnosis_raw, dict) else {}
    loss_phase = loss_phase_raw if isinstance(loss_phase_raw, dict) else {}
    compact_training_summary = None
    if training_summary_dict:
        compact_training_summary = {
            "available": bool(training_summary_dict.get("available")),
            "train_loss_last": train_loss.get("last"),
            "train_loss_reduction_ratio": train_loss.get("reduction_ratio"),
            "train_loss_trend": train_loss.get("trend"),
            "eval_loss_last": eval_loss.get("last"),
            "eval_loss_min": eval_loss.get("min"),
            "eval_loss_min_at_step": eval_loss.get("min_at_step"),
            "eval_count": eval_loss.get("eval_count"),
            "eval_train_gap_last": eval_loss.get("eval_train_gap_last"),
            "learning_rate_first": training_process.get("learning_rate_first"),
            "learning_rate_last": training_process.get("learning_rate_last"),
            "learning_rate_min": training_process.get("learning_rate_min"),
            "learning_rate_max": training_process.get("learning_rate_max"),
            "learning_rate_schedule_shape": training_process.get("learning_rate_schedule_shape"),
            "loss_phase": loss_phase.get("phase"),
            "scheduler_diagnosis": loss_phase.get("scheduler_diagnosis"),
            "loss_diagnosis": diagnosis.get("loss_diagnosis"),
        }
    return {
        "round": edge.get("round_id"),
        "template": metadata.get("action_key", ""),
        "lr": hyperparams.get("learning_rate"),
        "replay": metadata.get("replay_sample_ratio"),
        "forget": metrics.get("old_error_rate"),
        "probe": metrics.get("probe_acc_frozen", metrics.get("probe_acc_after")),
        "new_acc": metrics.get("new_skill_acc_after"),
        "cotest": metrics.get("cotest_acc_after"),
        "reward": edge.get("reward"),
        "raw_reward": raw_reward,
        "gate_bonus": reward_components.get("gate_bonus"),
        "outcome": edge.get("decision", metadata.get("decision", "")),
        "training_summary": compact_training_summary,
    }


# ── query_mcts_edges ─────────────────────────────────────────────────────────
# 按条件查询历史 DAG 边（用于 parameter_master 和 replay_teacher 做决策）。
#
# 支持的过滤条件：
#   - decision:    只查特定决策结果（如 "promote"）
#   - forgetting_gt: 只查遗忘率超过阈值的边（"哪里翻车了"）
#   - min_raw_reward: 只查 reward 超过阈值的边（"哪里做得好"）
#   - top_k:        返回 top-K 条（按 raw_reward 降序 + 遗忘率升序）
# ──────────────────────────────────────────────────────────────────────────────
def query_mcts_edges(
    nodes: list[dict],
    edges: list[dict],
    *,
    forgetting_gt: float | None = None,   # 只返回遗忘率 > 此值的边
    decision: str | None = None,           # 只返回特定决策的边
    min_raw_reward: float | None = None,   # 只返回 raw_reward ≥ 此值的边
    top_k: int = 5,                        # 返回 top-K 条
) -> list[dict[str, Any]]:
    """Small structured DAG query used by parameter/replay teachers."""
    nodes_by_id = _node_by_id(nodes)
    hits: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        summary = summarize_mcts_edge(edge, nodes_by_id)
        if decision and summary.get("outcome") != decision:
            continue
        if forgetting_gt is not None:
            try:
                forget = float(summary.get("forget", 0.0) or 0.0)
            except (TypeError, ValueError):
                forget = 0.0
            if forget <= forgetting_gt:
                continue
        if min_raw_reward is not None:
            try:
                raw_reward = float(summary.get("raw_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                raw_reward = 0.0
            if raw_reward < min_raw_reward:
                continue
        hits.append(summary)

    hits.sort(
        key=lambda item: (
            float(item.get("raw_reward", item.get("reward", 0.0)) or 0.0),
            -float(item.get("forget", 0.0) or 0.0),  # 遗忘率低的排前面
        ),
        reverse=True,
    )
    return hits[: max(0, int(top_k))]


# ── build_parameter_edge_summary ─────────────────────────────────────────────
# 为 parameter_master 的 LLM 调用准备上下文摘要。
# 包含两类边：
#   - recent_edges:   最近 3 轮的历史（了解最近发生了什么）
#   - forgetting_edges: 遗忘率 > 0.5 的边（哪些策略导致遗忘）
# ──────────────────────────────────────────────────────────────────────────────
def build_parameter_edge_summary(
    nodes: list[dict],
    edges: list[dict],
    current_template: str,              # 当前使用的 action 模板名
    template_defaults: dict,            # 当前模板的默认超参数
    rollback_streak: int,               # 连续回退次数
    limit: int = 5,                     # 非 recent 类最多条数
    recent_limit: int = 3,              # 最近历史条数
    metrics_after: dict | None = None,
    previous_decision: str = "",
    round_data_stats: dict | None = None,
) -> dict[str, Any]:
    nodes_by_id = _node_by_id(nodes)
    safe_recent_limit = max(0, int(recent_limit))
    safe_limit = max(0, int(limit))
    # 最近 3 条边
    recent_edges = [
        summarize_mcts_edge(edge, nodes_by_id)
        for edge in sorted(
            [edge for edge in edges if isinstance(edge, dict)],
            key=lambda item: int(item.get("round_id", -1) if item.get("round_id") is not None else -1),
        )[-safe_recent_limit:]
    ]
    # 遗忘边（遗忘率 > 0.5 的）
    forgetting_edges = query_mcts_edges(nodes, edges, forgetting_gt=0.5, top_k=safe_limit)
    history_leaf = build_mcts_history_leaf_result(
        search_dag_nodes=nodes,
        search_dag_edges=edges,
        metrics_after=metrics_after,
        previous_decision=previous_decision,
        rollback_streak=rollback_streak,
        round_data_stats=round_data_stats,
        candidate_actions=_candidate_action_space(),
    )
    return {
        "current_template": current_template,
        "template_defaults": {
            "lr": template_defaults.get("learning_rate"),
            "replay": template_defaults.get("replay_sample_ratio"),
        },
        "recent_3_edges": recent_edges,
        "forgetting_edges": forgetting_edges,
        "mcts_history_leaf": {
            "enabled": history_leaf.enabled,
            "query_type": history_leaf.query_type,
            "similar_nodes": history_leaf.similar_nodes,
            "edge_filtered_evidence": history_leaf.edge_filtered_evidence,
            "action_evidence": history_leaf.action_evidence,
            "constraints": history_leaf.constraints,
            "reason": history_leaf.reason,
        },
        "rollback_streak": rollback_streak,
        "cold_start_edges": MCTS_TUNER_COLD_START_EDGES,
        "scale_clamp": {
            "min": MCTS_MUTATION_SCALE_MIN,
            "max": MCTS_MUTATION_SCALE_MAX,
        },
    }


def _compact_node_metrics(node: dict[str, Any]) -> dict[str, Any]:
    metrics = _node_metrics(node)
    return {
        "probe_acc": _probe_metric(node),
        "old_error_rate": metrics.get("old_error_rate", node.get("forgetting_score")),
        "new_skill_acc_after": metrics.get("new_skill_acc_after"),
        "cotest_acc_after": metrics.get("cotest_acc_after", node.get("stability_score")),
        "forgetting_delta": metrics.get("forgetting_delta"),
    }


def _compact_current_metrics(metrics_after: dict[str, Any]) -> dict[str, Any]:
    training_summary = metrics_after.get("training_summary")
    summary = training_summary if isinstance(training_summary, dict) else {}
    eval_loss_raw = summary.get("eval_loss")
    eval_loss = eval_loss_raw if isinstance(eval_loss_raw, dict) else {}
    return {
        "probe_acc": metrics_after.get("probe_acc_frozen", metrics_after.get("probe_acc_after")),
        "old_error_rate": metrics_after.get("old_error_rate"),
        "new_skill_acc_after": metrics_after.get("new_skill_acc_after", metrics_after.get("new_skill_acc")),
        "cotest_acc_after": metrics_after.get("cotest_acc_after", metrics_after.get("cotest_acc")),
        "forgetting_delta": metrics_after.get("forgetting_delta"),
        "eval_train_gap_last": eval_loss.get("eval_train_gap_last"),
    }


def _compact_data_pressure(round_data_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "hard_ratio": _safe_float(round_data_stats.get("hard_ratio"), 0.0),
        "needs_more_data": bool(round_data_stats.get("needs_more_data", False)),
        "low_train_signal": bool(round_data_stats.get("low_train_signal", False)),
        "hard_dominated_signal": bool(round_data_stats.get("hard_dominated_signal", False)),
    }


def _compact_deterministic_action(decision: ParameterMasterDecision) -> dict[str, Any]:
    metadata = decision.action_metadata if isinstance(decision.action_metadata, dict) else {}
    return {
        "action_key": decision.action_key,
        "replay_sample_ratio": decision.replay_sample_ratio,
        "dataset_selection_mode": metadata.get("dataset_selection_mode", "single_shard"),
        "branch_parent_node_id": decision.branch_parent_node_id,
        "finetuning_type": metadata.get("finetuning_type", decision.training_hyperparams.get("finetuning_type", "full")),
        "action_reason": metadata.get("action_reason", decision.decision_summary),
        "previous_action_fallback": metadata.get("previous_action_fallback", ""),
        "method_retry_same_data": bool(metadata.get("method_retry_same_data", False)),
        "forced_diversification": bool(metadata.get("forced_diversification", False)),
        "data_window_offset": metadata.get("data_window_offset"),
        "data_window_size": metadata.get("data_window_size"),
    }


def _compact_edge_card(
    edge: dict[str, Any],
    parent: dict[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
    raw_reward = _edge_selection_reward(edge)
    decision = str(edge.get("decision") or metadata.get("decision") or "")
    effect = _branch_effect_metrics(parent, child)
    numeric_raw_reward = _safe_float(raw_reward, 0.0)
    score = (
        numeric_raw_reward
        + 0.25 * max(effect.values())
        + _decision_support_bonus(decision)
        - (0.15 if decision in {"rollback", "prune"} else 0.0)
    )
    return {
        "from_node_id": edge.get("from_node_id"),
        "to_node_id": edge.get("to_node_id"),
        "round_id": edge.get("round_id"),
        "action_key": str(metadata.get("action_key") or ""),
        "decision": decision,
        "score": score,
        "reward": edge.get("reward"),
        "raw_reward": raw_reward,
        "effect_metrics": {
            "probe_delta": effect.get("probe_delta"),
            "forget_recovery": effect.get("forget_recovery"),
            "new_skill_delta": effect.get("new_skill_delta"),
            "cotest_delta": effect.get("cotest_delta"),
        },
        "dataset_action": _dataset_action_from_edge(edge),
    }


def build_parameter_master_card(
    *,
    round_id: int,
    previous_decision: str,
    rollback_streak: int,
    metrics_after: dict | None,
    round_data_stats: dict | None,
    search_dag_nodes: list[dict],
    search_dag_edges: list[dict],
    decision: ParameterMasterDecision,
    top_nodes_k: int = PARAMETER_MASTER_CARD_TOP_NODES,
    top_edges_per_node_k: int = PARAMETER_MASTER_CARD_TOP_EDGES_PER_NODE,
) -> dict[str, Any]:
    """Build the compact Parameter Master evidence card for LLM context.

    The card exposes one retrieval view: top similar nodes, with the best
    outgoing edge(s) nested under each node. It avoids duplicating full DAG,
    dag_query, history_leaf, and action_metadata payloads in the prompt.
    """
    metrics = metrics_after if isinstance(metrics_after, dict) else {}
    data_stats = round_data_stats if isinstance(round_data_stats, dict) else {}
    node_limit = max(0, int(top_nodes_k))
    edge_limit = max(0, int(top_edges_per_node_k))
    nodes_by_id = _node_by_id(search_dag_nodes)
    history_leaf = build_mcts_history_leaf_result(
        search_dag_nodes=search_dag_nodes,
        search_dag_edges=search_dag_edges,
        metrics_after=metrics,
        previous_decision=previous_decision,
        rollback_streak=rollback_streak,
        round_data_stats=data_stats,
        candidate_actions=_candidate_action_space(),
    )

    top_nodes: list[dict[str, Any]] = []
    for similar in history_leaf.similar_nodes[:node_limit]:
        node_id = str(similar.get("node_id") or "")
        node = nodes_by_id.get(node_id)
        if not isinstance(node, dict):
            continue
        edge_cards: list[dict[str, Any]] = []
        for edge in _outgoing_edges(search_dag_edges, node_id):
            child = nodes_by_id.get(str(edge.get("to_node_id") or ""))
            if not isinstance(child, dict):
                continue
            card = _compact_edge_card(edge, node, child)
            if card.get("action_key"):
                edge_cards.append(card)
        edge_cards.sort(key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)
        top_nodes.append({
            "node_id": node_id,
            "similarity": similar.get("similarity"),
            "matched_reasons": similar.get("matched_reasons", []),
            "metrics": _compact_node_metrics(node),
            "top_edges": edge_cards[:edge_limit],
        })

    return {
        "top_nodes_k": node_limit,
        "top_edges_per_node_k": edge_limit,
        "current_state": {
            "round_id": round_id,
            "previous_decision": previous_decision,
            "rollback_streak": rollback_streak,
            "data_pressure": _compact_data_pressure(data_stats),
            "metrics_after": _compact_current_metrics(metrics),
        },
        "deterministic_action": _compact_deterministic_action(decision),
        "top_nodes": top_nodes,
        "action_evidence": history_leaf.action_evidence,
        "constraints": history_leaf.constraints,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  七、数据窗口偏移管理                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# 数据集被切分为多个 shard（分片），每个 shard 内又按 offset+size 取窗口。
# 随着训练轮次推进，系统需要滚动窗口来接触不同的数据子集，避免模型
# 反复训练同一批题目而过拟合。
#
# 关键概念：
#   data_window_size:   每个窗口的题目数
#   data_window_offset: 窗口起始偏移
#   shard_size:         单个分片大小
#   shard_span:         所有分片的总跨度（shard_size × shard_count）
#
# 窗口在 shard_span 内循环滚动，用完了从头来。
# ════════════════════════════════════════════════════════════════════════════════


def _coerce_int(value: object, default: int = 0) -> int:
    """安全的整数类型转换。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  八、参数大师决策 — decide_parameter_master_action() (Agent primary)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Agent 全权决策：换策略、换数据、换窗口均由 agent 决定。
# 本函数提供 slim 确定性 fallback，整合 action_selector + data_window 两个子决策。
#
# 消费方：parameter_master node 调用此函数获取 fallback，
#         然后通过 LLM agent 覆盖 action_key, dataset_mode, window 等。
# ════════════════════════════════════════════════════════════════════════════════


def decide_parameter_master_action(
    round_id: int,
    current_search_node_id: str,
    search_dag_nodes: list[dict],
    search_dag_edges: list[dict],
    previous_decision: str = "",
    rollback_streak: int = 0,
    round_data_stats: dict | None = None,
    edge_summary: dict | None = None,
    data_window_offset: int = 0,
    data_window_size: int = DATASET_SHARD_SIZE,
    metrics_after: dict | None = None,
) -> ParameterMasterDecision:
    """确定性参数大师 fallback。

    整合 action_selector fallback 结果。
    LLM agent 可以通过 decide_json() 覆盖 action_key 等字段。
    """
    action_space = _candidate_action_space()
    data_stats = round_data_stats if isinstance(round_data_stats, dict) else {}
    dag_query = _build_dag_query_result(
        search_dag_nodes=search_dag_nodes,
        search_dag_edges=search_dag_edges,
        metrics_after=metrics_after,
        previous_decision=previous_decision,
        rollback_streak=rollback_streak,
    )
    mcts_history_leaf = build_mcts_history_leaf_result(
        search_dag_nodes=search_dag_nodes,
        search_dag_edges=search_dag_edges,
        metrics_after=metrics_after,
        previous_decision=previous_decision,
        rollback_streak=rollback_streak,
        round_data_stats=data_stats,
        candidate_actions=action_space,
    )

    # 子决策 1: action 选择
    fallback_action = select_action_fallback(
        rollback_streak=rollback_streak,
        previous_decision=previous_decision,
        edge_summary=edge_summary,
    )
    if rollback_streak >= 2:
        allowed_action_keys = [
            key for key in _SAFE_ROLLBACK_ACTION_ORDER
            if key in action_space
        ]
        action_safety_mode = "rollback_safe_subset"
    else:
        allowed_action_keys = list(action_space.keys())
        action_safety_mode = "all_actions"
    if previous_decision in {"rollback", "prune"} and rollback_streak < 2:
        allowed_action_keys = [
            key for key in allowed_action_keys
            if key != fallback_action.action_key
        ] + [fallback_action.action_key]
    forbidden_action_keys = set(mcts_history_leaf.constraints.get("forbidden_action_keys", [])) if mcts_history_leaf.enabled else set()
    if forbidden_action_keys and len(allowed_action_keys) > len(forbidden_action_keys):
        allowed_action_keys = [key for key in allowed_action_keys if key not in forbidden_action_keys]
    combined_action_evidence: dict[str, dict[str, Any]] = {}
    if dag_query.enabled:
        combined_action_evidence.update(dag_query.action_evidence)
    if mcts_history_leaf.enabled:
        for action_key, evidence in mcts_history_leaf.action_evidence.items():
            base = dict(combined_action_evidence.get(action_key, {}))
            base_score = _safe_float(base.get("score"), 0.0)
            leaf_score = _safe_float(evidence.get("score"), 0.0)
            merged_score = max(-MCTS_QUERY_ACTION_BIAS_MAX, min(MCTS_QUERY_ACTION_BIAS_MAX, base_score + leaf_score))
            base.update(evidence)
            base["score"] = merged_score
            base["source"] = "dag_query+mcts_history_leaf" if action_key in combined_action_evidence else "mcts_history_leaf"
            combined_action_evidence[action_key] = base
    continuous_selection: ContinuousParamSelection | None = None
    ucb_selection: ActionUCBSelection | None = None
    if MCTS_CONTINUOUS_ACTION_ENABLED:
        continuous_default_action_key = fallback_action.action_key
        if previous_decision == "prune" and allowed_action_keys:
            continuous_default_action_key = allowed_action_keys[0]
        if combined_action_evidence:
            evidence_rank = sorted(
                (
                    (key, _safe_float(value.get("score"), 0.0))
                    for key, value in combined_action_evidence.items()
                    if key in allowed_action_keys and isinstance(value, dict)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if evidence_rank and evidence_rank[0][1] > 0.0:
                continuous_default_action_key = evidence_rank[0][0]
        continuous_selection = _select_continuous_params_by_surrogate(
            search_dag_edges=search_dag_edges,
            default_action_key=continuous_default_action_key,
            action_keys=allowed_action_keys,
            action_space=action_space,
            action_evidence=combined_action_evidence if combined_action_evidence else None,
            round_id=round_id,
        )
        action_result = ActionSelectionResult(
            action_key=continuous_selection.source_action_key,
            lr_scale=1.0,
            replay_scale=1.0,
            finetuning_type=_finetuning_type_for_action(continuous_selection.source_action_key, action_space),
            reason=f"continuous_surrogate_ucb:{continuous_selection.reason}; fallback={fallback_action.action_key}",
        )
    else:
        ucb_selection = _select_action_key_by_ucb(
            search_dag_edges=search_dag_edges,
            default_action_key=fallback_action.action_key,
            action_keys=allowed_action_keys,
            action_evidence=combined_action_evidence if combined_action_evidence else None,
        )
        action_result = ActionSelectionResult(
            action_key=ucb_selection.action_key,
            lr_scale=fallback_action.lr_scale,
            replay_scale=fallback_action.replay_scale,
            finetuning_type=_finetuning_type_for_action(ucb_selection.action_key, action_space),
            reason=f"mcts_action_ucb:{ucb_selection.reason}; fallback={fallback_action.action_key}",
        )

    forced_diversification = False
    method_retry_same_data = False
    dataset_selection_mode = "single_shard"
    data_pressure_needs_more = (
        bool(data_stats.get("needs_more_data"))
        or bool(data_stats.get("low_train_signal"))
        or bool(data_stats.get("hard_dominated_signal"))
    )
    if previous_decision in {"rollback", "prune"}:
        method_retry_same_data = rollback_streak < 2 and not data_pressure_needs_more
        forced_diversification = True
    force_single_offset_window = (
        DATASET_OFFSET_CACHE_MODE
        or str(DATASET_CACHE_MODE or "").strip().lower() == "offset"
    )
    if rollback_streak >= 2 or data_pressure_needs_more:
        method_retry_same_data = False
        if not (
            force_single_offset_window
            and str(DATASET_SHARD_SELECTION_POLICY or "").strip().lower() == "single"
        ):
            dataset_selection_mode = "merge_shards"
        forced_diversification = True

    current_offset = _coerce_int(data_window_offset, 0)
    window_size = max(1, _coerce_int(data_window_size, DATASET_SHARD_SIZE))
    used_offsets = set()
    for edge in search_dag_edges:
        metadata = edge.get("action_metadata", {}) if isinstance(edge.get("action_metadata"), dict) else {}
        if metadata.get("data_window_offset") is not None:
            used_offsets.add(_coerce_int(metadata.get("data_window_offset"), -1))
    if method_retry_same_data:
        next_offset = current_offset
    elif forced_diversification:
        next_offset = current_offset + window_size
    else:
        next_offset = current_offset
    if not method_retry_same_data:
        while next_offset in used_offsets:
            next_offset += window_size

    # 获取选中 action 的完整参数
    action_key = action_result.action_key
    selected = dict(action_space.get(action_key, action_space["balanced_default"]))
    replay_sample_ratio = float(selected.pop("replay_sample_ratio", 0.30))

    if continuous_selection is not None:
        selected["learning_rate"] = continuous_selection.learning_rate
        selected["num_train_epochs"] = continuous_selection.num_train_epochs
        replay_sample_ratio = continuous_selection.replay_sample_ratio

    # 应用变异缩放
    if continuous_selection is None and action_result.lr_scale != 1.0 and "learning_rate" in selected:
        selected["learning_rate"] = float(selected["learning_rate"]) * action_result.lr_scale
    if continuous_selection is None and action_result.replay_scale != 1.0:
        replay_sample_ratio = min(1.0, replay_sample_ratio * action_result.replay_scale)

    # 分支父节点
    branch_parent_node_id = select_next_parent_node(
        search_dag_nodes,
        fallback_node_id=current_search_node_id,
        candidate_node_ids=dag_query.top_candidate_node_ids if dag_query.enabled else None,
    )

    summary = (
        f"action={action_key} parent={branch_parent_node_id} "
        f"replay={replay_sample_ratio:.2f} lr_scale={action_result.lr_scale}"
    )

    if continuous_selection is not None:
        continuous_params = {
            "source_action_key": continuous_selection.source_action_key,
            "finetuning_type": action_result.finetuning_type,
            "learning_rate": continuous_selection.learning_rate,
            "replay_sample_ratio": replay_sample_ratio,
            "num_train_epochs": continuous_selection.num_train_epochs,
            "normalized_vector": list(continuous_selection.normalized_vector),
        }
        mcts_selector_metadata = {
            "action_reward_basis": "raw_reward_without_gate_bonus",
            "action_selector": "continuous_surrogate_ucb",
            "action_safety_mode": action_safety_mode,
            "selected_action_visits": continuous_selection.effective_samples,
            "selected_action_mean_reward": continuous_selection.predicted_reward,
            "selected_action_reason": continuous_selection.reason,
            "action_ucb_diagnostics": {},
            "continuous_search": continuous_selection.diagnostics,
        }
    else:
        continuous_params = None
        assert ucb_selection is not None
        mcts_selector_metadata = {
            "action_reward_basis": "raw_reward_without_gate_bonus",
            "action_selector": "ucb_with_forced_initial_trials",
            "action_safety_mode": action_safety_mode,
            "selected_action_visits": ucb_selection.visits,
            "selected_action_mean_reward": ucb_selection.mean_reward,
            "selected_action_reason": ucb_selection.reason,
            "action_ucb_diagnostics": ucb_selection.diagnostics,
            "continuous_search": {"enabled": False},
        }

    action_metadata = {
        "action_key": action_key,
        "action_type": "mcts_action",
        "search_space": "continuous" if continuous_selection is not None else "discrete",
        "branch_parent_node_id": branch_parent_node_id,
        "replay_sample_ratio": replay_sample_ratio,
        "training_hyperparams": selected,
        "finetuning_type": action_result.finetuning_type,
        "lr_scale": action_result.lr_scale,
        "replay_scale": action_result.replay_scale,
        "action_reason": action_result.reason,
        "round_id": round_id,
        "rollback_streak": rollback_streak,
        "method_retry_same_data": method_retry_same_data,
        "previous_action_fallback": fallback_action.action_key,
        "diagnostic_mode": "train",
        "data_window_offset": next_offset,
        "data_window_size": window_size,
        "dataset_selection_mode": dataset_selection_mode,
        "forced_diversification": forced_diversification,
        "fresh_dataset_forced": rollback_streak >= 2 or data_pressure_needs_more,
        "data_pressure": data_stats,
        "mcts": {
            **mcts_selector_metadata,
            "dag_query": {
                "enabled": dag_query.enabled,
                "query_type": dag_query.query_type,
                "skipped_reason": dag_query.skipped_reason,
                "threshold": dag_query.threshold,
                "matched_node_count": dag_query.matched_node_count,
                "candidate_top_k": MCTS_QUERY_CANDIDATE_TOP_K,
                "top_candidate_node_ids": dag_query.top_candidate_node_ids,
                "candidate_branches": dag_query.candidate_branches,
                "action_evidence": dag_query.action_evidence,
            },
            "history_leaf": {
                "enabled": mcts_history_leaf.enabled,
                "query_type": mcts_history_leaf.query_type,
                "similar_nodes": mcts_history_leaf.similar_nodes,
                "edge_filtered_evidence": mcts_history_leaf.edge_filtered_evidence,
                "action_evidence": mcts_history_leaf.action_evidence,
                "constraints": mcts_history_leaf.constraints,
                "reason": mcts_history_leaf.reason,
            },
        },
    }
    if continuous_params is not None:
        action_metadata["continuous_params"] = continuous_params
    return ParameterMasterDecision(
        action_key=action_key,
        action_type="mcts_action",
        decision_summary=summary,
        training_hyperparams=selected,
        replay_sample_ratio=replay_sample_ratio,
        branch_parent_node_id=branch_parent_node_id,
        action_metadata=action_metadata,
    )
