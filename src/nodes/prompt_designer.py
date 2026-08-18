# =============================================================================
# prompt_designer.py — 提示词设计师节点
# =============================================================================
# 本节点是系统启动后的第一个策略节点（bootstrap 之后立即执行）。
#
# 核心职责：
#   根据用户输入的目标（如 "improve coding ability"），通过 7 个 leaf LLM
#   子决策器分析出领域目标、能力点、搜索关键词、边界信号等，然后把这些
#   goal 相关信息注入到所有 MUTABLE agent 的 prompt 末尾。
#
# 工作流程：
#   1. 检查 session 目录下的 agent_prompt_pack.json 缓存
#      → 如果存在，直接加载（同一 session 多轮复用）
#   2. 如果不存在：调用 7 个 leaf LLM 分析 goal
#   3. 将分析结果注入 MUTABLE agent 的 prompt 末尾
#   4. 持久化到 agent_prompt_pack.json
#   5. 写入 state.agent_prompt_pack，供后续节点通过 prompt_for_agent() 读取
#
# 设计原则：
#   - 小模型填空：7 个 leaf prompt 各自只填一个 JSON 字段，降低单次调用复杂度
#   - 缓存复用：同一 session 后续轮次不重复调用 LLM，直接加载缓存
#   - 通用设计：不包含任何领域硬编码
# =============================================================================

import json
import os
from pathlib import Path

from config.settings import get_session_dir, get_classifier_labels, IS_CODE_DOMAIN
from src.models.state import EvoState
from src.tools.agent_prompts import (
    DEFAULT_AGENT_PROMPTS,               # 所有 agent 的默认 prompt 注册表
    CODE_AGENT_PROMPTS,                  # 代码域 prompt 注册表
    PROMPT_DESIGNER_LOCKED_AGENTS,       # 不会被注入 goal 信息的 agent（当前为空）
    PROMPT_DESIGNER_MUTABLE_AGENTS,      # 会被注入 goal 信息的 agent（所有策略 agent）
    PROMPT_DESIGNER_DOMAIN_GOAL_PROMPT,  # leaf 1: 领域目标
    PROMPT_DESIGNER_CAPABILITIES_PROMPT, # leaf 2: 能力点列表
    PROMPT_DESIGNER_KEYWORDS_PROMPT,     # leaf 3: HF 搜索关键词
    PROMPT_DESIGNER_BOUNDARY_PROMPT,     # leaf 4: 边界信号
    PROMPT_DESIGNER_RARE_PROMPT,         # leaf 5: 稀有/长尾信号
    PROMPT_DESIGNER_LABELS_PROMPT,       # leaf 6+7: 分类标签 + 标签说明
    INSTRUCTION_DESIGNER_PROMPT,         # leaf 8: 指令前缀设计
    get_agent_prompt,                    # 域感知的 prompt 选择
)
from src.tools.llm_decision import decide_json_leaf


# ── _fallback_prompt_design ──────────────────────────────────────────────────
# 确定性 fallback：当 USE_LLM_AGENTS=False 或 LLM 全部失败时，用此函数生成
# 默认的 goal 分析结果。不做任何领域翻译，直接使用原始 goal 文本。
# ──────────────────────────────────────────────────────────────────────────────
def _fallback_prompt_design(goal: str) -> dict:
    goal_text = (goal).strip()
    return {
        "domain_goal": goal_text,
        "target_capabilities": [
            f"core competency in {goal_text}",
            "accuracy and precision",
            "robustness across varied inputs",
            "resistance to common failure modes",
        ],
        "search_keywords": [
            goal_text,
            f"{goal_text} dataset",
            f"{goal_text} benchmark",
            f"{goal_text} training data",
        ],
        "boundary_signals": [
            "medium difficulty accuracy below target",
            "hard difficulty dominates failures",
            "candidate gains do not transfer to fixed benchmark",
        ],
        "rare_signals": [
            "low-frequency input patterns in failures",
            "formats absent from recent training data",
            "edge cases with misleading patterns",
        ],
        "classifier_labels": get_classifier_labels(),
        "classifier_label_notes": {},
        "instruction_prefix": "请解答下面的题目。",
    }


# ── _clean_string_array ──────────────────────────────────────────────────────
# 清洗 LLM 输出的字符串数组：截断长度 + 限制条目数。
# ──────────────────────────────────────────────────────────────────────────────
def _clean_string_array(raw, max_items=10, max_len=100):
    if isinstance(raw, list):
        return [str(item)[:max_len] for item in raw[:max_items]]
    return []


# ── _apply_leaf_design_decisions ─────────────────────────────────────────────
# 核心：顺序调用 7 个 leaf LLM 子决策器，分析用户 goal。
#
# 调用顺序及依赖：
#   ① domain_goal     — 先决定领域目标（后续 leaf 都需要它作为 context）
#   ② capabilities    — 基于 domain_goal 列出能力点
#   ③ keywords        — 基于 domain_goal 生成搜索关键词
#   ④ boundary        — 基于 domain_goal 列出边界信号
#   ⑤ rare            — 基于 domain_goal 列出稀有信号
#   ⑥ labels          — 生成分类标签列表
#   ⑦ label_notes     — 为每个分类标签写说明（依赖⑥的输出）
#
# 每个 leaf 调用 decide_json_leaf()：只填一个 JSON 字段，解析失败自动 fallback。
# ──────────────────────────────────────────────────────────────────────────────
def _apply_leaf_design_decisions(state: EvoState, fallback: dict, round_id: int) -> tuple[dict, list[str]]:
    trace_id = state.get("trace_id", "")
    goal = state.get("user_goal", "")
    successes: list[str] = []   # 记录哪些字段被 LLM 成功覆盖
    result = dict(fallback)

    # ── leaf ①: 领域目标（英文短语，10 词以内） ──
    raw_goal, ok = decide_json_leaf(
        agent_name="prompt_designer.domain_goal",
        prompt=get_agent_prompt("prompt_designer.domain_goal"),
        context={"user_goal": goal, "field_name": "domain_goal"},
        field_name="domain_goal",
        fallback_value=fallback["domain_goal"],
        trace_id=trace_id, round_id=round_id,
    )
    domain_goal = str(raw_goal or fallback["domain_goal"]).strip()[:120]
    if ok and domain_goal != fallback["domain_goal"]:
        successes.append("domain_goal")
    result["domain_goal"] = domain_goal

    # ── leaf ②: 能力点列表（3~6 个英文能力点） ──
    raw_caps, ok = decide_json_leaf(
        agent_name="prompt_designer.capabilities",
        prompt=get_agent_prompt("prompt_designer.capabilities"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "target_capabilities"},
        field_name="target_capabilities",
        fallback_value=fallback["target_capabilities"],
        trace_id=trace_id, round_id=round_id,
    )
    caps = _clean_string_array(raw_caps, max_items=8)
    if caps:
        result["target_capabilities"] = caps
        successes.append("target_capabilities")

    # ── leaf ③: HF 搜索关键词（5~10 个英文关键词） ──
    raw_kw, ok = decide_json_leaf(
        agent_name="prompt_designer.keywords",
        prompt=get_agent_prompt("prompt_designer.keywords"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "search_keywords"},
        field_name="search_keywords",
        fallback_value=fallback["search_keywords"],
        trace_id=trace_id, round_id=round_id,
    )
    keywords = _clean_string_array(raw_kw, max_items=12)
    if keywords:
        result["search_keywords"] = keywords
        successes.append("search_keywords")

    # ── leaf ④: 边界信号（何时需要调整策略） ──
    raw_b, ok = decide_json_leaf(
        agent_name="prompt_designer.boundary",
        prompt=get_agent_prompt("prompt_designer.boundary"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "boundary_signals"},
        field_name="boundary_signals",
        fallback_value=fallback["boundary_signals"],
        trace_id=trace_id, round_id=round_id,
    )
    boundary = _clean_string_array(raw_b, max_items=8, max_len=120)
    if boundary:
        result["boundary_signals"] = boundary
        successes.append("boundary_signals")

    # ── leaf ⑤: 稀有/长尾信号 ──
    raw_r, ok = decide_json_leaf(
        agent_name="prompt_designer.rare",
        prompt=get_agent_prompt("prompt_designer.rare"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "rare_signals"},
        field_name="rare_signals",
        fallback_value=fallback["rare_signals"],
        trace_id=trace_id, round_id=round_id,
    )
    rare = _clean_string_array(raw_r, max_items=8, max_len=120)
    if rare:
        result["rare_signals"] = rare
        successes.append("rare_signals")

    # ── leaf ⑥: 分类标签列表 ──
    raw_labels, ok = decide_json_leaf(
        agent_name="prompt_designer.labels",
        prompt=get_agent_prompt("prompt_designer.labels"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "classifier_labels"},
        field_name="classifier_labels",
        fallback_value=fallback["classifier_labels"],
        trace_id=trace_id, round_id=round_id,
    )
    labels = _clean_string_array(raw_labels, max_items=12, max_len=40)
    if labels:
        result["classifier_labels"] = labels
        successes.append("classifier_labels")

    # ── leaf ⑦: 分类标签说明（依赖⑥的输出） ──
    raw_notes, ok = decide_json_leaf(
        agent_name="prompt_designer.labels",
        prompt=get_agent_prompt("prompt_designer.labels"),
        context={"user_goal": goal, "domain_goal": domain_goal, "classifier_labels": result.get("classifier_labels", fallback["classifier_labels"]), "field_name": "classifier_label_notes"},
        field_name="classifier_label_notes",
        fallback_value=fallback["classifier_label_notes"],
        trace_id=trace_id, round_id=round_id,
    )
    if isinstance(raw_notes, dict) and raw_notes:
        result["classifier_label_notes"] = {str(k)[:40]: str(v)[:200] for k, v in raw_notes.items()}
        successes.append("classifier_label_notes")

    # ── leaf ⑧: 指令前缀（基于领域目标动态生成） ──
    raw_prefix, ok = decide_json_leaf(
        agent_name="instruction_designer",
        prompt=get_agent_prompt("instruction_designer"),
        context={"user_goal": goal, "domain_goal": domain_goal, "field_name": "instruction_prefix"},
        field_name="instruction_prefix",
        fallback_value=fallback["instruction_prefix"],
        trace_id=trace_id, round_id=round_id,
    )
    instruction_prefix = str(raw_prefix or fallback["instruction_prefix"]).strip()[:40]
    if ok and instruction_prefix != fallback["instruction_prefix"]:
        successes.append("instruction_prefix")
    result["instruction_prefix"] = instruction_prefix

    return result, successes


# ── _compose_prompt_pack ─────────────────────────────────────────────────────
# 将 goal 分析结果注入到所有 MUTABLE agent 的 prompt 末尾。
#
# 注入格式：在每个 MUTABLE agent 的 prompt 后追加一段结构化文本：
#   """
#   本轮训练目标槽位:
#   - domain_goal: ...
#   - target_capabilities: [...]
#   - search_keywords: [...]
#   - boundary_signals: [...]
#   - rare_signals: [...]
#   """
# 这样后续节点通过 prompt_for_agent() 获取 prompt 时，就能自动带上 goal 信息。
#
# LOCKED agent 的 prompt 保持不变（当前 LOCKED 为空，所有 agent 都注入）。
# ──────────────────────────────────────────────────────────────────────────────
def _compose_prompt_pack(design: dict) -> dict:
    # Domain-aware base: code domain uses code prompts so mutable agents steer
    # toward code-appropriate decisions (tested data, execution judging, LoRA).
    try:
        from config import settings as _settings
        _base = CODE_AGENT_PROMPTS if getattr(_settings, "IS_CODE_DOMAIN", False) else DEFAULT_AGENT_PROMPTS
    except Exception:
        _base = DEFAULT_AGENT_PROMPTS
    pack = {
        "version": 1,
        "prompt_designer": design,
        "instruction_prefix": design.get("instruction_prefix", "请解答下面的题目。"),
        "locked_agents": sorted(PROMPT_DESIGNER_LOCKED_AGENTS),
        "mutable_agents": sorted(PROMPT_DESIGNER_MUTABLE_AGENTS),
        "prompts": dict(_base),
    }
    # 组装要注入的 goal 信息文本块
    target_line = (
        "\n\n本轮训练目标槽位:\n"
        f"- domain_goal: {design.get('domain_goal', '')}\n"
        f"- target_capabilities: {json.dumps(design.get('target_capabilities', []), ensure_ascii=False)}\n"
        f"- search_keywords: {json.dumps(design.get('search_keywords', []), ensure_ascii=False)}\n"
        f"- boundary_signals: {json.dumps(design.get('boundary_signals', []), ensure_ascii=False)}\n"
        f"- rare_signals: {json.dumps(design.get('rare_signals', []), ensure_ascii=False)}\n"
    )
    # 遍历所有 MUTABLE agent，在其默认 prompt 末尾追加 goal 信息
    for agent in PROMPT_DESIGNER_MUTABLE_AGENTS:
        base_prompt = pack["prompts"].get(agent, "")
        if base_prompt:
            pack["prompts"][agent] = base_prompt.rstrip() + target_line
    return pack


# ── prompt_designer_node ─────────────────────────────────────────────────────
# LangGraph 节点入口。在 bootstrap 之后立即执行。
#
# 流程：
#   1. 检查 agent_prompt_pack.json 缓存 → 命中则直接返回（复用）
#   2. 缓存未命中 → 调 8 个 leaf LLM 分析 goal → 组装 pack
#   3. 持久化到 agent_prompt_pack.json
#   4. 设置 CLASSIFIER_TARGET_LABELS 环境变量（供 classifier 使用）
#   5. 写回 state.agent_prompt_pack，后续节点通过 prompt_for_agent() 读取
# ──────────────────────────────────────────────────────────────────────────────
def prompt_designer_node(state: EvoState) -> dict:
    trace_id = str(state.get("trace_id") or "")
    round_id = int(state.get("round_id", 0) or 0)
    session_dir = get_session_dir(trace_id)
    prompt_pack_path = session_dir / "agent_prompt_pack.json"

    # ── 缓存命中：直接加载已有 pack，不重复调 LLM ──
    if prompt_pack_path.exists():
        try:
            loaded = json.loads(prompt_pack_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("prompts"):
                cached_prefix = loaded.get("instruction_prefix", "").strip()
                if cached_prefix:
                    os.environ["INSTRUCTION_PREFIX"] = cached_prefix
                return {
                    "agent_prompt_pack": loaded,
                    "agent_prompt_pack_path": str(prompt_pack_path),
                }
        except (OSError, json.JSONDecodeError):
            pass  # 缓存损坏，重新生成

    # ── 缓存未命中：调 8 个 leaf LLM 分析 goal ──
    fallback = _fallback_prompt_design(state.get("user_goal", ""))
    design, successes = _apply_leaf_design_decisions(state, fallback, round_id)
    if successes:
        print(f"[prompt_designer] Leaf LLM filled slots: {successes}")

    # ── 用 LLM 结果覆盖 fallback 默认值 ──
    merged_design = {**fallback, **{k: v for k, v in design.items() if v}}
    # ── 组装 prompt pack 并注入到 MUTABLE agent ──
    pack = _compose_prompt_pack(merged_design)
    Path(prompt_pack_path).write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[prompt_designer] Built agent prompt pack "
        f"path={prompt_pack_path} domain_goal={merged_design.get('domain_goal', '')}"
    )

    # ── 设置分类器标签环境变量 ──
    labels = merged_design.get("classifier_labels", get_classifier_labels())
    if labels:
        os.environ["CLASSIFIER_TARGET_LABELS"] = ",".join(labels)
        print(f"[prompt_designer] Set classifier labels: {labels}")

    # ── 设置指令前缀环境变量（动态覆盖静态默认值） ──
    # Code domain: keep the env's code-specific instruction prefix (from
    # run_code.sh) — the prompt_designer's short phrase is not a good code
    # generation instruction. Math domain: override as before.
    instruction_prefix = merged_design.get("instruction_prefix", "").strip()
    if instruction_prefix and not IS_CODE_DOMAIN:
        os.environ["INSTRUCTION_PREFIX"] = instruction_prefix
        print(f"[prompt_designer] Set instruction prefix: {instruction_prefix}")
    elif IS_CODE_DOMAIN:
        print(f"[prompt_designer] Code domain: keeping env instruction prefix: "
              f"{os.environ.get('INSTRUCTION_PREFIX', '')[:80]}")

    return {
        "agent_prompt_pack": pack,
        "agent_prompt_pack_path": str(prompt_pack_path),
    }
