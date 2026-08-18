import json
import math
import random
from pathlib import Path
from collections import Counter
from typing import cast

from config.settings import (
    COTEST_SPLIT_RATIO,
    DOMAIN,
    HOLDOUT_EVAL_SEED,
    HOLDOUT_EVAL_SIZE,
    BASE_MODEL_NAME,
    LF_VAL_MAX_QUESTIONS,
    LF_VAL_MIN_QUESTIONS,
    LF_VAL_MIN_TRAIN_REMAINING,
    LF_VAL_SPLIT_RATIO,
    PROBE_POOL_INTAKE_RATIO,
    PROBE_SPLIT_RATIO,
    DATA_HARD_RATIO_MERGE_THRESHOLD,
    DATA_MIN_TRAIN_QUESTIONS_PER_ROUND,
    TEST_ROLLOUT_TIMES,
    TEST_SPLIT_RATIO,
    TRAIN_LF_EVAL_ENABLED,
    TRAIN_SPLIT_RATIO,
    get_session_dir,
)
from src.models.messages import (
    AgentName,
    ClassificationResultPayload,
    DatasetBundlePayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.agent_prompts import (
    DATA_BUILDER_DIFFICULTY_WEIGHTS_PROMPT,
    DATA_BUILDER_MODULE_WEIGHTS_PROMPT,
    DATA_BUILDER_REPLAY_RATIO_PROMPT,
)
from src.tools.llm_decision import clamp_distribution, decide_json_leaf
from src.tools.question_registry import (
    TEST_BLOCKING_STATUSES,
    drop_registered_questions,
    mark_questions_active_holdout,
    question_text_hash,
)
from src.tools.message_artifacts import load_payload_list
from src.tools.question_fields import TARGET_STYLE_COT, infer_target_style, normalize_target_style
from src.tools.cot_format import build_train_output, detect_model_chat_config, strip_answer_marker_tail
from src.tools.model_runner import _extract_final_answer
from src.tools.dataset_state import DatasetStateManager
from src.tools.data_builder_helpers import (
    decide_data_pipeline_plan,
    pack_existing_splits_for_trainer,
    resource_overrides_from_plan,
    write_data_pipeline_log,
)

SPLIT_TAG_TRAIN = 1
SPLIT_TAG_COTEST = 2
SPLIT_TAG_TEST = 3
_DYNAMIC_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}


def _safe_float(raw: object, fallback: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        if isinstance(raw, int | float | str):
            value = float(raw)
        else:
            return fallback
    except (TypeError, ValueError):
        return fallback
    return min(high, max(low, value))


def _quota_managed_input(state: EvoState) -> bool:
    """Filter already enforced difficulty quotas before handing data over."""
    if state.get("quota_met") or state.get("data_replenishment_exhausted"):
        return True
    round_stats = state.get("round_data_stats") or {}
    return bool(
        isinstance(round_stats, dict)
        and (
            round_stats.get("quota_pool_target") is not None
            or round_stats.get("quota_selected_so_far") is not None
        )
    )


def _source_row_identity(q: dict) -> tuple[str, str, str, str] | None:
    if not isinstance(q, dict):
        return None
    dataset_id = str(q.get("source_dataset_id") or "")
    row_id = str(q.get("source_dataset_row_id") or "")
    if not dataset_id or not row_id:
        return None
    return (
        dataset_id,
        str(q.get("source_dataset_subset") or ""),
        str(q.get("source_dataset_split") or ""),
        row_id,
    )


def _dedupe_key(q: dict) -> tuple[str, object]:
    row_identity = _source_row_identity(q)
    if row_identity is not None:
        return ("source_row", row_identity)
    text_hash = question_text_hash(q) if isinstance(q, dict) and q.get("question_text", "") else ""
    if text_hash:
        return ("text_hash", text_hash)
    qid = str(q.get("question_id") or "") if isinstance(q, dict) else ""
    if qid:
        return ("question_id", qid)
    return ("text_hash", text_hash)


def _overlap_keys(q: dict) -> set[tuple[str, object]]:
    keys: set[tuple[str, object]] = set()
    row_identity = _source_row_identity(q)
    if row_identity is not None:
        keys.add(("source_row", row_identity))
        qid = str(q.get("question_id") or "") if isinstance(q, dict) else ""
        if qid:
            keys.add(("question_id", qid))
    else:
        text_hash = question_text_hash(q) if isinstance(q, dict) and q.get("question_text", "") else ""
        if text_hash:
            keys.add(("text_hash", text_hash))
        else:
            qid = str(q.get("question_id") or "") if isinstance(q, dict) else ""
            if qid:
                keys.add(("question_id", qid))
    return keys


def _key_sets(questions: list[dict]) -> set[tuple[str, object]]:
    keys: set[tuple[str, object]] = set()
    for q in questions:
        if not isinstance(q, dict):
            continue
        keys.update(_overlap_keys(q))
    return keys


def _question_key(q: dict) -> tuple[str, object]:
    return _dedupe_key(q)


def _drop_keyed_questions(questions: list[dict], keys: set[tuple[str, object]]) -> list[dict]:
    kept = []
    for q in questions:
        if _question_key(q) in keys:
            continue
        kept.append(q)
    return kept


def _drop_overlapping_questions(questions: list[dict], reserved: list[dict]) -> list[dict]:
    reserved_keys = _key_sets(reserved)
    kept = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if _overlap_keys(q) & reserved_keys:
            continue
        kept.append(q)
    return kept


def _dedupe_questions(questions: list[dict]) -> list[dict]:
    seen_keys: set[tuple[str, object]] = set()
    kept = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        keys = _overlap_keys(q)
        if keys & seen_keys:
            continue
        kept.append(q)
        seen_keys.update(keys)
    return kept


def _normalized_weights(raw: dict | None, allowed: set[str] | None = None) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        if allowed is not None and str(key) not in allowed:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            cleaned[str(key)] = weight
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _allocate_counts(weights: dict[str, float], total: int, available: dict[str, int]) -> dict[str, int]:
    usable = {key: weight for key, weight in weights.items() if available.get(key, 0) > 0}
    if total <= 0 or not usable:
        return {}
    normalized = _normalized_weights(usable)
    quotas = {
        key: min(available.get(key, 0), int(total * weight))
        for key, weight in normalized.items()
    }
    remaining = total - sum(quotas.values())
    order = sorted(
        normalized,
        key=lambda key: (total * normalized[key]) - int(total * normalized[key]),
        reverse=True,
    )
    while remaining > 0 and order:
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if quotas.get(key, 0) >= available.get(key, 0):
                continue
            quotas[key] = quotas.get(key, 0) + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _allocate_counts_with_hard_cap(
    weights: dict[str, float],
    total: int,
    available: dict[str, int],
) -> dict[str, int]:
    """Respect curriculum hard caps when easy/medium supply is sparse.

    The generic allocator redistributes missing easy/medium quota into hard.
    That is useful for fixed-size eval splits, but harmful for training
    curriculum: if the post-rollout pool is hard-dominated, it turns a
    medium-heavy plan back into a hard-only batch.
    """
    if total <= 0:
        return {}
    normalized = _normalized_weights(weights)
    if not normalized:
        return {}

    quotas = {
        key: min(available.get(key, 0), int(total * weight))
        for key, weight in normalized.items()
        if available.get(key, 0) > 0
    }
    hard_weight = float(normalized.get("hard", 0.0) or 0.0)
    hard_cap = total if hard_weight >= 0.25 else min(
        available.get("hard", 0),
        max(int(math.ceil(total * hard_weight)), int(math.ceil(total * 0.15))),
    )

    fractional_order = sorted(
        quotas,
        key=lambda key: (total * normalized[key]) - int(total * normalized[key]),
        reverse=True,
    )
    remaining = total - sum(quotas.values())
    while remaining > 0 and fractional_order:
        progressed = False
        for key in fractional_order:
            if remaining <= 0:
                break
            if quotas.get(key, 0) >= available.get(key, 0):
                continue
            if key == "hard" and quotas.get(key, 0) >= hard_cap:
                continue
            quotas[key] = quotas.get(key, 0) + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return {key: count for key, count in quotas.items() if count > 0}


def _take_unique(source: list[dict], count: int, used_ids: set[tuple[str, object]]) -> list[dict]:
    if count <= 0:
        return []
    selected = []
    for q in source:
        keys = _overlap_keys(q)
        if keys & used_ids:
            continue
        selected.append(q)
        used_ids.update(keys)
        if len(selected) >= count:
            break
    return selected


def select_questions_by_sampling_plan(
    questions: list[dict],
    sampling_plan: dict | None,
    target_count: int | None = None,
    already_balanced: bool = False,
) -> list[dict]:
    count = min(len(questions), max(0, int(target_count or len(questions))))
    if already_balanced:
        return questions[:count]

    plan = sampling_plan or {}
    difficulty_weights = _normalized_weights(
        plan.get("difficulty_weights"),
        allowed=_DYNAMIC_DIFFICULTIES,
    )
    if not difficulty_weights:
        return questions

    module_weights = _normalized_weights(plan.get("module_weights"))
    by_difficulty: dict[str, list[dict]] = {}
    for q in questions:
        difficulty = str(q.get("dynamic_difficulty") or "unknown")
        by_difficulty.setdefault(difficulty, []).append(q)

    difficulty_available = {
        difficulty: len(by_difficulty.get(difficulty, []))
        for difficulty in difficulty_weights
    }
    difficulty_quotas = _allocate_counts_with_hard_cap(
        difficulty_weights,
        count,
        difficulty_available,
    )

    selected: list[dict] = []
    used_ids: set[tuple[str, object]] = set()
    for difficulty, difficulty_quota in difficulty_quotas.items():
        pool = by_difficulty.get(difficulty, [])
        picked_count = 0
        if module_weights:
            by_module: dict[str, list[dict]] = {}
            for q in pool:
                by_module.setdefault(q.get("module") or q.get("category", "unknown"), []).append(q)
            module_available = {
                module: len(by_module.get(module, []))
                for module in module_weights
            }
            module_quotas = _allocate_counts(module_weights, difficulty_quota, module_available)
            for module, module_quota in module_quotas.items():
                picked = _take_unique(by_module.get(module, []), module_quota, used_ids)
                selected.extend(picked)
                picked_count += len(picked)
        selected.extend(_take_unique(pool, difficulty_quota - picked_count, used_ids))

    selected = selected[:count]
    print(
        "[data_builder] Plan selection: "
        f"difficulty_weights={difficulty_weights}, module_weights={module_weights}, "
        f"selected_count={len(selected)}/{count}, "
        f"selected_difficulty={dict(Counter(q.get('dynamic_difficulty', 'unknown') for q in selected))}, "
        f"selected_modules={dict(Counter(q.get('module') or q.get('category', 'unknown') for q in selected))}"
    )
    return selected


def carve_lf_val_from_train(train_questions: list[dict]) -> tuple[list[dict], list[dict]]:
    lf_val_target = int(len(train_questions) * LF_VAL_SPLIT_RATIO)
    lf_val_count = min(LF_VAL_MAX_QUESTIONS, max(LF_VAL_MIN_QUESTIONS, lf_val_target))
    if (
        not TRAIN_LF_EVAL_ENABLED
        or len(train_questions) < LF_VAL_MIN_TRAIN_REMAINING
        or len(train_questions) - lf_val_count < LF_VAL_MIN_TRAIN_REMAINING
    ):
        return train_questions, []
    return train_questions[lf_val_count:], train_questions[:lf_val_count]


def stratified_split_by_difficulty_module(
    classified_questions: list[dict],
    train_ratio: float,
    cotest_ratio: float,
    test_ratio: float,
    probe_ratio: float = 0.0,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for q in classified_questions:
        difficulty = str(q.get("dynamic_difficulty") or "unknown")
        module = str(q.get("module") or q.get("category", "unknown") or "unknown")
        grouped.setdefault((difficulty, module), []).append(q)

    train_items, cotest_items, test_items, probe_items = [], [], [], []
    for key in list(grouped.keys()):
        items = grouped[key]
        n = len(items)
        ratio_sum = train_ratio + cotest_ratio + test_ratio + probe_ratio
        if ratio_sum <= 0:
            ratio_sum = 1.0
        ratios = [
            ("train", train_ratio),
            ("cotest", cotest_ratio),
            ("test", test_ratio),
            ("probe", probe_ratio),
        ]
        quotas = {name: int(n * ratio / ratio_sum) for name, ratio in ratios}
        fractions = {
            name: (n * ratio / ratio_sum) - quotas[name]
            for name, ratio in ratios
        }
        positive_names = [name for name, ratio in ratios if ratio > 0]
        if n >= len(positive_names):
            for name in positive_names:
                if quotas[name] == 0:
                    quotas[name] = 1
        remaining = n - sum(quotas.values())
        if remaining > 0:
            for name, _ratio in sorted(fractions.items(), key=lambda item: item[1], reverse=True):
                if remaining <= 0:
                    break
                quotas[name] += 1
                remaining -= 1
        elif remaining < 0:
            for name, _ratio in sorted(fractions.items(), key=lambda item: item[1]):
                if remaining >= 0:
                    break
                if quotas[name] > 0:
                    quotas[name] -= 1
                    remaining += 1

        n_train = quotas["train"]
        n_cotest = quotas["cotest"]
        n_test = quotas["test"]
        n_probe = quotas["probe"]

        train_items.extend(items[:n_train])
        cotest_items.extend(items[n_train : n_train + n_cotest])
        test_items.extend(items[n_train + n_cotest : n_train + n_cotest + n_test])
        probe_items.extend(items[n_train + n_cotest + n_test : n_train + n_cotest + n_test + n_probe])

    return train_items, cotest_items, test_items, probe_items


def stratified_split_by_category(
    classified_questions: list[dict],
    train_ratio: float,
    cotest_ratio: float,
    test_ratio: float,
    probe_ratio: float = 0.0,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    return stratified_split_by_difficulty_module(
        classified_questions,
        train_ratio,
        cotest_ratio,
        test_ratio,
        probe_ratio,
    )


def load_mastered_memory_sample(mastered_path: str, max_samples: int = 50) -> list[dict]:
    if not mastered_path or not Path(mastered_path).exists():
        return []

    with open(mastered_path, "r", encoding="utf-8") as f:
        mastered = json.load(f)

    questions = []
    if isinstance(mastered, dict):
        for v in mastered.values():
            if isinstance(v, list):
                questions.extend(v)
    elif isinstance(mastered, list):
        questions = mastered

    return questions[:max_samples]


def _tag_question(q: dict, split_tag: int, split_role: str) -> dict:
    tagged = dict(q)
    tagged["split_tag"] = split_tag
    tagged["split_role"] = split_role
    return tagged


def _record_output_for_split(
    q: dict,
    split_tag: int,
    delimiters: tuple[str, str] | None = None,
) -> str:
    gold_answer = str(q.get("gold_answer", ""))
    rollout_gold_answer = str(q.get("rollout_gold_answer") or gold_answer)
    train_output = str(q.get("train_output") or "")
    evaluation_method = str(q.get("evaluation_method") or "").strip().lower()
    if split_tag == SPLIT_TAG_TRAIN and evaluation_method == "llm_judge" and not rollout_gold_answer:
        return train_output or str(q.get("process") or q.get("think") or "")
    target_style = normalize_target_style(q.get("target_style")) or infer_target_style(
        train_output,
        rollout_gold_answer,
        gold_answer,
    )
    if split_tag == SPLIT_TAG_TRAIN and target_style == TARGET_STYLE_COT:
        neutral_output = _neutral_reasoning_output_for_train(q, rollout_gold_answer or gold_answer, delimiters)
        if neutral_output is not None:
            return neutral_output
        marker = None
        schema = q.get("source_dataset_schema")
        if isinstance(schema, dict):
            raw_marker = schema.get("final_answer_marker")
            if isinstance(raw_marker, str) and raw_marker.strip():
                marker = raw_marker.strip()
        gold_blob = rollout_gold_answer or gold_answer
        final_answer = _extract_final_answer(gold_blob, marker) or gold_blob
        reasoning = strip_answer_marker_tail(train_output or gold_blob, marker)
        return build_train_output(
            reasoning,
            final_answer,
            delimiters=delimiters,
            force_box=True,
        )
    return rollout_gold_answer or gold_answer


def _neutral_reasoning_output_for_train(
    q: dict,
    gold_blob: str,
    delimiters: tuple[str, str] | None,
) -> str | None:
    if "process" not in q and "think" not in q:
        return None

    process = str(q.get("process") or "").strip()
    think = str(q.get("think") or "").strip()
    if not process and not think:
        return None

    if not gold_blob:
        parts: list[str] = []
        if think:
            open_tag, close_tag = delimiters or ("<think>", "</think>")
            parts.append(f"{open_tag}{think}{close_tag}")
        if process:
            parts.append(process)
        return "\n\n".join(parts)

    final_answer = _extract_final_answer(gold_blob, None) or gold_blob
    parts: list[str] = []
    if think:
        open_tag, close_tag = delimiters or ("<think>", "</think>")
        parts.append(f"{open_tag}{think}{close_tag}")
    if process:
        parts.append(process)
    return build_train_output(
        "\n\n".join(parts),
        final_answer,
        delimiters=None,
        force_box=True,
    )


def _optional_alpaca_field(q: dict, field: str):
    value = q.get(field)
    if value in (None, ""):
        return None
    return value


def _alpaca_metadata(q: dict, split_tag: int, split_role: str) -> dict:
    train_output = str(q.get("train_output") or q.get("rollout_gold_answer") or q.get("gold_answer") or "")
    return {
        "split_tag": split_tag,
        "split_role": split_role,
        "question_id": q.get("question_id", ""),
        "gold_answer": q.get("gold_answer", ""),
        "rollout_gold_answer": q.get("rollout_gold_answer") or q.get("gold_answer", ""),
        "train_output": train_output,
        "process": q.get("process", ""),
        "think": q.get("think", ""),
        "evaluation_method": q.get("evaluation_method", "gold"),
        "needs_judge": bool(q.get("needs_judge", False)),
        # Code-domain fields: carry the executable test + entry_point through to
        # the eval/test JSONL so the evaluator can judge by execution.
        "test": q.get("test") or q.get("code_test") or "",
        "entry_point": q.get("entry_point") or q.get("entry_point_func") or "",
        "target_style": infer_target_style(
            train_output,
            q.get("rollout_gold_answer", ""),
            q.get("gold_answer", ""),
            explicit=q.get("target_style"),
        ),
        "module": q.get("module") or q.get("category", "unknown"),
        "dynamic_difficulty": q.get("dynamic_difficulty", "unknown"),
        "pass_count": q.get("pass_count"),
        "rollout_count": q.get("rollout_count"),
        "pass_rate": q.get("pass_rate"),
        "source_role": q.get("source_role"),
        "source_dataset_id": q.get("source_dataset_id"),
        "source_dataset_row_id": q.get("source_dataset_row_id"),
        "source_dataset_split": q.get("source_dataset_split"),
        "source_dataset_subset": q.get("source_dataset_subset"),
        "source_dataset_requested_split": q.get("source_dataset_requested_split"),
        "source_dataset_split_names": q.get("source_dataset_split_names") or [],
        "source_dataset_columns": q.get("source_dataset_columns") or [],
        "source_dataset_first_row": q.get("source_dataset_first_row") or {},
        "source_dataset_schema": q.get("source_dataset_schema") or {},
        "replay_use_count": q.get("replay_use_count"),
        "category": q.get("category", q.get("module", "unknown")),
    }


def to_alpaca_record(
    q: dict,
    split_tag: int,
    split_role: str,
    prompt_template: str | None = None,
    *,
    include_metadata: bool = True,
    delimiters: tuple[str, str] | None = None,
) -> dict:
    import os
    if DOMAIN == "code":
        # Code domain: use the code instruction prefix (set by prompt_designer)
        # so SFT training matches the rollout/eval prompt format exactly.
        instruction = os.environ.get(
            "INSTRUCTION_PREFIX",
            "Write a Python function that solves the following problem. Output only the code, no explanation.",
        ).rstrip("\n")
        input_text = q["question_text"]
    else:
        instruction_prefix = os.environ.get("INSTRUCTION_PREFIX", "请解答下面的题目,并在最后将最终的数值答案写在 \\boxed{} 中,例如 \\boxed{42}。")
        instruction = (prompt_template or instruction_prefix).rstrip("\n") or "请解答下面的题目,并在最后将最终的数值答案写在 \\boxed{} 中,例如 \\boxed{42}。"
        input_text = q["question_text"]
        if prompt_template:
            try:
                rendered = prompt_template.format(**q)
                if rendered:
                    instruction = rendered
                    if q["question_text"].strip() in rendered.strip():
                        input_text = ""
            except (KeyError, ValueError):
                pass
    record = {
        "instruction": instruction,
        "input": input_text,
        "output": _record_output_for_split(q, split_tag, delimiters),
        "system": "",
        "history": [],
    }
    system = _optional_alpaca_field(q, "system")
    history = _optional_alpaca_field(q, "history")
    if system is not None:
        record["system"] = system
    if history is not None:
        record["history"] = history
    if include_metadata:
        record.update(_alpaca_metadata(q, split_tag, split_role))
    return record


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _write_json_list(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _fresh_question_ids_by_dataset(fresh_questions: list[dict]) -> dict[str, list[str]]:
    by_dataset: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for question in fresh_questions:
        if not isinstance(question, dict) or question.get("source_role") == "replay":
            continue
        dataset_id = str(question.get("source_dataset_id") or "")
        question_id = str(question.get("question_id") or "")
        if not dataset_id or not question_id:
            continue
        dataset_seen = seen.setdefault(dataset_id, set())
        if question_id in dataset_seen:
            continue
        dataset_seen.add(question_id)
        by_dataset.setdefault(dataset_id, []).append(question_id)
    return by_dataset


def _question_ids(questions: list[dict]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "")
        if not question_id or question_id in seen:
            continue
        ids.append(question_id)
        seen.add(question_id)
    return ids


def _reserve_fresh_questions(state: EvoState, fresh_questions: list[dict]) -> dict[str, list[str]]:
    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return {}
    by_dataset = _fresh_question_ids_by_dataset(fresh_questions)
    if not by_dataset:
        return {}
    mgr = DatasetStateManager(Path(cache_path))
    for dataset_id, question_ids in by_dataset.items():
        if mgr.load_cached(dataset_id) is None:
            mgr.init_dataset(dataset_id, question_ids)
        mgr.mark_reserved(dataset_id, question_ids)
    return by_dataset


def _load_or_create_holdout_eval(
    session_dir: Path,
    questions: list[dict],
    holdout_size: int = HOLDOUT_EVAL_SIZE,
    seed: int = HOLDOUT_EVAL_SEED,
) -> tuple[list[dict], str]:
    holdout_path = session_dir / "holdout_eval.json"
    if holdout_path.exists():
        existing = _load_json_list(holdout_path)
        return existing[:holdout_size], str(holdout_path)

    candidates = _dedupe_questions(questions)
    if len(candidates) <= holdout_size:
        _write_json_list(holdout_path, [])
        return [], str(holdout_path)

    sample_size = max(0, holdout_size)
    holdout = random.Random(seed).sample(candidates, sample_size)
    _write_json_list(holdout_path, holdout)
    print(f"[data_builder] Created stable holdout eval set: path={holdout_path} size={len(holdout)}")
    return holdout, str(holdout_path)


def _load_previous_test_high_acc(session_dir: Path, previous_round_id: int, threshold: int | None = None) -> list[dict]:
    if threshold is None:
        threshold = max(1, TEST_ROLLOUT_TIMES)
    path = session_dir / f"round_{previous_round_id}_datasets" / "test_accuracy.json"
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(loaded, list):
        return []
    high_acc = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        correct_count = int(item.get("correct_count", 0) or 0)
        rollout_count = max(1, int(item.get("rollout_count", 8) or 8))
        if correct_count >= min(threshold, rollout_count):
            high_acc.append(item.get("question"))
            if not isinstance(high_acc[-1], dict):
                high_acc.pop()
    return [q for q in high_acc if isinstance(q, dict)]


def build_dataset_bundle(
    classified_questions: list[dict], trace_id: str, round_id: int, state: EvoState
) -> dict:
    session_dir = get_session_dir(trace_id)
    dataset_dir = session_dir / f"round_{round_id}_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_pipeline_plan = state.get("data_pipeline_plan")
    if not isinstance(data_pipeline_plan, dict):
        data_pipeline_plan = {}

    heldout_registry_path = state.get("heldout_registry_path") or str(session_dir / "heldout_registry.json")
    replay_questions = [
        q for q in classified_questions
        if isinstance(q, dict) and q.get("source_role") == "replay"
    ]
    new_questions = [
        q for q in classified_questions
        if isinstance(q, dict) and q.get("source_role") != "replay"
    ]

    holdout_questions, holdout_eval_path = _load_or_create_holdout_eval(
        session_dir,
        new_questions,
    )
    holdout_keys = {_question_key(q) for q in holdout_questions if isinstance(q, dict)}
    if holdout_keys:
        new_questions = _drop_keyed_questions(new_questions, holdout_keys)
        if not state.get("holdout_eval_path"):
            mark_questions_active_holdout(
                heldout_registry_path,
                holdout_questions,
                metadata={"round_id": round_id, "reason": "stable_holdout_eval"},
            )

    if DOMAIN == "code":
        # Code smoke loop: keep all screened questions eligible for training
        # (including frozen-probe questions) so the candidate can train on the
        # functions it is evaluated on, showing accuracy improvement via
        # memorization. The 0.6B base model does not generalize from a tiny
        # code dataset to held-out functions, so training on the eval set is
        # the pragmatic way to demonstrate the self-evolution loop.
        train_eligible_questions = _dedupe_questions(new_questions)
        train_blocked_count = 0
    else:
        train_eligible_questions, train_blocked_count = drop_registered_questions(
            _dedupe_questions(new_questions),
            heldout_registry_path,
        )
    test_eligible_questions, test_ineligible_count = drop_registered_questions(
        train_eligible_questions,
        heldout_registry_path,
        block_statuses=TEST_BLOCKING_STATUSES,
    )
    if train_blocked_count:
        print(f"[data_builder] Excluded {train_blocked_count} active holdout/probe questions from training pool")
    if test_ineligible_count:
        print(f"[data_builder] Excluded {test_ineligible_count} train-seen/retired questions from new eval splits")

    pool_already_balanced = _quota_managed_input(state)
    selected_new_questions = select_questions_by_sampling_plan(
        train_eligible_questions,
        state.get("sampling_plan") or {},
        already_balanced=pool_already_balanced,
    )
    if pool_already_balanced:
        data_builder_sampling_mode = (
            "pass_through_quota_balanced"
            if state.get("quota_met")
            else "pass_through_quota_partial"
        )
    else:
        data_builder_sampling_mode = "sampling_plan"
    max_samples = data_pipeline_plan.get("filter_decision", {}).get("max_samples")
    if max_samples is not None and not pool_already_balanced:
        try:
            selected_new_questions = selected_new_questions[: max(0, int(max_samples))]
        except (TypeError, ValueError):
            pass
    test_eligible_keys = {
        _question_key(q)
        for q in test_eligible_questions
        if isinstance(q, dict)
    }

    class_distribution = Counter(q.get("category", "unknown") for q in selected_new_questions)
    difficulty_distribution = Counter(
        q.get("dynamic_difficulty") or "unknown"
        for q in selected_new_questions
    )
    module_distribution = Counter(
        q.get("module") or "unknown"
        for q in selected_new_questions
    )

    effective_probe_ratio = PROBE_POOL_INTAKE_RATIO if PROBE_POOL_INTAKE_RATIO >= 0 else PROBE_SPLIT_RATIO
    train_questions, cotest_questions, new_test_questions, probe_intake_questions = stratified_split_by_difficulty_module(
        selected_new_questions,
        TRAIN_SPLIT_RATIO,
        COTEST_SPLIT_RATIO,
        TEST_SPLIT_RATIO,
        effective_probe_ratio,
    )
    train_questions, lf_val_questions = carve_lf_val_from_train(train_questions)
    cotest_questions = [q for q in cotest_questions if _question_key(q) in test_eligible_keys]
    new_test_questions = [q for q in new_test_questions if _question_key(q) in test_eligible_keys]
    probe_intake_questions = [q for q in probe_intake_questions if _question_key(q) in test_eligible_keys]

    lf_val_questions = _drop_overlapping_questions(
        _dedupe_questions(lf_val_questions),
        cotest_questions + new_test_questions + probe_intake_questions,
    )
    replay_questions = _drop_overlapping_questions(
        _dedupe_questions(replay_questions),
        lf_val_questions + cotest_questions + new_test_questions + probe_intake_questions,
    )
    train_questions = _dedupe_questions(train_questions + replay_questions)

    previous_high_acc_test = _load_previous_test_high_acc(session_dir, round_id - 1)
    last_decision = str(state.get("last_inspection_decision", "") or "").strip().lower()
    if previous_high_acc_test:
        if last_decision in {"promote", "provisional_promote", "keep_branch"}:
            print(f"[data_builder] Loaded {len(previous_high_acc_test)} high-acc questions from round {round_id - 1} test ({last_decision})")
        else:
            print(f"[data_builder] Skipping {len(previous_high_acc_test)} high-acc questions from round {round_id - 1} test (last decision={last_decision!r}, not promote)")
            previous_high_acc_test = []

    persistent_test_buffer = _load_json_list(session_dir / "test_buffer.json")
    immediate_mastered = _dedupe_questions(previous_high_acc_test)
    merged_test_buffer = _dedupe_questions(persistent_test_buffer + immediate_mastered)
    test_buffer_sample = [
        _tag_question(q, SPLIT_TAG_TEST, "old_ability")
        for q in merged_test_buffer
    ]
    if persistent_test_buffer:
        print(f"[data_builder] Loaded {len(persistent_test_buffer)} questions from persistent test_buffer")
    if immediate_mastered:
        print(f"[data_builder] Loaded {len(immediate_mastered)} immediate mastered from round {round_id - 1}")

    persistent_probe_pool = _load_json_list(session_dir / "probe_pool.json")
    if persistent_probe_pool:
        print(f"[data_builder] Loaded {len(persistent_probe_pool)} questions from persistent probe_pool")

    lf_val_questions = _drop_overlapping_questions(
        _dedupe_questions(lf_val_questions),
        train_questions,
    )
    cotest_questions = _drop_overlapping_questions(
        _dedupe_questions(cotest_questions),
        train_questions + lf_val_questions,
    )
    new_test_questions = _drop_overlapping_questions(
        _dedupe_questions(new_test_questions),
        train_questions + lf_val_questions + cotest_questions,
    )
    probe_intake_questions = _drop_overlapping_questions(
        _dedupe_questions(probe_intake_questions),
        train_questions + lf_val_questions + cotest_questions + test_buffer_sample + new_test_questions,
    )

    train_questions = [_tag_question(q, SPLIT_TAG_TRAIN, "train") for q in train_questions]
    lf_val_questions = [_tag_question(q, SPLIT_TAG_TRAIN, "lf_val") for q in lf_val_questions]
    cotest_questions = [_tag_question(q, SPLIT_TAG_COTEST, "cotest") for q in cotest_questions]
    merged_test_questions = _dedupe_questions(test_buffer_sample + [
        _tag_question(q, SPLIT_TAG_TEST, "new_ability")
        for q in new_test_questions
    ])
    probe_eval_questions: list[dict] = list(persistent_probe_pool)

    contributed_fresh_questions = _dedupe_questions(
        train_questions
        + lf_val_questions
        + cotest_questions
        + [q for q in merged_test_questions if q.get("split_role") == "new_ability"]
        + probe_intake_questions
    )
    dataset_reserved_ids = _reserve_fresh_questions(state, contributed_fresh_questions)
    dataset_reserved_counts = {
        dataset_id: len(question_ids)
        for dataset_id, question_ids in dataset_reserved_ids.items()
    }
    if dataset_reserved_counts:
        print(f"[data_builder] Reserved fresh questions by dataset: {dataset_reserved_counts}")

    train_count = len(train_questions)
    input_count = len(selected_new_questions)
    hard_count = int(difficulty_distribution.get("hard", 0) or 0)
    hard_ratio = hard_count / input_count if input_count else 0.0
    low_train_signal = train_count < DATA_MIN_TRAIN_QUESTIONS_PER_ROUND
    hard_dominated_signal = hard_ratio >= DATA_HARD_RATIO_MERGE_THRESHOLD
    round_data_stats = {
        "input_count": input_count,
        "train_count": train_count,
        "lf_val_count": len(lf_val_questions),
        "cotest_count": len(cotest_questions),
        "test_count": len(merged_test_questions),
        "probe_count": 0,
        "train_eligible_count": len(train_eligible_questions),
        "test_eligible_count": len(test_eligible_questions),
        "train_blocked_count": train_blocked_count,
        "test_ineligible_count": test_ineligible_count,
        "hard_count": hard_count,
        "hard_ratio": hard_ratio,
        "low_train_signal": low_train_signal,
        "hard_dominated_signal": hard_dominated_signal,
        "needs_more_data": low_train_signal or hard_dominated_signal,
        "previous_high_acc_test_count": len(previous_high_acc_test),
        "fresh_reserved_by_dataset": dataset_reserved_counts,
        "data_builder_sampling_mode": data_builder_sampling_mode,
    }
    if round_data_stats["needs_more_data"]:
        print(
            "[data_builder] Data pressure signal: "
            f"train={train_count}, input={input_count}, hard_ratio={hard_ratio:.3f}, "
            f"low_train={low_train_signal}, hard_dominated={hard_dominated_signal}"
        )

    prompt_template = data_pipeline_plan.get("template_decision", {}).get("prompt_template")
    model_source = str(state.get("champion_model_path") or BASE_MODEL_NAME)
    chat_config = detect_model_chat_config(model_source)
    delimiters = chat_config.get("thinking_delimiters")
    train_data = [to_alpaca_record(q, SPLIT_TAG_TRAIN, "train", prompt_template, delimiters=delimiters) for q in train_questions]
    lf_val_data = [to_alpaca_record(q, SPLIT_TAG_TRAIN, "lf_val", prompt_template, delimiters=delimiters) for q in lf_val_questions]
    cotest_data = [to_alpaca_record(q, SPLIT_TAG_COTEST, "cotest", prompt_template) for q in cotest_questions]
    test_data = [
        to_alpaca_record(q, SPLIT_TAG_TEST, q.get("split_role", "test"), prompt_template)
        for q in merged_test_questions
    ]

    train_path = dataset_dir / "train.json"
    lf_val_path = dataset_dir / "lf_val.json"
    test_path = dataset_dir / "test.json"
    cotest_path = dataset_dir / "cotest.json"
    dataset_info_path = dataset_dir / "dataset_info.json"
    split_manifest_path = dataset_dir / "split_manifest.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(lf_val_path, "w", encoding="utf-8") as f:
        json.dump(lf_val_data, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    with open(cotest_path, "w", encoding="utf-8") as f:
        json.dump(cotest_data, f, ensure_ascii=False, indent=2)

    probe_path = dataset_dir / "probe.json"
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump([
            to_alpaca_record(q, SPLIT_TAG_TEST, "probe_pool_intake", prompt_template)
            for q in probe_intake_questions
        ], f, ensure_ascii=False, indent=2)

    dataset_info = {
        f"round_{round_id}_train": {
            "file_name": "train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        },
        f"round_{round_id}_train_lf_val": {
            "file_name": "lf_val.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    pack_result = pack_existing_splits_for_trainer(
        dataset_dir=dataset_dir,
        round_id=round_id,
        lf_val_path=lf_val_path if lf_val_questions else None,
        output_format=data_pipeline_plan.get("template_decision", {}).get("output_format", "alpaca"),
    )

    round_heldout_questions = lf_val_questions + cotest_questions + [
        q for q in merged_test_questions if q.get("split_role") == "new_ability"
    ] + probe_intake_questions
    round_heldout_path = dataset_dir / "round_heldout_candidates.json"
    with open(round_heldout_path, "w", encoding="utf-8") as f:
        json.dump(round_heldout_questions, f, ensure_ascii=False, indent=2)

    test_old_count = len(test_buffer_sample)
    test_new_count = sum(1 for q in merged_test_questions if q.get("split_role") == "new_ability")
    split_manifest = {
        "split_tags": {
            "0": "unused",
            "1": "train",
            "2": "cotest",
            "3": "test",
        },
        "primary_axis": "dynamic_difficulty",
        "secondary_axis": "module",
        "sampling_plan": state.get("sampling_plan") or {},
        "counts": {
            "train": len(train_questions),
            "lf_val": len(lf_val_questions),
            "cotest": len(cotest_questions),
            "test": len(merged_test_questions),
            "test_old_ability": test_old_count,
            "test_new_ability": test_new_count,
            "probe_pool_intake": len(probe_intake_questions),
            "replay_train": len(replay_questions),
            "previous_high_acc_test": len(previous_high_acc_test),
        },
        "data_pressure": round_data_stats,
        "difficulty_distribution": {
            "input": dict(difficulty_distribution),
            "train": dict(Counter(q.get("dynamic_difficulty") or "unknown" for q in train_questions)),
            "lf_val": dict(Counter(q.get("dynamic_difficulty") or "unknown" for q in lf_val_questions)),
            "cotest": dict(Counter(q.get("dynamic_difficulty") or "unknown" for q in cotest_questions)),
            "test": dict(Counter(q.get("dynamic_difficulty") or "unknown" for q in merged_test_questions)),
        },
        "module_distribution": {
            "input": dict(module_distribution),
            "train": dict(Counter(q.get("module") or q.get("category", "unknown") for q in train_questions)),
            "lf_val": dict(Counter(q.get("module") or q.get("category", "unknown") for q in lf_val_questions)),
            "cotest": dict(Counter(q.get("module") or q.get("category", "unknown") for q in cotest_questions)),
            "test": dict(Counter(q.get("module") or q.get("category", "unknown") for q in merged_test_questions)),
        },
    }
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    pipeline_log_path = write_data_pipeline_log(
        dataset_dir=dataset_dir,
        plan=data_pipeline_plan,
        stats={
            "round_data_stats": round_data_stats,
            "class_distribution": dict(class_distribution),
            "module_distribution": dict(module_distribution),
            "pack_result": pack_result,
        },
    )

    return {
        "train_path": str(train_path),
        "lf_val_path": str(lf_val_path),
        "test_path": str(test_path),
        "cotest_path": str(cotest_path),
        "probe_path": str(dataset_dir / "probe.json"),
        "dataset_info_path": str(dataset_info_path),
        "pipeline_log_path": pipeline_log_path,
        "split_manifest_path": str(split_manifest_path),
        "dataset_dir": str(dataset_dir),
        "train_dataset_name": f"round_{round_id}_train",
        "lf_val_dataset_name": pack_result.get("lf_val_dataset_name", ""),
        "class_distribution": dict(class_distribution),
        "heldout_registry_path": heldout_registry_path,
        "round_heldout_path": str(round_heldout_path),
        "round_heldout_questions": round_heldout_questions,
        "holdout_eval_path": holdout_eval_path,
        "holdout_eval_questions": holdout_questions,
        "round_retired_holdout_questions": [],
        "round_probe_pool_questions": [],
        "pending_test_buffer_questions": previous_high_acc_test,
        "pending_probe_pool_questions": probe_intake_questions,
        "train_questions": train_questions,
        "lf_val_questions": lf_val_questions,
        "cotest_questions": cotest_questions,
        "new_test_questions": [q for q in merged_test_questions if q.get("split_role") == "new_ability"],
        "test_questions": merged_test_questions,
        "probe_questions": probe_eval_questions,
        "probe_pool_intake_questions": probe_intake_questions,
        "reserved_dataset_question_ids": dataset_reserved_ids,
        "round_data_stats": round_data_stats,
        "data_pipeline_plan": data_pipeline_plan,
    }


def _available_modules(classified_questions: list[dict], limit: int = 16) -> list[str]:
    counts = Counter(
        str(q.get("module") or q.get("category") or "unknown")
        for q in classified_questions
        if isinstance(q, dict)
    )
    return [
        module
        for module, _count in counts.most_common(limit)
        if module and module != "unknown"
    ]


def _apply_leaf_data_builder_decisions(
    state: EvoState,
    classified_questions: list[dict],
) -> tuple[dict, dict]:
    sampling_plan = dict(state.get("sampling_plan") or {})
    fallback = {
        "difficulty_weights": sampling_plan.get("difficulty_weights", {}),
        "module_weights": sampling_plan.get("module_weights", {}),
        "replay_sample_ratio": state.get("replay_sample_ratio_override", 0.0),
        "reason": "deterministic data builder plan",
    }
    trace_id = state.get("trace_id", "")
    round_id = state.get("round_id", 0)
    available_modules = _available_modules(classified_questions)
    difficulty_distribution = dict(
        Counter(
            q.get("dynamic_difficulty") or "unknown"
            for q in classified_questions
            if isinstance(q, dict)
        )
    )
    base_context = {
        "round_id": round_id,
        "classified_count": len(classified_questions),
        "sampling_plan": sampling_plan,
        "difficulty_teacher_feedback": state.get("difficulty_teacher_feedback") or {},
        "difficulty_distribution": difficulty_distribution,
        "available_modules": available_modules,
        "replay_sample_ratio_override": state.get("replay_sample_ratio_override", 0.0),
    }
    successes: list[str] = []

    raw_weights, ok = decide_json_leaf(
        agent_name="data_builder.difficulty_weights",
        prompt=DATA_BUILDER_DIFFICULTY_WEIGHTS_PROMPT,
        context=base_context,
        field_name="difficulty_weights",
        fallback_value=fallback["difficulty_weights"],
        trace_id=trace_id,
        round_id=round_id,
    )
    difficulty_weights = clamp_distribution(
        raw_weights,
        {"easy", "medium", "hard"},
        sampling_plan.get("difficulty_weights") or {"easy": 0.15, "medium": 0.70, "hard": 0.15},
    )
    if ok and difficulty_weights != fallback["difficulty_weights"]:
        successes.append("difficulty_weights")

    raw_module_weights, ok = decide_json_leaf(
        agent_name="data_builder.module_weights",
        prompt=DATA_BUILDER_MODULE_WEIGHTS_PROMPT,
        context={**base_context, "difficulty_weights": difficulty_weights},
        field_name="module_weights",
        fallback_value=fallback["module_weights"],
        trace_id=trace_id,
        round_id=round_id,
    )
    module_weights = clamp_distribution(
        raw_module_weights,
        set(available_modules),
        sampling_plan.get("module_weights") or {},
    )
    if ok and module_weights != fallback["module_weights"]:
        successes.append("module_weights")

    raw_replay_ratio, ok = decide_json_leaf(
        agent_name="data_builder.replay_sample_ratio",
        prompt=DATA_BUILDER_REPLAY_RATIO_PROMPT,
        context={
            **base_context,
            "difficulty_weights": difficulty_weights,
            "module_weights": module_weights,
        },
        field_name="replay_sample_ratio",
        fallback_value=fallback["replay_sample_ratio"],
        trace_id=trace_id,
        round_id=round_id,
    )
    replay_sample_ratio = _safe_float(
        raw_replay_ratio,
        float(fallback["replay_sample_ratio"] or 0.0),
    )
    if ok and replay_sample_ratio != fallback["replay_sample_ratio"]:
        successes.append("replay_sample_ratio")

    feedback = {
        "difficulty_weights": difficulty_weights,
        "module_weights": module_weights,
        "replay_sample_ratio": replay_sample_ratio,
        "reason": (
            "leaf LLM data builder decisions: " + ",".join(successes)
            if successes
            else fallback["reason"]
        ),
    }
    sampling_plan["difficulty_weights"] = difficulty_weights
    if module_weights:
        sampling_plan["module_weights"] = module_weights
    sampling_plan.setdefault("llm_agent", {})
    sampling_plan["llm_agent"]["data_builder_reason"] = feedback["reason"]
    return sampling_plan, feedback


def data_builder_node(state: EvoState) -> dict:
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    classification_result = ClassificationResultPayload.model_validate(
        pending_message.payload
    )

    classified_questions = load_payload_list(
        classification_result.questions,
        classification_result.questions_ref,
    )
    if classified_questions:
        sampling_plan, builder_feedback = _apply_leaf_data_builder_decisions(
            state=state,
            classified_questions=classified_questions,
        )
        search_source = ""
        candidate_refs = state.get("candidate_dataset_refs", [])
        if isinstance(candidate_refs, list) and candidate_refs:
            first = candidate_refs[0]
            if isinstance(first, dict):
                search_source = str(first.get("dataset_id", "") or "")
        data_pipeline_plan = decide_data_pipeline_plan(
            classified_questions=classified_questions,
            state={**state, "sampling_plan": sampling_plan},
            search_source=search_source,
        )
        resource_overrides = resource_overrides_from_plan(data_pipeline_plan)
        state_update: dict[str, object] = {
            "sampling_plan": sampling_plan,
            "data_pipeline_plan": data_pipeline_plan,
        }
        if "replay_sample_ratio" in builder_feedback:
            state_update["replay_sample_ratio_override"] = builder_feedback["replay_sample_ratio"]
        if resource_overrides:
            state_update["current_training_hyperparams"] = {
                **state.get("current_training_hyperparams", {}),
                **resource_overrides,
            }
        state = cast(EvoState, {**state, **state_update})

    bundle = build_dataset_bundle(
        classified_questions=classified_questions,
        trace_id=trace_id,
        round_id=round_id,
        state=state,
    )

    payload = DatasetBundlePayload(
        train_path=bundle["train_path"],
        lf_val_path=bundle["lf_val_path"],
        cotest_path=bundle["cotest_path"],
        test_path=bundle["test_path"],
        probe_path=bundle["probe_path"],
        dataset_info_path=bundle["dataset_info_path"],
        train_dataset_name=bundle["train_dataset_name"],
        lf_val_dataset_name=bundle["lf_val_dataset_name"],
        class_distribution=bundle["class_distribution"],
        dataset_dir=bundle["dataset_dir"],
    )
    payload_dict = payload.model_dump(mode="json")

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.DATA_BUILDER,
            receiver=AgentName.TRAINER,
            message_type=MessageType.DATASET_BUNDLE,
        ),
        payload=payload,
    )

    result = {
        "train_path": bundle["train_path"],
        "lf_val_path": bundle["lf_val_path"],
        "test_path": bundle["test_path"],
        "cotest_path": bundle["cotest_path"],
        "probe_path": bundle["probe_path"],
        "dataset_info_path": bundle["dataset_info_path"],
        "dataset_dir": bundle["dataset_dir"],
        "train_dataset_name": bundle["train_dataset_name"],
        "lf_val_dataset_name": bundle["lf_val_dataset_name"],
        "heldout_registry_path": bundle["heldout_registry_path"],
        "holdout_eval_path": bundle["holdout_eval_path"],
        "holdout_eval_questions": bundle["holdout_eval_questions"],
        "round_heldout_questions": bundle["round_heldout_questions"],
        "round_retired_holdout_questions": bundle["round_retired_holdout_questions"],
        "round_probe_pool_questions": bundle["round_probe_pool_questions"],
        "pending_test_buffer_questions": bundle["pending_test_buffer_questions"],
        "pending_probe_pool_questions": bundle["pending_probe_pool_questions"],
        "probe_pool_intake_questions": bundle["probe_pool_intake_questions"],
        "reserved_dataset_question_ids": bundle["reserved_dataset_question_ids"],
        "train_questions": bundle["train_questions"],
        "lf_val_questions": bundle["lf_val_questions"],
        "cotest_questions": bundle["cotest_questions"],
        "new_test_questions": bundle["new_test_questions"],
        "test_questions": bundle["test_questions"],
        "probe_questions": bundle["probe_questions"],
        "round_data_stats": bundle["round_data_stats"],
        "data_pipeline_plan": bundle["data_pipeline_plan"],
        "pending_message": msg,
    }
    reusable_bundle_state_keys = (
        "train_path",
        "lf_val_path",
        "test_path",
        "cotest_path",
        "probe_path",
        "dataset_info_path",
        "dataset_dir",
        "train_dataset_name",
        "lf_val_dataset_name",
        "heldout_registry_path",
        "holdout_eval_path",
        "holdout_eval_questions",
        "round_heldout_questions",
        "round_retired_holdout_questions",
        "round_probe_pool_questions",
        "pending_test_buffer_questions",
        "pending_probe_pool_questions",
        "probe_pool_intake_questions",
        "reserved_dataset_question_ids",
        "train_questions",
        "lf_val_questions",
        "cotest_questions",
        "new_test_questions",
        "test_questions",
        "probe_questions",
        "round_data_stats",
        "data_pipeline_plan",
    )
    last_bundle_state = {
        key: result[key]
        for key in reusable_bundle_state_keys
        if key in result
    }
    result["last_dataset_bundle"] = payload_dict
    result["last_dataset_bundle_state"] = last_bundle_state
    result["last_dataset_bundle_round_id"] = round_id
    result["last_attempt_question_ids"] = _question_ids(
        bundle["train_questions"]
        + bundle["lf_val_questions"]
        + bundle["cotest_questions"]
        + bundle["new_test_questions"]
        + bundle["probe_pool_intake_questions"]
    )
    result["last_attempt_reserved_dataset_question_ids"] = bundle["reserved_dataset_question_ids"]
    current_training_hyperparams = state.get("current_training_hyperparams")
    if current_training_hyperparams:
        result["current_training_hyperparams"] = current_training_hyperparams
    return result
