from collections import Counter
import importlib
from typing import Literal, TypeAlias

from config.settings import CLASSIFIER_MIN_CONFIDENCE, GLINER_MODEL_NAME, USE_GLINER, get_classifier_labels
from src.models.messages import (
    AgentName,
    ClassificationResultPayload,
    FilterResultPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.dataset_bank import infer_module
from src.tools.message_artifacts import load_payload_list, write_json_artifact
from src.tools.question_fields import add_processed_question_fields

# GLiNER 模型缓存（懒加载，加载失败后全局禁用）
_gliner_model = None
_gliner_disabled = False
_GLINER_UNAVAILABLE: Literal["unavailable"] = "unavailable"
_GlinerResult: TypeAlias = tuple[str, float] | None | Literal["unavailable"]


def _infer_module_with_gliner(text: str) -> _GlinerResult:
    """用 GLiNER 模型从题目文本中提取分类标签。

    懒加载模型，加载失败后全局禁用（_gliner_disabled=True）避免重复尝试。
    返回 (标签名, 置信度) 表示 GLiNER 命中；返回 None 表示 GLiNER 可用但未命中；
    返回 _GLINER_UNAVAILABLE 表示依赖/模型不可用，可进入关键词兜底。
    """
    global _gliner_model, _gliner_disabled
    if not text:
        return None
    if not USE_GLINER or _gliner_disabled:
        return _GLINER_UNAVAILABLE
    try:
        if _gliner_model is None:
            GLiNER = importlib.import_module("gliner").GLiNER
            _gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
        labels = get_classifier_labels()
        if not labels:
            return _GLINER_UNAVAILABLE
        entities = _gliner_model.predict_entities(text, labels, threshold=CLASSIFIER_MIN_CONFIDENCE)
    except Exception as exc:
        _gliner_disabled = True
        print(f"[classifier] GLiNER unavailable, falling back to rules: {type(exc).__name__}: {exc}")
        return _GLINER_UNAVAILABLE
    if not entities:
        return None

    scored_labels: list[tuple[str, float]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        label = str(entity.get("label") or "unknown")
        try:
            score = float(entity.get("score", CLASSIFIER_MIN_CONFIDENCE))
        except (TypeError, ValueError):
            score = CLASSIFIER_MIN_CONFIDENCE
        if score >= CLASSIFIER_MIN_CONFIDENCE:
            scored_labels.append((label, score))
    if not scored_labels:
        return None
    counts = Counter(label for label, _score in scored_labels)
    label, _count = counts.most_common(1)[0]
    confidence = max(score for scored_label, score in scored_labels if scored_label == label)
    return label, confidence


def classify_questions(questions: list[dict]) -> list[dict]:
    """对题目列表分类，为每条题目补充分类和置信度。

    分类优先级（由高到低）：
    1. GLiNER 模型（默认启用且可用时唯一分类器）
    2. GLiNER 不可用时，历史 module 字段作为兜底元数据
    3. GLiNER 不可用且无历史 module 时，规则降级（infer_module 关键词匹配）

    返回 enrichment 后的题目列表，每条增加 category / confidence 字段。
    """
    classified = []
    for q in questions:
        text = q.get("question_text", "")
        module = str(q.get("module") or "").strip()

        # 策略层要求 classifier 由 GLiNER 优先决策；历史 module 只作为兜底元数据。
        gliner_result = _infer_module_with_gliner(text)
        if gliner_result == _GLINER_UNAVAILABLE:
            if module:
                category = module
                confidence = 1.0
            else:
                category = infer_module(text)
                confidence = 0.7 if category != "unknown" else 0.3
                module = category
        elif gliner_result:
            category, confidence = gliner_result
            module = category
        else:
            category = "unknown"
            confidence = 0.0

        classified.append(
            add_processed_question_fields(
                {
                    "question_id": q["question_id"],
                    "question_text": q["question_text"],
                    "gold_answer": q["gold_answer"],
                    "source_dataset_id": q.get("source_dataset_id"),
                    "source_dataset_row_id": q.get("source_dataset_row_id"),
                    "source_dataset_split": q.get("source_dataset_split"),
                    "source_dataset_subset": q.get("source_dataset_subset"),
                    "source_dataset_requested_split": q.get("source_dataset_requested_split"),
                    "source_dataset_split_names": q.get("source_dataset_split_names") or [],
                    "source_dataset_columns": q.get("source_dataset_columns") or [],
                    "source_dataset_first_row": q.get("source_dataset_first_row") or {},
                    "source_dataset_schema": q.get("source_dataset_schema") or {},
                    "source_role": q.get("source_role"),
                    "category": category,
                    "module": module,
                    "dynamic_difficulty": q.get("dynamic_difficulty"),
                    "pass_count": q.get("pass_count"),
                    "rollout_count": q.get("rollout_count"),
                    "pass_rate": q.get("pass_rate"),
                    "replay_use_count": q.get("replay_use_count"),
                    "confidence": confidence,
                },
                q,
            )
        )
    return classified


def classifier_node(state: EvoState) -> dict:
    """LangGraph 节点入口：接收过滤后的题目列表，分类后传给 data_builder。

    产出 ClassificationResultPayload → 写入 artifact → 发送给 DATA_BUILDER。
    """
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    filter_result = FilterResultPayload.model_validate(pending_message.payload)
    questions = load_payload_list(filter_result.questions, filter_result.questions_ref)

    print(f"[classifier] Classifying {len(questions)} questions")
    classified = classify_questions(questions)

    cat_counts = Counter(q["category"] for q in classified)
    print(f"[classifier] Categories: {dict(cat_counts)}")

    questions_ref = write_json_artifact(
        trace_id=trace_id,
        round_id=round_id,
        producer="classifier",
        name="classified_questions",
        data=classified,
    )
    payload = ClassificationResultPayload(
        questions=[],
        questions_ref=questions_ref,
        dropped_unknown_count=0,
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.CLASSIFIER,
            receiver=AgentName.DATA_BUILDER,
            message_type=MessageType.CLASSIFICATION_RESULT,
        ),
        payload=payload,
    )

    return {"pending_message": msg}
