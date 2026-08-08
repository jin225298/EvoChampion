from src.models.messages import (
    AgentName,
    ClassificationResultPayload,
    DatasetBundlePayload,
    DatasetSchemaRequestPayload,
    DatasetSchemaResultPayload,
    DatasetReviewResultPayload,
    DiagnosticRequestPayload,
    EvalResultPayload,
    FilterResultPayload,
    FormatErrorPayload,
    GoalRequestPayload,
    InspectionResultPayload,
    MaterializedDatasetPayload,
    MessageHeader,
    MessageType,
    render_message_for_ui,
    RolloutRequestPayload,
    RolloutResultPayload,
    RolloutTaskPayload,
    RolloutTaskResultPayload,
    RoutedMessage,
    RouteStatus,
    SearchResultPayload,
    SearchRequestPayload,
    TrainRequestPayload,
    TrainResultPayload,
    DiagnosticResultPayload,
)
from src.models.state import EvoState
from src.tools.harness_contracts import validate_message_contract
from config.settings import get_session_dir, ENFORCE_HARNESS_CONTRACTS


MESSAGE_TYPE_TO_PAYLOAD = {
    MessageType.GOAL_REQUEST: GoalRequestPayload,
    MessageType.SEARCH_REQUEST: SearchRequestPayload,
    MessageType.SEARCH_RESULT: SearchResultPayload,
    MessageType.DATASET_SCHEMA_REQUEST: DatasetSchemaRequestPayload,
    MessageType.DATASET_SCHEMA_RESULT: DatasetSchemaResultPayload,
    MessageType.MATERIALIZED_DATASET: MaterializedDatasetPayload,
    MessageType.ROLLOUT_REQUEST: RolloutRequestPayload,
    MessageType.ROLLOUT_TASK: RolloutTaskPayload,
    MessageType.ROLLOUT_TASK_RESULT: RolloutTaskResultPayload,
    MessageType.ROLLOUT_RESULT: RolloutResultPayload,
    MessageType.FILTER_RESULT: FilterResultPayload,
    MessageType.CLASSIFICATION_RESULT: ClassificationResultPayload,
    MessageType.DATASET_BUNDLE: DatasetBundlePayload,
    MessageType.DATASET_REVIEW_RESULT: DatasetReviewResultPayload,
    MessageType.DIAGNOSTIC_REQUEST: DiagnosticRequestPayload,
    MessageType.TRAIN_REQUEST: TrainRequestPayload,
    MessageType.TRAIN_RESULT: TrainResultPayload,
    MessageType.DIAGNOSTIC_RESULT: DiagnosticResultPayload,
    MessageType.EVAL_RESULT: EvalResultPayload,
    MessageType.INSPECTION_RESULT: InspectionResultPayload,
    MessageType.FORMAT_ERROR: FormatErrorPayload,
}


def _payload_summary(payload) -> dict:
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else {}
    summary: dict = {}
    for key, value in data.items():
        if key.endswith("_ref") and isinstance(value, dict):
            summary[key] = {
                "artifact_id": value.get("artifact_id"),
                "local_path": value.get("local_path"),
                "count": value.get("count"),
            }
            continue
        if isinstance(value, list):
            summary[key] = len(value)
        elif isinstance(value, dict):
            summary[key] = sorted(value.keys())[:20]
        else:
            summary[key] = value
    return summary


def router_node(state: EvoState) -> dict:
    import json
    from datetime import datetime

    message = state.get("pending_message")
    if message is None:
        raise ValueError("pending_message missing")

    try:
        validated = RoutedMessage.model_validate(message)

        expected_payload_type = MESSAGE_TYPE_TO_PAYLOAD.get(validated.header.message_type)
        if expected_payload_type and not isinstance(validated.payload, expected_payload_type):
            raise ValueError(
                f"message_type {validated.header.message_type} expects {expected_payload_type.__name__}, "
                f"got {type(validated.payload).__name__}"
            )

        contract_violations = validate_message_contract(validated)
        for violation in contract_violations:
            print(f"[router] CONTRACT VIOLATION: {violation}")

        if ENFORCE_HARNESS_CONTRACTS and contract_violations:
            violation_details = "; ".join(contract_violations)
            raise ValueError(f"Harness contract violations: {violation_details}")

        validated.header.route_status = RouteStatus.ACCEPTED

        rendered = render_message_for_ui(validated)

        # Persist message to session directory
        trace_id = state.get("trace_id", "unknown")
        try:
            session_dir = get_session_dir(trace_id)
            msgs_dir = session_dir / "router_messages"
            msgs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            short_sender = validated.header.sender.value
            short_recv = validated.header.receiver.value
            short_mt = validated.header.message_type.value
            fname = msgs_dir / f"r{state.get('round_id', 0)}_{short_sender}_to_{short_recv}_{short_mt}_{ts}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({
                    "header": validated.header.model_dump(),
                    "payload": validated.payload.model_dump(mode="json"),
                }, f, ensure_ascii=False, indent=2)
            jsonl_path = session_dir / "router_messages.jsonl"
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "round": validated.header.round_id,
                "sender": short_sender,
                "receiver": short_recv,
                "message_type": short_mt,
                "route_status": validated.header.route_status.value,
                "message_file": str(fname),
                "payload_summary": _payload_summary(validated.payload),
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # best-effort logging

        return {
            "pending_message": validated,
            "message_log": [rendered],
            "route_decision": validated.header.receiver.value,
        }
    except Exception as exc:
        bad_sender = getattr(message.header, "sender", AgentName.SYSTEM) if hasattr(message, "header") else AgentName.SYSTEM

        error_message = RoutedMessage(
            header=MessageHeader(
                trace_id=str(state.get("trace_id", "")),
                round_id=int(state.get("round_id", 0) or 0),
                sender=AgentName.ROUTER,
                receiver=AgentName.FORMAT_ERROR,
                message_type=MessageType.FORMAT_ERROR,
                route_status=RouteStatus.REJECTED,
            ),
            payload=FormatErrorPayload(
                bad_message_type=str(getattr(message.header, "message_type", "unknown") if hasattr(message, "header") else "unknown"),
                missing_fields=[],
                reason=str(exc),
            ),
        )
        return {
            "pending_message": error_message,
            "message_log": [render_message_for_ui(error_message)],
            "route_decision": "format_error",
        }
