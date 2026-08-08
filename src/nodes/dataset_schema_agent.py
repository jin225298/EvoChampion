from src.models.messages import (
    AgentName,
    DatasetSchemaRequestPayload,
    DatasetSchemaResultPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.dataset_adapter import detect_columns_via_llm


def dataset_schema_agent_node(state: EvoState) -> dict:
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    payload = DatasetSchemaRequestPayload.model_validate(pending_message.payload)
    schema = detect_columns_via_llm(payload.inspect_result, dict(state))
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.DATASET_SCHEMA_AGENT,
            receiver=AgentName.SCREENING_ENTRY,
            message_type=MessageType.DATASET_SCHEMA_RESULT,
        ),
        payload=DatasetSchemaResultPayload(
            dataset_ref=payload.dataset_ref,
            schema=schema,
            inspect_result=payload.inspect_result,
        ),
    )

    return {
        "pending_message": msg,
        "dataset_schema_info": {
            "dataset_id": payload.dataset_ref.dataset_id,
            "dataset_ref": payload.dataset_ref.model_dump(),
            "schema": schema,
            "inspect_result": payload.inspect_result,
        },
    }
