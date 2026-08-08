"""
Simple test to verify the system can start and process basic messages.
"""

from src.models.messages import (
    AgentName,
    GoalRequestPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    RouteStatus,
)


def test_message_creation():
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id="test123",
            round_id=0,
            sender=AgentName.SYSTEM,
            receiver=AgentName.TEACHER,
            message_type=MessageType.GOAL_REQUEST,
            route_status=RouteStatus.PENDING,
        ),
        payload=GoalRequestPayload(goal="提高数学能力"),
    )

    assert msg.header.trace_id == "test123"
    assert msg.header.sender == AgentName.SYSTEM
    assert msg.header.receiver == AgentName.TEACHER
    assert msg.payload.goal == "提高数学能力"

    print("✓ Message creation test passed")


def test_message_validation():
    msg_dict = {
        "header": {
            "trace_id": "test456",
            "round_id": 1,
            "sender": "teacher",
            "receiver": "searcher",
            "message_type": "search_request",
            "route_status": "待校验",
        },
        "payload": {
            "search_sources": ["huggingface"],
            "search_query": "math problems",
            "retrieval_mode": "full_dataset",
            "dataset_role": "train_dataset",
            "sampling_owner": "filter",
        },
    }

    msg = RoutedMessage.model_validate(msg_dict)

    assert msg.header.sender == AgentName.TEACHER
    assert msg.payload.search_query == "math problems"

    print("✓ Message validation test passed")


if __name__ == "__main__":
    test_message_creation()
    test_message_validation()
    print("\n✅ All tests passed!")
