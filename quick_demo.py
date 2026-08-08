#!/usr/bin/env python3
"""
Quick demo to show the system's message flow without full execution.
"""

from src.models.messages import (
    AgentName,
    GoalRequestPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    RouteStatus,
    SearchRequestPayload,
)
from src.models.messages import render_message_for_ui


def demo_message_flow():
    print("="*60)
    print("EvoChampion Demo - Message Flow Example")
    print("="*60)
    print()

    print("Step 1: User Goal → System → Teacher")
    print("-"*60)
    msg1 = RoutedMessage(
        header=MessageHeader(
            trace_id="demo_001",
            round_id=0,
            sender=AgentName.SYSTEM,
            receiver=AgentName.TEACHER,
            message_type=MessageType.GOAL_REQUEST,
            route_status=RouteStatus.PENDING,
        ),
        payload=GoalRequestPayload(goal="提高数学能力"),
    )
    print(render_message_for_ui(msg1))
    print()

    print("Step 2: Teacher → Searcher (Search Strategy)")
    print("-"*60)
    msg2 = RoutedMessage(
        header=MessageHeader(
            trace_id="demo_001",
            round_id=0,
            sender=AgentName.TEACHER,
            receiver=AgentName.SEARCHER,
            message_type=MessageType.SEARCH_REQUEST,
            route_status=RouteStatus.ACCEPTED,
        ),
        payload=SearchRequestPayload(
            search_sources=["huggingface", "web"],
            search_query="math problems focusing on trigonometry, function, geometry",
            retrieval_mode="full_dataset",
            dataset_role="train_dataset",
            sampling_owner="filter",
            target_labels=["三角函数", "函数", "几何"],
        ),
    )
    print(render_message_for_ui(msg2))
    print()

    print("="*60)
    print("✓ Message flow demo completed!")
    print()
    print("Next steps:")
    print("1. Edit .env with your configuration")
    print("2. Run: python3 main.py '提高数学能力'")
    print("="*60)


if __name__ == "__main__":
    demo_message_flow()
