from loguru import logger

from src.models.state import EvoState


def format_error_node(state: EvoState) -> dict:
    error_msg = state["pending_message"]

    logger.error(
        f"Format error from {error_msg.header.sender.value}: "
        f"{error_msg.payload.reason}"
    )

    return {}
