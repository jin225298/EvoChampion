from src.models.state import EvoState


def rollout_dispatcher_node(state: EvoState) -> dict:
    questions_data = state.get("candidate_questions", [])
    model_path = state.get("champion_model_path", "")

    print(f"[rollout_dispatcher] Dispatching {len(questions_data)} questions "
          f"for single-cascade rollout with model={model_path}")

    return {
        "rollout_runs": None,
    }
