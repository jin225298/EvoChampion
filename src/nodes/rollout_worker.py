import time

from config.settings import ROLLOUT_TEMPERATURE, ROLLOUT_TOP_P
from src.models.state import EvoState
from src.tools.difficulty_tagger import tag_questions_by_pass_rate


def rollout_worker_node(state: EvoState) -> dict:
    t0 = time.time()
    questions = state.get("candidate_questions", [])
    model_path = state.get("champion_model_path", "")
    worker_idx = int(state.get("rollout_worker_idx", 0) or 0)
    rollout_start_idx = int(state.get("rollout_start_idx", 0) or 0)

    _tagged, single_run, difficulty_counts = tag_questions_by_pass_rate(
        questions=questions,
        model_path=model_path,
        rollout_count=1,
        temperature=ROLLOUT_TEMPERATURE,
        top_p=ROLLOUT_TOP_P,
        round_id=int(state.get("round_id", 0) or 0),
        rollout_start_idx=rollout_start_idx,
        trace_id=str(state.get("trace_id", "")),
        trace_stage=f"rollout_worker_{worker_idx}",
        model_role="champion",
        rollout_config_hash=str(state.get("rollout_config_hash", "") or ""),
        rollout_judge_version=str(state.get("rollout_judge_version", "") or ""),
    )

    elapsed = time.time() - t0
    correct_count = sum(1 for row in single_run if row["correct"])

    print(f"[rollout_worker] worker={worker_idx} "
          f"processed {len(questions)} questions with cascade rollout "
          f"from idx={rollout_start_idx} in {elapsed:.2f}s "
          f"correct={correct_count}/{len(single_run)} "
          f"dynamic_difficulty={difficulty_counts}")

    return {
        "rollout_runs": [single_run],
    }
