import argparse
import sys
from loguru import logger


def setup_logging():
    from config.settings import LOG_FORMAT, LOG_LEVEL

    logger.remove()
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True,
    )


def run_evolution(user_goal: str):
    setup_logging()

    from src.harness import compile_graph
    from src.models.state import EvoState

    logger.info(f"Starting evolution for goal: {user_goal}")

    graph = compile_graph()

    initial_state: EvoState = {
        "user_goal": user_goal,
    }

    try:
        final_state = graph.invoke(initial_state)

        logger.info("Evolution completed!")
        logger.info(f"Final state summary: {_summarize_final_state(final_state)}")

        return final_state
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


def _summarize_final_state(final_state: dict) -> dict:
    return {
        "trace_id": final_state.get("trace_id"),
        "round_id": final_state.get("round_id"),
        "champion_model_path": final_state.get("champion_model_path"),
        "candidate_model_path": final_state.get("candidate_model_path"),
        "should_stop": final_state.get("should_stop", False),
        "should_promote_candidate": final_state.get("should_promote_candidate", False),
        "budget_exhausted": final_state.get("budget_exhausted", False),
        "termination_reason": final_state.get("termination_reason", ""),
        "metrics_after": final_state.get("metrics_after", {}),
        "dag_nodes": len(final_state.get("search_dag_nodes", [])),
        "dag_edges": len(final_state.get("search_dag_edges", [])),
        "replay_entries": len(final_state.get("replay_buffer_entries", [])),
        "replay_buffer_used_count": final_state.get("replay_buffer_used_count", 0),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="EvoChampion — Model adaptation and curriculum search loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "improve math reasoning"
  python main.py --goal "improve Python coding" --benchmark openai/humaneval --benchmark-question-key prompt --benchmark-answer-key canonical_solution
  python main.py --goal "improve English translation" --benchmark wmt16 --benchmark-subset de-en
        """,
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="Evolution goal (e.g. 'improve math reasoning')",
    )
    parser.add_argument(
        "--goal", dest="goal_kw",
        help="Evolution goal (keyword form)",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark dataset ID on HuggingFace (e.g. 'gsm8k', 'openai/humaneval')",
    )
    parser.add_argument(
        "--benchmark-subset",
        help="Benchmark subset/configuration name",
    )
    parser.add_argument(
        "--benchmark-split",
        help="Benchmark split for training data (default: train)",
    )
    parser.add_argument(
        "--benchmark-eval-split",
        help="Benchmark split for evaluation (default: test)",
    )
    parser.add_argument(
        "--benchmark-question-key",
        help="Field name for question text in benchmark rows",
    )
    parser.add_argument(
        "--benchmark-answer-key",
        help="Field name for gold answer in benchmark rows",
    )
    parser.add_argument(
        "--benchmark-format",
        choices=["gsm8k", "alpaca", "generic"],
        help="Benchmark format type",
    )
    parser.add_argument(
        "--instruction-prefix",
        help="Instruction prefix for inference prompts",
    )
    parser.add_argument(
        "--max-rounds", type=int,
        help="Maximum evolution rounds",
    )
    parser.add_argument(
        "--model",
        help="Base model name or path",
    )
    return parser.parse_args()


def _apply_cli_overrides(args):
    overrides = {}
    if args.goal or args.goal_kw:
        pass
    if args.benchmark:
        overrides["BENCHMARK_DATASET_ID"] = args.benchmark
    if args.benchmark_subset:
        overrides["BENCHMARK_SUBSET"] = args.benchmark_subset
    if args.benchmark_split:
        overrides["BENCHMARK_SPLIT"] = args.benchmark_split
    if args.benchmark_eval_split:
        overrides["BENCHMARK_EVAL_SPLIT"] = args.benchmark_eval_split
    if args.benchmark_question_key:
        overrides["BENCHMARK_QUESTION_KEY"] = args.benchmark_question_key
    if args.benchmark_answer_key:
        overrides["BENCHMARK_ANSWER_KEY"] = args.benchmark_answer_key
    if args.benchmark_format:
        overrides["BENCHMARK_FORMAT"] = args.benchmark_format
    if args.instruction_prefix:
        overrides["INSTRUCTION_PREFIX"] = args.instruction_prefix
    if args.max_rounds:
        overrides["MAX_ROUNDS"] = str(args.max_rounds)
    if args.model:
        overrides["BASE_MODEL_NAME"] = args.model
        overrides["AGENT_BASE_MODEL_NAME"] = args.model
    for key, value in overrides.items():
        import os
        os.environ[key] = value
    return overrides


if __name__ == "__main__":
    args = _parse_args()
    goal = args.goal or args.goal_kw
    if not goal:
        print("Usage: python main.py 'your goal here' [--benchmark ...]")
        print("Example: python main.py 'improve math reasoning' --benchmark gsm8k")
        sys.exit(1)

    overrides = _apply_cli_overrides(args)
    if overrides:
        setup_logging()
        logger.info(f"CLI overrides: {overrides}")

    result = run_evolution(goal)
    print("\n" + "=" * 50)
    print("Evolution Result:")
    print(f"Champion Model: {result.get('champion_model_path', 'N/A')}")
    print(f"Final Round: {result.get('round_id', 0)}")
    print(f"Should Stop: {result.get('should_stop', False)}")
    print(f"Budget Exhausted: {result.get('budget_exhausted', False)}")
    print(f"Termination Reason: {result.get('termination_reason', '')}")
