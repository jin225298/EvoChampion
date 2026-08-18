import time
import json
from pathlib import Path

from config.settings import (
    BASE_MODEL_NAME,
    DOMAIN,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    TRAINING_CONFIG_TEMPLATE,
    TRAIN_FINETUNING_TYPE,
    get_session_dir,
)
from src.models.messages import (
    AgentName,
    DatasetBundlePayload,
    DiagnosticRequestPayload,
    DiagnosticResultPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    TrainResultPayload,
)
from src.models.state import EvoState
from src.tools.llm_factory import launch_training


def _load_train_record_count(train_path: str) -> int:
    path = Path(train_path)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return len(data) if isinstance(data, list) else 0


def _run_probe_diagnostic(state: EvoState) -> dict:
    from src.nodes.evaluator import evaluate_probe_set_detailed

    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    session_dir = get_session_dir(trace_id)
    if pending_message.header.message_type == MessageType.DIAGNOSTIC_REQUEST:
        request = DiagnosticRequestPayload.model_validate(pending_message.payload)
        frozen_probe_path = request.probe_set_path
        champion_model_path = request.champion_model_path
    else:
        frozen_probe_path = state.get("probe_frozen_set_path") or state.get("global_probe_set_path", "")
        champion_model_path = state.get("champion_model_path", "")
    target_bucket = ""
    diagnostic_dir = session_dir / f"round_{round_id}_diagnostic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    worst_probe_path = diagnostic_dir / "worst_probe_questions.json"
    diagnostic_path = diagnostic_dir / "diagnostic.json"

    accuracy, per_difficulty_acc, failed_examples = evaluate_probe_set_detailed(
        champion_model_path,
        frozen_probe_path,
        reference_model_path=champion_model_path,
        trace_id=trace_id,
    )
    if per_difficulty_acc:
        target_bucket = min(per_difficulty_acc, key=lambda key: per_difficulty_acc[key])
    worst_questions = [
        item for item in failed_examples
        if not target_bucket or item.get("dynamic_difficulty") == target_bucket
    ][:20]
    if len(worst_questions) < 20:
        seen = {item.get("question_id", "") for item in worst_questions}
        for item in failed_examples:
            qid = item.get("question_id", "")
            if qid and qid in seen:
                continue
            worst_questions.append(item)
            if qid:
                seen.add(qid)
            if len(worst_questions) >= 20:
                break
    with open(worst_probe_path, "w", encoding="utf-8") as f:
        json.dump(worst_questions, f, ensure_ascii=False, indent=2)

    diagnostic = {
        "round_id": round_id,
        "mode": "probe_diagnostic",
        "probe_acc_frozen_champion": accuracy,
        "per_difficulty_acc_frozen_champion": per_difficulty_acc,
        "target_bucket": target_bucket,
        "worst_probe_count": len(worst_questions),
        "worst_probe_path": str(worst_probe_path),
    }
    with open(diagnostic_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic, f, ensure_ascii=False, indent=2)

    print(
        f"[trainer] Probe diagnostic mode: skipped training, "
        f"target_bucket={target_bucket}, worst_probe_count={len(worst_questions)}"
    )

    payload = DiagnosticResultPayload(
        candidate_model_path=champion_model_path,
        diagnostic_path=str(diagnostic_path),
        worst_probe_path=str(worst_probe_path),
        worst_probe_count=len(worst_questions),
        target_bucket=target_bucket,
    )
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.TRAINER,
            receiver=AgentName.STRATEGY_INSPECTOR,
            message_type=MessageType.DIAGNOSTIC_RESULT,
        ),
        payload=payload,
    )
    return {
        "candidate_model_path": champion_model_path,
        "diagnostic_path": str(diagnostic_path),
        "worst_probe_path": str(worst_probe_path),
        "target_bucket": target_bucket,
        "pending_message": msg,
    }


def trainer_node(state: EvoState) -> dict:
    t0 = time.time()
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    if (
        state.get("diagnostic_mode") == "probe_diagnostic"
        or pending_message.header.message_type == MessageType.DIAGNOSTIC_REQUEST
    ):
        return _run_probe_diagnostic(state)

    dataset_bundle = DatasetBundlePayload.model_validate(pending_message.payload)

    print(f"[trainer] Starting training for trace_id={trace_id} round_id={round_id}")

    train_record_count = _load_train_record_count(dataset_bundle.train_path)
    if train_record_count <= 0:
        session_dir = get_session_dir(trace_id)
        failure_dir = session_dir / f"round_{round_id}_training_skipped"
        failure_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = failure_dir / "train_log.txt"
        error_message = "empty training dataset; skipped LLaMA-Factory launch"
        train_log_path.write_text(error_message + "\n", encoding="utf-8")
        print(f"[trainer] {error_message}: train_path={dataset_bundle.train_path}")
        payload = TrainResultPayload(
            candidate_model_path=state.get("champion_model_path") or BASE_MODEL_NAME,
            train_log_path=str(train_log_path),
            success=False,
            error_message=error_message,
        )
        msg = RoutedMessage(
            header=MessageHeader(
                trace_id=trace_id,
                round_id=round_id,
                sender=AgentName.TRAINER,
                receiver=AgentName.EVALUATOR,
                message_type=MessageType.TRAIN_RESULT,
            ),
            payload=payload,
        )
        return {
            "candidate_model_path": payload.candidate_model_path,
            "pending_message": msg,
        }

    model_source = state.get("champion_model_path") or BASE_MODEL_NAME
    training_hyperparams = dict(state.get("current_training_hyperparams", {}))
    action_metadata = dict(state.get("current_action_metadata", {}))

    if DOMAIN == "code":
        # Code domain: force the configured finetuning type (LoRA by default).
        # The small-data code loop must use LoRA; the LLM hyperparams agent
        # tends to emit the math default ("full"), which regresses the model.
        finetuning_type = str(TRAIN_FINETUNING_TYPE or "lora").strip().lower()
        training_hyperparams.pop("finetuning_type", None)
    else:
        finetuning_type = str(
            training_hyperparams.pop("finetuning_type", None)
            or action_metadata.get("finetuning_type")
            or TRAIN_FINETUNING_TYPE
            or "full"
        ).strip().lower()
    lora_rank = int(
        training_hyperparams.pop("lora_rank", None)
        or action_metadata.get("lora_rank")
        or LORA_RANK
        or 0
    )
    lora_alpha = int(
        training_hyperparams.pop("lora_alpha", None)
        or action_metadata.get("lora_alpha")
        or LORA_ALPHA
        or 0
    )
    # LoRA needs a higher LR than full finetune. If the agent supplied a
    # full-finetune-scale LR (<= 2e-5) for a LoRA run, bump it to a LoRA-
    # appropriate default so small-data LoRA actually learns.
    if finetuning_type == "lora":
        try:
            agent_lr = float(training_hyperparams.get("learning_rate") or 0.0)
        except (TypeError, ValueError):
            agent_lr = 0.0
        if agent_lr <= 2e-5:
            training_hyperparams["learning_rate"] = 1e-4
        if lora_rank <= 0:
            lora_rank = LORA_RANK or 8
        if lora_alpha <= 0:
            lora_alpha = LORA_ALPHA or 16
        training_hyperparams.setdefault("lora_target", LORA_TARGET_MODULES or "q_proj,v_proj")
        training_hyperparams.setdefault("lora_dropout", LORA_DROPOUT if LORA_DROPOUT is not None else 0.05)
    packing = training_hyperparams.pop("packing", None)
    if packing is None:
        packing = action_metadata.get("packing")
    neat_packing = training_hyperparams.pop("neat_packing", None)
    if neat_packing is None:
        neat_packing = action_metadata.get("neat_packing")
    tokenized_path = training_hyperparams.pop("tokenized_path", None)
    if tokenized_path is None:
        tokenized_path = action_metadata.get("tokenized_path")
    resume_from_checkpoint = bool(
        state.get("resume_training")
        or action_metadata.get("resume_from_checkpoint")
        or action_metadata.get("retry_resume_from_checkpoint")
    )

    result = launch_training(
        model_name_or_path=model_source,
        dataset_dir=dataset_bundle.dataset_dir,
        train_dataset_name=dataset_bundle.train_dataset_name,
        config_template_path=TRAINING_CONFIG_TEMPLATE,
        trace_id=trace_id,
        round_id=round_id,
        hyperparameters=training_hyperparams,
        finetuning_type=finetuning_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        eval_dataset_name=dataset_bundle.lf_val_dataset_name,
        resume_from_checkpoint=resume_from_checkpoint,
        packing=bool(packing) if packing is not None else None,
        neat_packing=bool(neat_packing) if neat_packing is not None else None,
        tokenized_path=str(tokenized_path) if tokenized_path else None,
    )

    elapsed = time.time() - t0
    # P0: propagate training failure — default to failure, not success
    training_success = bool(result.get("success"))
    print(f"[trainer] Training completed in {elapsed:.2f}s "
          f"(candidate={result['candidate_model_path']}, success={training_success})")

    payload = TrainResultPayload(
        candidate_model_path=result["candidate_model_path"],
        train_log_path=result["train_log_path"],
        success=training_success,
        error_message=str(result.get("error_message", "")),
        trainer_log_jsonl_path=str(result.get("trainer_log_jsonl_path", "")),
        training_loss_jsonl_path=str(result.get("training_loss_jsonl_path", "")),
        all_results_path=str(result.get("all_results_path", "")),
        trainer_state_path=str(result.get("trainer_state_path", "")),
        train_results_path=str(result.get("train_results_path", "")),
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.TRAINER,
            receiver=AgentName.EVALUATOR,
            message_type=MessageType.TRAIN_RESULT,
        ),
        payload=payload,
    )

    return {
        "candidate_model_path": result["candidate_model_path"],
        "current_action_metadata": action_metadata,
        "current_training_hyperparams": training_hyperparams,
        "pending_message": msg,
    }
