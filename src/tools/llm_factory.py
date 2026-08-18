"""
LLaMA-Factory based training for the EvoChampion system.
Uses llamafactory-cli for SFT with LoRA fine-tuning.

Loading strategy:
  1. Resolve model to local cache path (avoids hf-xet corruption)
  2. Try offline/local cache first (instant, reliable on GPU nodes)
  3. If offline fails (model not cached), try online with limited retries

Critical: hf-xet hijacks from_pretrained even in offline mode, causing
AttributeError: 'NoneType' object has no attribute 'endswith'.
We work around this by resolving the model to a local snapshot path
before passing it to llamafactory-cli, bypassing the hub download path.
XET_DISABLE=1 is also set as a defense-in-depth measure.
"""

import json
import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from config.settings import (
    CANDIDATE_MODEL_DIR,
    EXPORT_DEVICE,
    EXPORT_SIZE_GB,
    EXPORT_TEMPLATE,
    TRAINING_CONFIG_TEMPLATE,
    TRAINING_TIMEOUT_SECONDS,
    TRAIN_EVAL_BATCH_SIZE,
    TRAIN_EVAL_STEPS,
    TRAIN_EVAL_STRATEGY,
    TRAIN_FINETUNING_TYPE,
    TRAIN_LOAD_BEST_MODEL_AT_END,
    TRAIN_NEAT_PACKING,
    TRAIN_PACKING,
    TRAIN_RESUME_ENABLED,
    TRAIN_SAVE_ONLY_MODEL,
    TRAIN_SAVE_STEPS,
    TRAIN_SAVE_TOTAL_LIMIT,
    TRAIN_TOKENIZED_PATH,
)
from src.utils.hf_cache import resolve_model_path
from src.tools.cot_format import detect_model_chat_config


def _make_env(offline: bool = False) -> dict:
    """构建训练子进程所需的环境变量。

    始终设置：HF 镜像端点、禁用 xet/transfer 的坑。
    offline=True 时额外设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，
    阻止 LLaMA-Factory 在离线模式下尝试网络下载。
    """
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["XET_DISABLE"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def launch_training(
    model_name_or_path: str,
    dataset_dir: str,
    train_dataset_name: str,
    config_template_path: str,
    trace_id: str,
    round_id: int,
    hyperparameters: dict | None = None,
    finetuning_type: str = "full",
    lora_rank: int = 0,
    lora_alpha: int = 0,
    eval_dataset_name: str = "",
    resume_from_checkpoint: bool = False,
    packing: bool | None = None,
    neat_packing: bool | None = None,
    tokenized_path: str | None = None,
) -> dict:
    """启动 LLaMA-Factory 训练，离线优先、在线兜底。

    流程：
    1. 解析模型路径为本地快照（绕过 hf-xet 污染）
    2. 载入训练配置模板并注入当前轮次的参数
    3. 写入临时 YAML 配置文件
    4. 先以离线模式（HF_HUB_OFFLINE=1）启动训练
    5. 若离线失败，自动降级为在线模式重试
    6. 验证输出目录中是否有可加载的模型文件

    返回字典包含 candidate_model_path、train_log_path、success、error_message，
    以及可选的 LLaMA-Factory 结构化日志路径。
    """
    candidate_dir = Path(CANDIDATE_MODEL_DIR)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    output_dir = candidate_dir / f"candidate_{trace_id}_round{round_id}"
    train_log_path = candidate_dir / f"train_log_{trace_id}_round{round_id}.txt"
    success = False
    error_message = ""

    resolved_model_path = resolve_model_path(model_name_or_path)

    active_hyperparameters = hyperparameters or {}
    active_resume = bool(
        resume_from_checkpoint
        and TRAIN_RESUME_ENABLED
        and not TRAIN_SAVE_ONLY_MODEL
        and not active_hyperparameters
    )
    if resume_from_checkpoint and TRAIN_SAVE_ONLY_MODEL:
        print("[trainer] Resume skipped because TRAIN_SAVE_ONLY_MODEL=true omits trainer state")
    if resume_from_checkpoint and active_hyperparameters:
        print("[trainer] Resume skipped because hyperparameter overrides are present")
    latest_checkpoint = _find_latest_checkpoint(output_dir) if active_resume else None

    config = _load_and_patch_config(
        template_path=config_template_path or TRAINING_CONFIG_TEMPLATE,
        model_name_or_path=resolved_model_path,
        dataset_dir=dataset_dir,
        dataset_name=train_dataset_name,
        output_dir=str(output_dir),
        hyperparameters=active_hyperparameters,
        finetuning_type=finetuning_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        eval_dataset_name=eval_dataset_name,
        packing=packing,
        neat_packing=neat_packing,
        tokenized_path=tokenized_path,
    )
    if latest_checkpoint is not None:
        config["resume_from_checkpoint"] = str(latest_checkpoint)
        config["overwrite_output_dir"] = False
    else:
        config["overwrite_output_dir"] = True

    config_path = candidate_dir / f"config_{trace_id}_round{round_id}.yaml"
    _write_config(config, config_path)

    cmd = ["llamafactory-cli", "train", str(config_path)]

    result = None

    try:
        _clear_inference_cache_before_training()
        print(f"[trainer] Starting training: {' '.join(cmd)}")
        print(f"[trainer]   model={model_name_or_path} (resolved: {resolved_model_path})")
        print(f"[trainer]   dataset_dir={dataset_dir}")
        print(f"[trainer]   dataset={train_dataset_name}")
        if eval_dataset_name:
            print(f"[trainer]   eval_dataset={eval_dataset_name}")
        if latest_checkpoint is not None:
            print(f"[trainer]   resume_from_checkpoint={latest_checkpoint}")
        print(f"[trainer]   hyperparameters={active_hyperparameters}")
        print(f"[trainer]   output_dir={output_dir}")

        print("[trainer] Attempt 1: offline cache mode (HF_HUB_OFFLINE=1, XET_DISABLE=1)")
        env_offline = _make_env(offline=True)
        with open(train_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== Attempt 1: offline cache mode (model={resolved_model_path}) ===\n")
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env_offline,
                check=False,
                timeout=TRAINING_TIMEOUT_SECONDS,
            )

        if result.returncode != 0:
            print(f"[trainer] Offline training exited with code {result.returncode}, trying online mode...")
            env_online = _make_env(offline=False)

            print("[trainer] Attempt 2: online mode (network access)")
            with open(train_log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n\n=== Attempt 2: online mode (network access) ===\n")
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env_online,
                    check=False,
                    timeout=TRAINING_TIMEOUT_SECONDS,
                )

        if result is not None and result.returncode != 0:
            error_message = f"training exited with code {result.returncode}"
            print(f"[trainer] Training exited with code {result.returncode}")
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== Training exited with return code: {result.returncode} ===\n")

        model_files_exist = _verify_output(output_dir)
        success = bool(result is not None and result.returncode == 0 and model_files_exist)
        if not model_files_exist:
            error_message = error_message or "training produced no loadable model files"
            print(f"[trainer] WARNING: No model files found in {output_dir}")

        # Merge LoRA into base model so vLLM can load it directly
        if success and finetuning_type == "lora":
            _merge_lora_into_base(output_dir, train_log_path, training_config_path=config_path)

    except subprocess.TimeoutExpired:
        error_message = "training timed out after 3600s"
        print("[trainer] Training timed out after 3600s")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write("\n\n=== Training TIMED OUT ===\n")
    except FileNotFoundError:
        error_message = "llamafactory-cli not found"
        print("[trainer] llamafactory-cli not found! Training skipped.")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write("\n\n=== llamafactory-cli not found, training skipped ===\n")
    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        print(f"[trainer] Training error: {type(e).__name__}: {e}")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Training error: {type(e).__name__}: {e} ===\n")

    return {
        "candidate_model_path": str(output_dir),
        "train_log_path": str(train_log_path),
        "success": success,
        "error_message": error_message,
        "trainer_log_jsonl_path": _artifact_path(output_dir, "trainer_log.jsonl"),
        "training_loss_jsonl_path": _artifact_path(output_dir, "training_loss.jsonl"),
        "all_results_path": _artifact_path(output_dir, "all_results.json"),
        "trainer_state_path": _artifact_path(output_dir, "trainer_state.json"),
        "train_results_path": _artifact_path(output_dir, "train_results.json"),
    }


def _artifact_path(output_dir: Path, file_name: str) -> str:
    path = output_dir / file_name
    return str(path) if path.exists() else ""


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    latest: tuple[int, Path] | None = None
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match:
            continue
        step = int(match.group(1))
        if latest is None or step > latest[0]:
            latest = (step, path)
    return latest[1] if latest is not None else None


def _clamp_full_finetune_batch_size(raw_value: object) -> int:
    try:
        batch_size = int(float(raw_value))
    except (TypeError, ValueError):
        batch_size = 1
    return min(2, max(1, batch_size))


def _merge_lora_into_base(
    output_dir: Path,
    train_log_path: Path,
    training_config_path: Path | None = None,
) -> None:
    if _merge_lora_via_export(output_dir, train_log_path, training_config_path=training_config_path):
        return
    _merge_lora_in_process(output_dir, train_log_path)


def _merge_lora_via_export(
    output_dir: Path,
    train_log_path: Path,
    training_config_path: Path | None = None,
) -> bool:
    adapter_path = output_dir / "adapter_config.json"
    if not adapter_path.exists():
        return False

    try:
        cfg = json.loads(adapter_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[trainer] LoRA export skipped: could not read adapter_config.json ({exc})")
        return False

    base_model_path = str(cfg.get("base_model_name_or_path") or "")
    if not base_model_path:
        print("[trainer] LoRA export skipped: no base_model_name_or_path in adapter_config.json")
        return False

    template = _resolve_export_template(training_config_path)
    export_dir = output_dir / "merged"
    export_config_path = output_dir / "export_config.yaml"
    export_config = {
        "model_name_or_path": base_model_path,
        "adapter_name_or_path": str(output_dir),
        "export_dir": str(export_dir),
        "export_size": EXPORT_SIZE_GB,
        "export_device": EXPORT_DEVICE,
        "export_legacy_format": False,
    }
    if template:
        export_config["template"] = template

    _write_config(export_config, export_config_path)
    cmd = ["llamafactory-cli", "export", str(export_config_path)]
    try:
        print(f"[trainer] Exporting LoRA via LLaMA-Factory: {' '.join(cmd)}")
        with open(train_log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== LLaMA-Factory native LoRA export ===\n")
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=_make_env(offline=False),
                check=False,
                timeout=TRAINING_TIMEOUT_SECONDS,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"[trainer] LoRA export failed ({type(exc).__name__}: {exc}), falling back to PEFT merge")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== LoRA export failed: {type(exc).__name__}: {exc} ===\n")
        return False

    if result.returncode != 0:
        print(f"[trainer] LoRA export exited with code {result.returncode}, falling back to PEFT merge")
        return False
    if not _verify_output(export_dir):
        print(f"[trainer] LoRA export produced no loadable files in {export_dir}, falling back to PEFT merge")
        return False

    _copy_exported_model(export_dir, output_dir)
    for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        fpath = output_dir / fname
        if fpath.exists():
            fpath.unlink()
    print(f"[trainer] LoRA export complete -> {export_dir}")
    with open(train_log_path, "a", encoding="utf-8") as f:
        f.write("\n=== LoRA exported via LLaMA-Factory for vLLM compatibility ===\n")
    return True


def _copy_exported_model(export_dir: Path, output_dir: Path) -> None:
    for source in export_dir.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _resolve_export_template(training_config_path: Path | None = None) -> str:
    yaml_module = importlib.import_module("yaml")
    yaml_error = getattr(yaml_module, "YAMLError", Exception)

    for path in (training_config_path, Path(TRAINING_CONFIG_TEMPLATE) if TRAINING_CONFIG_TEMPLATE else None):
        if not path or not path.exists():
            continue
        try:
            loaded = yaml_module.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml_error):
            continue
        if isinstance(loaded, dict) and loaded.get("template"):
            return str(loaded["template"])
    return EXPORT_TEMPLATE


def _merge_lora_in_process(output_dir: Path, train_log_path: Path) -> None:
    adapter_path = output_dir / "adapter_config.json"
    if not adapter_path.exists():
        return

    try:
        torch = importlib.import_module("torch")
        peft_module = importlib.import_module("peft")
        transformers_module = importlib.import_module("transformers")
        PeftModel = peft_module.PeftModel
        AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
        AutoTokenizer = transformers_module.AutoTokenizer

        cfg = json.loads(adapter_path.read_text(encoding="utf-8"))
        base_model_path = cfg.get("base_model_name_or_path", "")
        if not base_model_path:
            print("[trainer] LoRA merge skipped: no base_model_name_or_path in adapter_config.json")
            return

        print(f"[trainer] Merging LoRA adapter into base model for vLLM compatibility...")
        print(f"[trainer]   base_model={base_model_path}")
        print(f"[trainer]   adapter_dir={output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged_model = peft_model.merge_and_unload()

        merged_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Remove LoRA adapter files so vLLM treats this as a full model
        for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            fpath = output_dir / fname
            if fpath.exists():
                fpath.unlink()

        del merged_model, peft_model, base_model, tokenizer
        torch.cuda.empty_cache()

        print(f"[trainer] LoRA merge complete -> {output_dir}")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== LoRA merged into base model for vLLM compatibility ===\n")
    except Exception as exc:
        print(f"[trainer] LoRA merge failed ({type(exc).__name__}: {exc}), vLLM will fall back to HF")
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== LoRA merge failed: {type(exc).__name__}: {exc} ===\n")


def _clear_inference_cache_before_training() -> None:
    """训练前清空推理模型缓存，释放 GPU 显存。

    LLaMA-Factory 训练会吃掉大量显存，不清缓存可能导致 CUDA OOM。
    """
    try:
        from src.tools.model_runner import clear_model_cache

        clear_model_cache()
        print("[trainer] Cleared inference model cache before training")
    except Exception as exc:
        print(f"[trainer] Warning: could not clear inference cache before training: {exc}")


def _load_and_patch_config(
    template_path: str,
    model_name_or_path: str,
    dataset_dir: str,
    dataset_name: str,
    output_dir: str,
    hyperparameters: dict | None = None,
    finetuning_type: str = "full",
    lora_rank: int = 0,
    lora_alpha: int = 0,
    eval_dataset_name: str = "",
    packing: bool | None = None,
    neat_packing: bool | None = None,
    tokenized_path: str | None = None,
) -> dict:
    """读取 YAML 训练配置模板并注入当前轮次的动态参数。

    模板来自 config/templates/llama_factory_sft.yaml，
    运行时覆盖：模型路径、数据集路径、输出目录、微调类型、
    LoRA 参数（仅在 lora 模式下生效）、以及节点传入的超参。
    """
    yaml_module = importlib.import_module("yaml")

    with open(template_path, "r", encoding="utf-8") as f:
        config = yaml_module.safe_load(f)

    config["model_name_or_path"] = model_name_or_path
    config["dataset_dir"] = dataset_dir
    config["dataset"] = dataset_name
    config["output_dir"] = output_dir
    config["finetuning_type"] = finetuning_type
    config["bf16"] = True

    detected_template = detect_model_chat_config(model_name_or_path).get("template_name")
    if detected_template and detected_template != "default":
        config["template"] = detected_template
    if finetuning_type == "lora":
        from config.settings import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES
        config.setdefault("lora_rank", lora_rank if lora_rank > 0 else LORA_RANK)
        config.setdefault("lora_alpha", lora_alpha if lora_alpha > 0 else LORA_ALPHA)
        config.setdefault("lora_dropout", LORA_DROPOUT)
        config.setdefault("lora_target", LORA_TARGET_MODULES)
    for key, value in (hyperparameters or {}).items():
        config[key] = value

    # LoRA parameter normalization is handled by the parameter_master node
    # (_ensure_lora_defaults + _normalize_training_safety), not here. The
    # system decides learning_rate, epochs, and gradient_accumulation based on
    # the code domain and evaluation feedback across rounds.

    if eval_dataset_name:
        config["eval_dataset"] = eval_dataset_name
        config["do_eval"] = True
        config["eval_strategy"] = TRAIN_EVAL_STRATEGY
        config["eval_steps"] = TRAIN_EVAL_STEPS
        config["per_device_eval_batch_size"] = TRAIN_EVAL_BATCH_SIZE
        config["save_strategy"] = TRAIN_EVAL_STRATEGY
        config["save_steps"] = TRAIN_EVAL_STEPS
        config["load_best_model_at_end"] = TRAIN_LOAD_BEST_MODEL_AT_END
        config["metric_for_best_model"] = "eval_loss"
        config["greater_is_better"] = False
        config.pop("val_size", None)
        config.pop("evaluation_strategy", None)
    else:
        config.pop("eval_dataset", None)
        config.pop("do_eval", None)
        config.pop("load_best_model_at_end", None)
        config.pop("metric_for_best_model", None)
        config.pop("greater_is_better", None)
        config.setdefault("eval_strategy", "no")

    if config.get("load_best_model_at_end"):
        config["save_strategy"] = config.get("eval_strategy", TRAIN_EVAL_STRATEGY)
        config["save_steps"] = config.get("eval_steps", TRAIN_EVAL_STEPS)
    else:
        config["save_steps"] = TRAIN_SAVE_STEPS
    if str(finetuning_type).lower() == "full":
        config["per_device_train_batch_size"] = _clamp_full_finetune_batch_size(
            config.get("per_device_train_batch_size", 1)
        )
        config["gradient_checkpointing"] = True
    config["save_total_limit"] = TRAIN_SAVE_TOTAL_LIMIT
    config["save_only_model"] = TRAIN_SAVE_ONLY_MODEL

    active_neat_packing = TRAIN_NEAT_PACKING if neat_packing is None else bool(neat_packing)
    active_packing = TRAIN_PACKING if packing is None else bool(packing)
    if active_neat_packing:
        config["neat_packing"] = True
        config["packing"] = True
    elif active_packing:
        config["packing"] = True

    active_tokenized_path = tokenized_path if tokenized_path is not None else TRAIN_TOKENIZED_PATH
    if active_tokenized_path:
        config["tokenized_path"] = active_tokenized_path

    return config


def _write_config(config: dict, config_path: Path) -> None:
    """将组装好的训练配置字典写入 YAML 文件，供 llamafactory-cli 读取。"""
    yaml_module = importlib.import_module("yaml")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml_module.dump(config, f, default_flow_style=False, allow_unicode=True)


def _verify_output(output_dir: Path) -> bool:
    """验证训练输出目录中是否存在可加载的模型文件。

    检查三类产物：safetensors 权重文件、pytorch bin 文件、LoRA adapter 文件。
    任意一类存在即视为训练产出有效。
    """
    if not output_dir.exists():
        return False

    has_weights = bool(list(output_dir.glob("*.safetensors"))) or \
                 bool(list(output_dir.glob("pytorch_model*.bin"))) or \
                 bool(list(output_dir.glob("model*.safetensors")))
    has_adapter = (output_dir / "adapter_model.safetensors").exists() or \
                  (output_dir / "adapter_model.bin").exists()

    return has_weights or has_adapter
