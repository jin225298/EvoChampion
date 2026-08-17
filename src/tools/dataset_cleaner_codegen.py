from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


_FORBIDDEN_CODE_TOKENS = (
    "open(",
    "__import__",
    "importlib",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "eval(",
    "exec(",
    "compile(",
    "locals(",
    "globals(",
    "vars(",
    "dir(",
    "pathlib",
    "os.",
    "sys.",
    "shutil",
)

_ALLOWED_IMPORTS = {"json", "re", "html", "math", "decimal", "fractions"}
_FORBIDDEN_CALLS = {"open", "__import__", "eval", "exec", "compile", "locals", "globals", "vars", "dir"}
_RESIDUE_TOKENS = (
    "steppq",
    '"P"',
    '"Q"',
    '"depth"',
    '"score"',
    '"logprob"',
    '"reward"',
)

_FORMAL_CODE_MARKERS = (
    "import ",
    "theorem ",
    "lemma ",
    "example ",
    "begin",
    "end",
    ":=",
    "#eval",
    "def ",
    "by ",
    "qed",
    "coq",
    "lean",
    "isabelle",
)

_DEEPSEEK_CLEANER_OUTPUT_MAX_TOKENS = 10000
_NEUTRAL_DOWNSTREAM_CONTRACT = {
    "question": "non-empty user-facing problem statement",
    "answer": "concise final answer only when the source has a real answer field; empty string for judge-only proof/solution rows",
    "process": "visible solution/explanation/proof text; empty string if absent",
    "evaluation_method": '"gold" when answer is a real source answer, or "llm_judge" when no real gold answer exists',
    "needs_judge": "true when evaluation_method is llm_judge, otherwise false",
    "think": "explicit hidden reasoning/thinking text from the source row; empty string if absent",
    "rejected": '{"status": "rejected", "reject_reason": "..."} for unusable rows',
}

_CLEANER_SEMANTIC_RULES_PROMPT = (
    "Semantic cleaning rules:\n"
    "- question must be a self-contained problem or task that a model can answer without hidden dataset context.\n"
    "- question must not be only a name/id/slug/filename or opaque row identifier.\n"
    "- Never use identifiers like correct_by_msg__..._round1 as question.\n"
    "- Reject rows where answer == question.\n"
    "- If a row only has name + formal_proof, and no natural-language problem statement or formal theorem "
    "statement can be safely separated from the proof, return rejected; do not invent a question.\n"
    "- For rows with informal_statement + informal_proof + formal_proof, map exactly: "
    "question = informal_statement, answer/gold_answer/rollout_gold_answer = empty string, "
    "process/train_output = informal_proof, evaluation_method = llm_judge, needs_judge = true, "
    "think = an explicit source thinking field if one exists, otherwise an empty string. "
    "formal_proof is proof/code reference material, not a final-answer gold string for rollout matching.\n"
    "- For rows with only informal_statement + informal_proof and no formal_proof, map exactly: "
    "question = informal_statement, process/train_output = informal_proof, "
    "answer/gold_answer/rollout_gold_answer = empty string, evaluation_method = llm_judge, "
    "needs_judge = true. Do not extract or infer a final answer from informal_proof. "
    "Never use informal_statement as answer. Never use the whole informal_proof as answer; "
    "it belongs in process/train_output so a later LLM judge can compare against it.\n"
    "- For rows with only formal_proof, reject unless the code can safely split a formal theorem statement/task "
    "from the proof text; do not use the whole proof as both question context and answer.\n"
    "- If a row has only id/name/slug/filename/title plus answer/formal_proof/proof-like text and no question, "
    "problem, informal_statement, or dedicated formal theorem statement field, return rejected. "
    "Values from name/id/slug/filename/title fields are identifiers, not questions, when they look like snake_case, "
    "camelCase, paths, filenames, opaque labels, or names with numeric suffixes such as nat_add_case_17. "
    "Do not derive a question by extracting a variable, theorem name, opaque fragment, or tiny proposition "
    "such as x from answer/proof text. A formal theorem statement must include the actual proposition/task, "
    "not just a theorem name and not just a proof body like by ring.\n"
    "- answer must be the final answer used for evaluation only when the source row has a real answer/final-answer "
    "field. Formal proof/code fields such as formal_proof, Lean, Coq, or Isabelle proof text are not short "
    "final answers for rollout matching. process must hold the visible solution or explanation. "
    "If no real answer/final-answer field exists and the row only has a solution/proof/process, leave answer empty "
    "and set evaluation_method = llm_judge with needs_judge = true. "
    "Do not put the whole assistant response into answer when it contains both solution steps and a final answer; "
    "split the solution steps into process and keep answer concise.\n"
    "- For chat or instruction-template rows, parse structural markers before stripping them. "
    "For [INST]...[/INST] text, use a non-greedy span inside [INST] and before [/INST] as question, "
    "and use only the text after [/INST] as assistant/output for process and answer. "
    "For human:/assistant: text, question is before assistant: and assistant/output is after assistant:. "
    "Never remove chat tags first and then split by periods; never use a first-sentence/period heuristic when "
    "template markers exist. Never append assistant/output text to question, and preserve whitespace at marker boundaries. "
    "After extracting spans, strip tags from each field. "
    "If assistant/output contains reasoning plus a final answer, put reasoning in process and final answer in answer. "
    "If assistant/output has no explicit final answer marker, infer answer only when a final value is unambiguous, "
    "such as a boxed value, the right side of a final equation, or a clear last standalone numeric/algebraic value; "
    "otherwise reject instead of inventing. Keep the full visible solution in process when it is used. "
    "Never build process by naively subtracting/removing the answer string from the assistant/output span; "
    "do not call replace(answer, '') or similar on process. Do not copy the question into process. "
    "If the assistant text says 4 * 5 = 20, process must preserve the complete expression 4 * 5 = 20. "
    "process must remain a grammatical, complete solution; it must not end with operators such as =, +, -, *, /, "
    "and must not contain broken fragments such as The . or final answer is . "
    "Keep necessary computed results inside process; it is okay for the final answer to appear in both process and answer.\n"
    "- Strip all chat/template residue from question, answer, and process, including <s>, </s>, "
    "[INST], [/INST], human:, and assistant:.\n"
    "- answer should contain only the final answer plus necessary solution steps. Remove boilerplate such as "
    "hope you enjoyed, thank you, fictional example, response formulation, post-response reflection, "
    "and similar meta commentary. Remove complete boilerplate sentences before extracting process/answer, "
    "so no fragments such as this ., The 2. this ., or isolated ! remain in process. Collapse repeated spaces.\n"
    "- Remove search/reward metadata from all output fields, including steppq, P/Q/depth, "
    "score/logprob/reward/depth, and trailing diagnostic key-value fragments. Treat metadata markers as cut points: "
    "if steppq, score=, logprob=, reward=, depth=, or P/Q/depth appears in answer/process/think, "
    "drop that marker and everything after it in that field. For Answer/Final fields, extract only the content "
    "before the first metadata marker or following diagnostic line.\n"
)

_CLEANER_CODE_SYSTEM_PROMPT = (
    "You generate deterministic Python dataset cleaner code. Return JSON only. "
    'The JSON object must contain exactly one code field, shaped like {"code": "...python source..."}. '
    "Do not return markdown, prose, or already-cleaned sample rows. "
    "The code must define clean_record(record: dict, context: dict | None = None) -> dict. "
    "clean_record must return a dict. Do not return None. "
    'For usable rows return {"status": "cleaned", "question": "...", "answer": "...", '
    '"process": "...", "think": "...", "evaluation_method": "gold"}. '
    'For rows with no real source answer and only a solution/proof/process, return '
    '{"status": "cleaned", "question": "...", "answer": "", "process": "...", "think": "", '
    '"evaluation_method": "llm_judge", "needs_judge": true}. '
    'For unusable rows return {"status": "rejected", "reject_reason": "..."}. '
    "question must be non-empty for cleaned rows. answer must be non-empty only when evaluation_method is gold; "
    "for evaluation_method llm_judge, answer must be empty and process or train_output must be non-empty. "
    "process and think must be strings and may be empty. "
    "Extract question/answer even when they are mixed in one column. "
    "Remove irrelevant metadata or reward/search traces such as steppq, P/Q/depth/score/logprob/reward. "
    "Use process for visible solution/explanation. Use think only when the source row explicitly contains "
    "hidden reasoning, chain-of-thought, rationale, or <think>...</think> content. "
    + _CLEANER_SEMANTIC_RULES_PROMPT +
    "Do not emit LLaMA-Factory fields, chat template tags, system prompts, instruction/input/output, "
    "question_text, gold_answer, rollout_gold_answer, or target_style. "
    "You may emit train_output only when it is the same visible solution/proof text as process. "
    "Use safe Python stdlib only. Do not use file, network, subprocess, eval, exec, compile, "
    "locals, globals, vars, dir, reflection, importlib, os, sys, pathlib, shell access, or dynamic imports. "
    "Update named variables directly instead of trying to mutate locals()."
)


@dataclass(frozen=True)
class CleanerCodegenRequest:
    dataset_id: str
    source: str
    subset: str | None
    split: str
    user_goal: str
    dataset_card: dict[str, Any]
    source_dataset_columns: list[str]
    source_dataset_first_row: dict[str, Any]
    source_dataset_raw_rows: list[dict[str, Any]]
    normalized_samples: list[dict[str, Any]]
    cache_root: Path


@dataclass
class CleanerValidationReport:
    status: str
    cleaned_count: int = 0
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    cleaned_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "cleaned_count": self.cleaned_count,
            "rejected_count": self.rejected_count,
            "errors": list(self.errors),
            "reject_reasons": dict(self.reject_reasons),
            "cleaned_examples": list(self.cleaned_examples),
        }


@dataclass
class CleanerCodegenResult:
    status: str
    cleaner_id: str = ""
    spec_path: str = ""
    code_path: str = ""
    validation_report_path: str = ""
    cleaner_cache_ref: dict[str, Any] = field(default_factory=dict)
    schema_override: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


class CleanerCodegenProvider(Protocol):
    def generate_spec(self, request: CleanerCodegenRequest) -> dict[str, Any]: ...

    def generate_code(self, request: CleanerCodegenRequest, spec: dict[str, Any]) -> str: ...

    def repair_code(
        self,
        request: CleanerCodegenRequest,
        spec: dict[str, Any],
        code: str,
        errors: list[str],
    ) -> str: ...


class LocalModelCleanerCodegenProvider:
    def generate_spec(self, request: CleanerCodegenRequest) -> dict[str, Any]:
        raise RuntimeError("local cleaner provider unavailable")

    def generate_code(self, request: CleanerCodegenRequest, spec: dict[str, Any]) -> str:
        raise RuntimeError("local cleaner provider unavailable")

    def repair_code(
        self,
        request: CleanerCodegenRequest,
        spec: dict[str, Any],
        code: str,
        errors: list[str],
    ) -> str:
        raise RuntimeError("local cleaner provider unavailable")


class DeepSeekCleanerCodegenProvider:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _chat_json(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for deepseek cleaner provider")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])

    def generate_spec(self, request: CleanerCodegenRequest) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You design conservative deterministic dataset cleaner specs. "
                    "Return JSON only, with no prose or markdown. The downstream cleaner "
                    "contract is question, answer, process, and think. "
                    + _CLEANER_SEMANTIC_RULES_PROMPT
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a cleaner spec for this dataset. Prefer rejecting ambiguous rows "
                    "over inventing data. Context JSON:\n"
                    + _request_context_for_prompt(request)
                ),
            },
        ]
        return _extract_json_object(self._chat_json(messages, max_tokens=_DEEPSEEK_CLEANER_OUTPUT_MAX_TOKENS))

    def generate_code(self, request: CleanerCodegenRequest, spec: dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": _CLEANER_CODE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Write the cleaner implementation for this spec and context.\n"
                    "Spec JSON:\n"
                    + json.dumps(spec, ensure_ascii=True, sort_keys=True)
                    + "\nContext JSON:\n"
                    + _request_context_for_prompt(request)
                ),
            },
        ]
        return self._chat_json(messages, max_tokens=_DEEPSEEK_CLEANER_OUTPUT_MAX_TOKENS)

    def repair_code(
        self,
        request: CleanerCodegenRequest,
        spec: dict[str, Any],
        code: str,
        errors: list[str],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": _CLEANER_CODE_SYSTEM_PROMPT + " Repair the cleaner while preserving the same contract.",
            },
            {
                "role": "user",
                "content": (
                    "Repair this cleaner to satisfy validation.\n"
                    "Errors JSON:\n"
                    + json.dumps(errors, ensure_ascii=True)
                    + "\nSpec JSON:\n"
                    + json.dumps(spec, ensure_ascii=True, sort_keys=True)
                    + "\nCurrent code:\n"
                    + code
                    + "\nContext JSON:\n"
                    + _request_context_for_prompt(request)
                ),
            },
        ]
        return self._chat_json(messages, max_tokens=_DEEPSEEK_CLEANER_OUTPUT_MAX_TOKENS)


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _looks_like_formal_proof_code(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    hits = sum(1 for marker in _FORMAL_CODE_MARKERS if marker in text)
    return hits >= 2 or ("import " in text and ("begin" in text or ":=" in text))


def _normalised_context_columns(context: dict[str, Any] | None) -> list[str]:
    return [str(column).strip().lower() for column in (context or {}).get("columns", [])]


def _has_formal_proof_source(context: dict[str, Any] | None) -> bool:
    return any(
        "formal_proof" in column
        or "lean_proof" in column
        or "coq_proof" in column
        or "isabelle_proof" in column
        for column in _normalised_context_columns(context)
    )


def _has_natural_proof_reference(context: dict[str, Any] | None) -> bool:
    return any(
        column in {"informal_proof", "solution", "reasoning", "rationale", "explanation"}
        for column in _normalised_context_columns(context)
    )


def _natural_proof_reference_from_raw_row(context: dict[str, Any] | None) -> str:
    row = (context or {}).get("raw_row")
    if not isinstance(row, dict):
        return ""
    for key in ("informal_proof", "solution", "reasoning", "rationale", "explanation"):
        value = row.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _request_context_for_prompt(request: CleanerCodegenRequest) -> str:
    context = {
        "dataset_id": request.dataset_id,
        "subset": request.subset,
        "split": request.split,
        "user_goal": request.user_goal,
        "dataset_card": request.dataset_card,
        "source_dataset_columns": request.source_dataset_columns,
        "source_dataset_first_row": request.source_dataset_first_row,
        "source_dataset_raw_rows": request.source_dataset_raw_rows[:3],
        "normalized_samples": request.normalized_samples[:3],
        "downstream_contract": _NEUTRAL_DOWNSTREAM_CONTRACT,
    }
    return json.dumps(context, ensure_ascii=True, sort_keys=True, default=str)


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    raw = text or ""
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output must contain a JSON object")


def cleaner_fingerprint(request: CleanerCodegenRequest) -> str:
    first_row_shape = {
        key: type(value).__name__
        for key, value in dict(request.source_dataset_first_row).items()
    }
    return _stable_json_hash(
        {
            "dataset_id": request.dataset_id,
            "subset": request.subset,
            "split": request.split,
            "columns": list(request.source_dataset_columns),
            "first_row_shape": first_row_shape,
        }
    )


def _cache_dir(request: CleanerCodegenRequest) -> Path:
    return Path(request.cache_root) / "dataset_cleaners" / cleaner_fingerprint(request)


def _ready_result_from_cache(cache_dir: Path) -> CleanerCodegenResult | None:
    spec_path = cache_dir / "cleaner_spec.json"
    code_path = cache_dir / "cleaner.py"
    report_path = cache_dir / "validation_report.json"
    if not (spec_path.exists() and code_path.exists() and report_path.exists()):
        return None

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("status") != "ready":
        return None

    cleaner_id = cache_dir.name
    cache_ref = {
        "status": "ready",
        "cleaner_id": cleaner_id,
        "spec_path": str(spec_path),
        "code_path": str(code_path),
        "validation_report_path": str(report_path),
    }
    return CleanerCodegenResult(
        status="ready",
        cleaner_id=cleaner_id,
        spec_path=str(spec_path),
        code_path=str(code_path),
        validation_report_path=str(report_path),
        cleaner_cache_ref=cache_ref,
    )


def ensure_cleaner_for_ref(
    request: CleanerCodegenRequest,
    *,
    provider: CleanerCodegenProvider,
    max_repair_attempts: int,
) -> CleanerCodegenResult:
    cache_dir = _cache_dir(request)
    cached = _ready_result_from_cache(cache_dir)
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path = cache_dir / "sample_rows.json"
    dataset_card_path = cache_dir / "dataset_card.json"
    spec_path = cache_dir / "cleaner_spec.json"
    code_path = cache_dir / "cleaner.py"
    report_path = cache_dir / "validation_report.json"
    cleaner_id = cache_dir.name

    sample_path.write_text(
        json.dumps(request.source_dataset_raw_rows, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    dataset_card_path.write_text(
        json.dumps(request.dataset_card, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    try:
        spec = provider.generate_spec(request)
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True, default=str), encoding="utf-8")

        raw_code = provider.generate_code(request, spec)
        code, report = _extract_and_validate_cleaner(request, raw_code)
        attempts = 0
        while report.status != "ready" and attempts < max(0, int(max_repair_attempts)):
            attempts += 1
            raw_code = provider.repair_code(request, spec, code, list(report.errors))
            code, report = _extract_and_validate_cleaner(request, raw_code)

        code_path.write_text(code, encoding="utf-8")
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")

        status = report.status
        cache_ref = {
            "status": status,
            "cleaner_id": cleaner_id,
            "spec_path": str(spec_path),
            "code_path": str(code_path),
            "validation_report_path": str(report_path),
        }
        schema_override = {}
        target_style = str(spec.get("target_style") or "")
        if status == "ready" and target_style in {"answer", "cot"}:
            schema_override["target_style"] = target_style

        return CleanerCodegenResult(
            status=status,
            cleaner_id=cleaner_id,
            spec_path=str(spec_path),
            code_path=str(code_path),
            validation_report_path=str(report_path),
            cleaner_cache_ref=cache_ref if status == "ready" else {},
            schema_override=schema_override,
            failure_reason="" if status == "ready" else "; ".join(report.errors),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failed_report = CleanerValidationReport(status="failed", errors=[error])
        report_path.write_text(
            json.dumps(failed_report.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return CleanerCodegenResult(
            status="failed",
            cleaner_id=cleaner_id,
            spec_path=str(spec_path),
            code_path=str(code_path),
            validation_report_path=str(report_path),
            failure_reason=error,
        )


def _extract_and_validate_cleaner(
    request: CleanerCodegenRequest,
    raw_code: str,
) -> tuple[str, CleanerValidationReport]:
    try:
        code = extract_python_code(raw_code)
    except ValueError as exc:
        code = raw_code
        return code, CleanerValidationReport(status="failed", errors=[str(exc)])
    return code, validate_generated_cleaner(request, code)


def extract_python_code(text: str) -> str:
    try:
        payload = _extract_json_object(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        for field_name in ("code", "python_code", "clean_record", "code_markdown"):
            field_value = payload.get(field_name)
            if isinstance(field_value, str) and "def clean_record" in field_value:
                text = field_value
                break

    matches = _extract_line_fenced_python_blocks(text or "")
    if len(matches) == 1:
        code = matches[0].strip() + "\n"
    elif len(matches) == 0 and "def clean_record" in (text or ""):
        code = (text or "").strip() + "\n"
    else:
        raise ValueError("model output must contain exactly one fenced Python block")
    if "def clean_record" not in code:
        raise ValueError("cleaner code must define clean_record")
    return code


def _extract_line_fenced_python_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip().lower()
        if line not in {"```python", "```py"}:
            idx += 1
            continue
        idx += 1
        block_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != "```":
            block_lines.append(lines[idx])
            idx += 1
        if idx < len(lines) and lines[idx].strip() == "```":
            blocks.append("\n".join(block_lines))
        idx += 1
    return blocks


def validate_cleaner_code_safety(code: str) -> list[str]:
    errors: list[str] = []
    lowered = code.lower()
    for token in _FORBIDDEN_CODE_TOKENS:
        if token.lower() in lowered:
            errors.append(f"forbidden token: {token}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return errors + [f"syntax error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORTS:
                errors.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                errors.append(f"forbidden call: {node.func.id}")
    return errors


def _load_cleaner_function(code: str):
    safety_errors = validate_cleaner_code_safety(code)
    if safety_errors:
        raise ValueError("; ".join(safety_errors))

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if root not in _ALLOWED_IMPORTS:
            raise ImportError(f"import not allowed in generated cleaner: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "getattr": getattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    exec(code, namespace)
    clean_record = namespace.get("clean_record")
    if not callable(clean_record):
        raise ValueError("clean_record is not callable")
    return clean_record


def _valid_cleaned_record(result: dict[str, Any]) -> list[str]:
    if result.get("status") != "cleaned":
        return []

    result = _normalize_cleaned_record(result)
    errors: list[str] = []
    if not str(result.get("question_text") or "").strip():
        errors.append("missing question_text")
    gold_answer = str(result.get("gold_answer") or "").strip()
    rollout_gold_answer = str(result.get("rollout_gold_answer") or "").strip()
    train_output = str(result.get("train_output") or "").strip()
    process = str(result.get("process") or "").strip()
    evaluation_method = str(result.get("evaluation_method") or "").strip()
    if evaluation_method == "gold" and not (gold_answer or rollout_gold_answer):
        errors.append("missing gold answer")
    if evaluation_method == "llm_judge" and not (train_output or process):
        errors.append("llm_judge row missing reference solution")
    if str(result.get("target_style") or "") not in {"answer", "cot"}:
        errors.append("invalid target_style")

    for field_name in ("train_output", "process", "think"):
        field_value = str(result.get(field_name) or "")
        if field_name == "train_output" and field_value.lstrip().startswith(("[", "{")):
            errors.append("train_output has raw structured residue")
        field_value_lower = field_value.lower()
        for token in _RESIDUE_TOKENS:
            if token.lower() in field_value_lower:
                errors.append(f"{field_name} contains residue token: {token}")
    return errors


def _normalize_cleaned_record(result: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    if result.get("status") != "cleaned":
        return dict(result)

    normalized = dict(result)
    question = str(
        result.get("question_text")
        or result.get("question")
        or result.get("input")
        or result.get("instruction")
        or ""
    ).strip()
    answer = str(result.get("answer") or result.get("output") or "").strip()
    explicit_gold_answer = str(result.get("gold_answer") or "").strip()
    explicit_rollout_gold_answer = str(result.get("rollout_gold_answer") or "").strip()
    process = str(result.get("process") or result.get("solution") or result.get("explanation") or "").strip()
    think = str(result.get("think") or result.get("thinking") or result.get("reasoning") or "").strip()
    train_output = str(result.get("train_output") or process or think or answer).strip()
    gold_answer = explicit_gold_answer or answer
    rollout_gold_answer = explicit_rollout_gold_answer or gold_answer
    formal_proof_answer = (
        _has_formal_proof_source(context)
        and _has_natural_proof_reference(context)
        and _looks_like_formal_proof_code(gold_answer or rollout_gold_answer)
        and (process or think or train_output)
    )
    if formal_proof_answer:
        natural_reference = _natural_proof_reference_from_raw_row(context)
        answer = ""
        gold_answer = ""
        rollout_gold_answer = ""
        train_output = natural_reference or process or think
        process = process or natural_reference
    requested_method = str(result.get("evaluation_method") or result.get("judge_mode") or "").strip().lower()
    requested_needs_judge = result.get("needs_judge")
    if requested_method not in {"gold", "llm_judge"}:
        requested_method = ""
    evaluation_method = "gold" if (gold_answer or rollout_gold_answer) else "llm_judge"
    if formal_proof_answer:
        requested_method = "llm_judge"
    if requested_method == "llm_judge" and not (gold_answer or rollout_gold_answer):
        evaluation_method = "llm_judge"
    if requested_method == "gold" and (gold_answer or rollout_gold_answer):
        evaluation_method = "gold"
    needs_judge = _clean_bool(requested_needs_judge) or evaluation_method == "llm_judge"
    target_style = str(result.get("target_style") or "").strip().lower()
    if target_style not in {"answer", "cot"}:
        target_style = "cot" if (process or think or (train_output and gold_answer and train_output != gold_answer)) else "answer"

    normalized.update(
        {
            "question": question,
            "answer": answer,
            "process": process,
            "think": think,
            "question_text": question,
            "gold_answer": gold_answer,
            "rollout_gold_answer": rollout_gold_answer,
            "train_output": train_output,
            "target_style": target_style,
            "evaluation_method": evaluation_method,
            "needs_judge": needs_judge,
        }
    )
    return normalized


def validate_generated_cleaner(request: CleanerCodegenRequest, code: str) -> CleanerValidationReport:
    safety_errors = validate_cleaner_code_safety(code)
    if safety_errors:
        return CleanerValidationReport(status="failed", errors=safety_errors)

    try:
        clean_record = _load_cleaner_function(code)
    except Exception as exc:
        return CleanerValidationReport(status="failed", errors=[f"load failed: {type(exc).__name__}: {exc}"])

    cleaned: list[dict[str, Any]] = []
    rejected_count = 0
    reject_reasons: dict[str, int] = {}
    errors: list[str] = []
    context = {"dataset_id": request.dataset_id, "columns": list(request.source_dataset_columns)}

    for row in request.source_dataset_raw_rows:
        row_context = {**context, "raw_row": row if isinstance(row, dict) else {}}
        result = _run_cleaner(clean_record, row, row_context)
        if result.get("status") == "cleaned":
            normalized = _normalize_cleaned_record(result, row_context)
            row_errors = _valid_cleaned_record(normalized)
            if row_errors:
                rejected_count += 1
                reason = "; ".join(row_errors)
                errors.extend(row_errors)
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            else:
                cleaned.append(normalized)
        else:
            rejected_count += 1
            reason = str(result.get("reject_reason") or "rejected")
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    if not cleaned:
        errors.append("cleaner produced no cleaned sample")

    return CleanerValidationReport(
        status="ready" if cleaned and not errors else "failed",
        cleaned_count=len(cleaned),
        rejected_count=rejected_count,
        errors=errors,
        reject_reasons=reject_reasons,
        cleaned_examples=cleaned[:3],
    )


def apply_cleaner_to_rows(
    rows: Any,
    cleaner_cache_ref: dict[str, Any],
    *,
    dataset_id: str,
    source_dataset_split: str | None,
    source_dataset_subset: str | None,
    source_dataset_requested_split: str | None,
    source_dataset_split_names: list[str],
    source_dataset_columns: list[str],
    source_dataset_first_row: dict[str, Any],
    source_dataset_schema: dict[str, Any],
    max_items: int | None = None,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    code_path = Path(str(cleaner_cache_ref.get("code_path") or ""))
    code = code_path.read_text(encoding="utf-8")
    clean_record = _load_cleaner_function(code)

    questions: list[dict[str, Any]] = []
    rejected_count = 0
    processed_count = 0
    context = {"dataset_id": dataset_id, "columns": list(source_dataset_columns)}
    schema = {**source_dataset_schema, "cleaner_cache_ref": dict(cleaner_cache_ref)}

    for row_idx, row in enumerate(rows):
        if row_idx < offset:
            continue
        processed_count += 1
        row_context = {**context, "raw_row": row if isinstance(row, dict) else {}}
        result = _run_cleaner(clean_record, row, row_context)
        normalized = _normalize_cleaned_record(result, row_context)
        if normalized.get("status") == "cleaned" and not _valid_cleaned_record(normalized):
            questions.append(
                _standard_question_from_cleaned(
                    normalized,
                    row_idx,
                    dataset_id=dataset_id,
                    source_dataset_split=source_dataset_split,
                    source_dataset_subset=source_dataset_subset,
                    source_dataset_requested_split=source_dataset_requested_split,
                    source_dataset_split_names=source_dataset_split_names,
                    source_dataset_columns=source_dataset_columns,
                    source_dataset_first_row=source_dataset_first_row,
                    source_dataset_schema=schema,
                )
            )
            if max_items is not None and len(questions) >= max_items:
                break
        else:
            rejected_count += 1

    return questions, {
        "cleaned_count": len(questions),
        "rejected_count": rejected_count,
        "processed_count": processed_count,
    }


def _run_cleaner(clean_record: Any, row: Any, context: dict[str, Any]) -> dict[str, Any]:
    try:
        result = clean_record(dict(row), dict(context))
    except Exception as exc:
        return {"status": "rejected", "reject_reason": f"exception: {type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        return {"status": "rejected", "reject_reason": "clean_record returned non-dict"}
    return result


def _question_id_for_cleaned_row(dataset_id: str, idx: int, subset: str | None, split: str | None) -> str:
    parts = [_question_id_part(dataset_id)]
    if subset:
        parts.append(_question_id_part(str(subset)))
    if split:
        parts.append(_question_id_part(str(split)))
    parts.append(str(idx))
    return "_".join(parts)


def _question_id_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _standard_question_from_cleaned(
    cleaned: dict[str, Any],
    idx: int,
    *,
    dataset_id: str,
    source_dataset_split: str | None,
    source_dataset_subset: str | None,
    source_dataset_requested_split: str | None,
    source_dataset_split_names: list[str],
    source_dataset_columns: list[str],
    source_dataset_first_row: dict[str, Any],
    source_dataset_schema: dict[str, Any],
) -> dict[str, Any]:
    cleaned = _normalize_cleaned_record(cleaned, {"columns": list(source_dataset_columns)})
    question_text = str(cleaned.get("question_text") or "").strip()
    gold_answer = str(cleaned.get("gold_answer") or "").strip()
    rollout_gold_answer = str(cleaned.get("rollout_gold_answer") or "").strip()
    train_output = str(
        cleaned.get("train_output")
        or cleaned.get("process")
        or cleaned.get("think")
        or rollout_gold_answer
        or gold_answer
    ).strip()
    # Code-domain fields: prefer the cleaner output, fall back to the raw row.
    first_row = source_dataset_first_row if isinstance(source_dataset_first_row, dict) else {}
    code_test = str(
        cleaned.get("test") or cleaned.get("code_test")
        or first_row.get("test") or first_row.get("code_test") or ""
    ).strip()
    code_entry_point = str(
        cleaned.get("entry_point") or cleaned.get("entry_point_func")
        or first_row.get("entry_point") or first_row.get("entry_point_func") or ""
    ).strip()
    evaluation_method = str(cleaned.get("evaluation_method") or "").strip()
    if evaluation_method not in {"gold", "llm_judge", "code_exec"}:
        if code_test and code_entry_point:
            evaluation_method = "code_exec"
        else:
            evaluation_method = "gold" if (gold_answer or rollout_gold_answer) else "llm_judge"
    needs_judge = _clean_bool(cleaned.get("needs_judge")) or evaluation_method == "llm_judge"
    schema = dict(source_dataset_schema)
    marker = cleaned.get("final_answer_marker")
    if marker:
        schema["final_answer_marker"] = str(marker)

    return {
        "question_id": _question_id_for_cleaned_row(dataset_id, idx, source_dataset_subset, source_dataset_split),
        "question_text": question_text,
        "gold_answer": gold_answer,
        "rollout_gold_answer": rollout_gold_answer,
        "train_output": train_output,
        "target_style": str(cleaned.get("target_style") or "answer"),
        "evaluation_method": evaluation_method,
        "needs_judge": needs_judge,
        "process": str(cleaned.get("process") or "").strip(),
        "think": str(cleaned.get("think") or "").strip(),
        "dedup_key": question_text,
        "source_dataset_id": dataset_id,
        "source_dataset_row_id": str(idx),
        "source_dataset_split": source_dataset_split,
        "source_dataset_subset": source_dataset_subset,
        "source_dataset_requested_split": source_dataset_requested_split,
        "source_dataset_split_names": list(source_dataset_split_names),
        "source_dataset_columns": list(source_dataset_columns),
        "source_dataset_first_row": dict(source_dataset_first_row),
        "source_dataset_schema": schema,
        "test": code_test,
        "entry_point": code_entry_point,
    }


def build_cleaner_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> CleanerCodegenProvider | None:
    from config import settings

    selected = (provider_name if provider_name is not None else settings.DATA_CLEANER_PROVIDER).strip().lower()
    if selected == "off":
        return None
    if selected == "local":
        return LocalModelCleanerCodegenProvider()
    if selected == "deepseek":
        return DeepSeekCleanerCodegenProvider(
            api_key=api_key if api_key is not None else settings.DEEPSEEK_API_KEY,
            base_url=base_url if base_url is not None else settings.DEEPSEEK_BASE_URL,
            model=model if model is not None else settings.DEEPSEEK_CLEANER_MODEL,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else settings.DATA_CLEANER_REQUEST_TIMEOUT_SECONDS
            ),
        )
    return None


# Passthrough cleaner for already-normalized local code datasets (e.g. the
# code_smoke benchmark). Such a dataset already has question/answer/test/
# entry_point columns, so no LLM/DeepSeek codegen is needed: the cleaner just
# maps the raw row to the standard cleaned shape, carrying the executable test
# and entry_point through for execution-based judging.
_PASSTHROUGH_CODE_CLEANER = '''def clean_record(row, context):
    question = row.get("question", row.get("input", row.get("problem", "")))
    answer = row.get("answer", row.get("output", row.get("target", "")))
    test = row.get("test", row.get("code_test", ""))
    entry_point = row.get("entry_point", row.get("entry_point_func", ""))
    return {
        "status": "cleaned",
        "question_text": str(question),
        "gold_answer": str(answer),
        "rollout_gold_answer": str(answer),
        "train_output": str(answer),
        "target_style": "answer",
        "evaluation_method": "code_exec",
        "test": str(test),
        "entry_point": str(entry_point),
    }
'''


def build_passthrough_code_cleaner_ref(cache_dir, dataset_id: str = "") -> dict[str, Any]:
    """Write and validate a passthrough cleaner for a local code dataset.

    Returns a ready ``cleaner_cache_ref`` (``{status, code_path, ...}``) so the
    screening path can load an already-normalized code dataset without an LLM
    or DeepSeek cleaner provider.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    code_path = cache_dir / "passthrough_code_cleaner.py"
    code_path.write_text(_PASSTHROUGH_CODE_CLEANER, encoding="utf-8")
    # Validate the generated cleaner loads under the sandboxed builtins.
    _load_cleaner_function(_PASSTHROUGH_CODE_CLEANER)
    return {
        "status": "ready",
        "code_path": str(code_path),
        "provider": "passthrough_code",
        "dataset_id": str(dataset_id),
    }


def is_local_code_dataset(ref_dataset_id: str, samples: list[dict]) -> bool:
    """True when a dataset ref points at a local code dataset with test+entry_point.

    Used to auto-accept already-normalized local code datasets (e.g. the
    code_smoke benchmark) without requiring an LLM reviewer or cleaner provider.
    """
    if not ref_dataset_id:
        return False
    path = Path(ref_dataset_id).expanduser()
    if not path.exists():
        return False
    for sample in samples:
        if isinstance(sample, dict) and sample.get("test") and sample.get("entry_point"):
            return True
    return False
