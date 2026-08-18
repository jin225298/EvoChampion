"""
Unified message protocol for the EvoChampion system.
All communication between components must use these standardized message formats.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    SYSTEM = "system"
    ROUTER = "router"
    HARNESS = "harness"
    TEACHER = "teacher"
    SEARCHER = "searcher"
    HF_SEARCH_TOOL = "hf_search_tool"
    SCREENING_ENTRY = "screening_entry"
    DATASET_SCHEMA_AGENT = "dataset_schema_agent"
    DATASET_REVIEWER = "dataset_reviewer"
    ROLLOUT_DISPATCHER = "rollout_dispatcher"
    ROLLOUT_WORKER = "rollout_worker"
    ROLLOUT_AGGREGATOR = "rollout_aggregator"
    FILTER = "filter"
    CLASSIFIER = "classifier"
    DATA_BUILDER = "data_builder"
    PARAMETER_MASTER = "parameter_master"
    TRAINER = "trainer"
    EVALUATOR = "evaluator"
    STRATEGY_INSPECTOR = "strategy_inspector"
    DIAGNOSTIC = "diagnostic"
    FORMAT_ERROR = "format_error"


class MessageType(str, Enum):
    GOAL_REQUEST = "goal_request"
    SEARCH_REQUEST = "search_request"
    SEARCH_RESULT = "search_result"
    DATASET_SCHEMA_REQUEST = "dataset_schema_request"
    DATASET_SCHEMA_RESULT = "dataset_schema_result"
    MATERIALIZED_DATASET = "materialized_dataset"
    ROLLOUT_REQUEST = "rollout_request"
    ROLLOUT_TASK = "rollout_task"
    ROLLOUT_TASK_RESULT = "rollout_task_result"
    ROLLOUT_RESULT = "rollout_result"
    FILTER_RESULT = "filter_result"
    CLASSIFICATION_RESULT = "classification_result"
    DATASET_BUNDLE = "dataset_bundle"
    DATASET_REVIEW_RESULT = "dataset_review_result"
    DIAGNOSTIC_REQUEST = "diagnostic_request"
    TRAIN_REQUEST = "train_request"
    TRAIN_RESULT = "train_result"
    DIAGNOSTIC_RESULT = "diagnostic_result"
    EVAL_RESULT = "eval_result"
    INSPECTION_RESULT = "inspection_result"
    FORMAT_ERROR = "format_error"


class RouteStatus(str, Enum):
    PENDING = "待校验"
    ACCEPTED = "接受并中转"
    REJECTED = "拒绝中转"


class MessageHeader(BaseModel):
    trace_id: str
    round_id: int
    sender: AgentName
    receiver: AgentName
    message_type: MessageType
    route_status: RouteStatus = RouteStatus.PENDING
    schema_version: str = "v1"


# =============================================================================
# Payload Definitions
# =============================================================================

# 机制层只关心消息结构，策略层只关心如何决策。大数据列表通过
# DataArtifactRef 传地址，避免 router 消息文件保存整批题目。
class DataArtifactRef(BaseModel):
    artifact_id: str
    local_path: str
    count: int = 0
    kind: str = "json"
    content_type: str = "application/json"
    schema_version: str = "v1"


# 确保每个 Payload 都包含必要字段，字段类型明确，便于后续处理和验证。
class GoalRequestPayload(BaseModel):
    goal: str


class QuestionPayload(BaseModel):
    question_id: str
    question_text: str
    gold_answer: str
    rollout_gold_answer: str = ""
    train_output: str = ""
    target_style: Literal["answer", "cot"] = "answer"
    evaluation_method: Literal["gold", "llm_judge", "code_execution"] = "gold"
    needs_judge: bool = False
    source_dataset_id: str | None = None
    source_dataset_row_id: str | None = None
    source_dataset_split: str | None = None
    source_dataset_subset: str | None = None
    source_dataset_requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    source_role: str | None = None
    module: str | None = None
    dynamic_difficulty: str | None = None
    pass_count: int | None = None
    rollout_count: int | None = None
    pass_rate: float | None = None
    replay_use_count: int | None = None

MathQuestionPayload = QuestionPayload


class SearchRequestPayload(BaseModel):  # 教师给搜索专家/检索工具的检索要求
    search_sources: list[Literal["huggingface", "web", "github"]]
    search_query: str
    goal: str = ""
    retrieval_mode: Literal["full_dataset"] = "full_dataset"
    dataset_role: Literal["train_dataset"] = "train_dataset"
    sampling_owner: Literal["filter"] = "filter"
    target_labels: list[str] = Field(default_factory=lambda: ["unknown"])#分类标签由配置注入
    sampling_plan: dict[str, Any] = Field(default_factory=dict)


class DatasetRef(BaseModel):  # 搜索到的数据集地址及其分片信息
    dataset_id: str
    source: Literal["huggingface", "web", "github"]
    subset: str | None = None
    split: str | None = None
    requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_raw_rows: list[dict[str, Any]] = Field(default_factory=list)
    cleaner_cache_ref: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    local_cache_path: str | None = None
    score_hint: float | None = None
    shard_id: int | None = None
    shard_size: int | None = None
    shard_start: int | None = None
    shard_end: int | None = None
    merge_group: str | None = None


class SearchResultPayload(BaseModel):  # 检索结果只传数据集引用，不传完整数据
    datasets: list[DatasetRef]
    search_summary: str


class DatasetSchemaRequestPayload(BaseModel):
    dataset_ref: DatasetRef
    inspect_result: dict[str, Any] = Field(default_factory=dict)


class DatasetSchemaResultPayload(BaseModel):
    dataset_ref: DatasetRef
    schema: dict[str, Any] = Field(default_factory=dict)
    inspect_result: dict[str, Any] = Field(default_factory=dict)


class DatasetReviewVerdict(BaseModel):
    dataset_id: str
    verdict: Literal["accept", "reject"]
    reason: str
    sample_count: int = 3
    suitability_score: float = 0.0


class DatasetReviewResultPayload(BaseModel):
    accepted_refs: list[DatasetRef]
    rejected_ids: list[str]
    verdicts: list[DatasetReviewVerdict] = Field(default_factory=list)
    review_summary: str = ""


class MaterializedDatasetPayload(BaseModel):  # 题目列表落盘后通过 questions_ref 传递
    dataset_refs: list[DatasetRef]
    questions: list[QuestionPayload] = Field(default_factory=list)
    questions_ref: DataArtifactRef | None = None
    materialization_summary: str


class FilterStrategyPayload(BaseModel):  # 过滤器的机制参数，具体配比由教师/参数大师给出
    target_questions_per_round: int
    deduplicate: bool = True
    drop_mastered: bool = True
    sampling_method: Literal["random", "stratified", "difficulty_balanced"] = "stratified"
    target_labels: list[str] = Field(default_factory=lambda: ["unknown"])


class FilterResultPayload(BaseModel):
    questions: list[QuestionPayload] = Field(default_factory=list)
    questions_ref: DataArtifactRef | None = None


class RolloutRequestPayload(BaseModel):  # cascade rollout 请求，题目列表通过引用传递
    questions: list[QuestionPayload] = Field(default_factory=list)
    questions_ref: DataArtifactRef | None = None
    rollout_times: int = 1
    model_path: str
    rollout_config_hash: str = ""
    rollout_judge_version: str = ""


class RolloutTaskPayload(BaseModel):
    questions: list[QuestionPayload] = Field(default_factory=list)
    questions_ref: DataArtifactRef | None = None
    rollout_idx: int
    model_path: str
    rollout_config_hash: str = ""
    rollout_judge_version: str = ""


class SingleRolloutAnswer(BaseModel):  # 单次 rollout 的答案与判定结果
    question_id: str
    question_text: str
    gold_answer: str
    rollout_gold_answer: str = ""
    train_output: str = ""
    target_style: Literal["answer", "cot"] = "answer"
    evaluation_method: Literal["gold", "llm_judge", "code_execution"] = "gold"
    needs_judge: bool = False
    source_dataset_id: str | None = None
    source_dataset_row_id: str | None = None
    source_dataset_split: str | None = None
    source_dataset_subset: str | None = None
    source_dataset_requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    source_role: str | None = None
    module: str | None = None
    dynamic_difficulty: str | None = None
    pass_count: int | None = None
    rollout_count: int | None = None
    pass_rate: float | None = None
    rollout_stage: str | None = None
    rollout_model_key: str | None = None
    rollout_config_hash: str | None = None
    rollout_judge_version: str | None = None
    replay_use_count: int | None = None
    correct: bool


class RolloutTaskResultPayload(BaseModel):
    rollout_idx: int
    answers: list[SingleRolloutAnswer] = Field(default_factory=list)
    answers_ref: DataArtifactRef | None = None


class QuestionScore(BaseModel):
    question_id: str
    question_text: str
    gold_answer: str
    rollout_gold_answer: str = ""
    train_output: str = ""
    target_style: Literal["answer", "cot"] = "answer"
    evaluation_method: Literal["gold", "llm_judge", "code_execution"] = "gold"
    needs_judge: bool = False
    source_dataset_id: str | None = None
    source_dataset_row_id: str | None = None
    source_dataset_split: str | None = None
    source_dataset_subset: str | None = None
    source_dataset_requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    dataset_window_id: str | None = None
    dataset_window_offset: int | None = None
    dataset_window_limit: int | None = None
    source_role: str | None = None
    module: str | None = None
    dynamic_difficulty: str | None = None
    pass_count: int | None = None
    rollout_count: int | None = None
    pass_rate: float | None = None
    rollout_stage: str | None = None
    rollout_model_key: str | None = None
    rollout_config_hash: str | None = None
    rollout_judge_version: str | None = None
    replay_use_count: int | None = None
    correct_flags: Annotated[list[bool], Field(description="每次 rollout 是否答对")]


class RolloutResultPayload(BaseModel):
    scored_questions: list[QuestionScore] = Field(default_factory=list)
    scored_questions_ref: DataArtifactRef | None = None





class ClassifiedQuestion(BaseModel):
    question_id: str
    question_text: str
    gold_answer: str
    rollout_gold_answer: str = ""
    train_output: str = ""
    target_style: Literal["answer", "cot"] = "answer"
    evaluation_method: Literal["gold", "llm_judge", "code_execution"] = "gold"
    needs_judge: bool = False
    source_dataset_id: str | None = None
    source_dataset_row_id: str | None = None
    source_dataset_split: str | None = None
    source_dataset_subset: str | None = None
    source_dataset_requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    source_role: str | None = None
    category: str
    module: str | None = None
    dynamic_difficulty: str | None = None
    pass_count: int | None = None
    rollout_count: int | None = None
    pass_rate: float | None = None
    replay_use_count: int | None = None
    confidence: float


class ClassificationResultPayload(BaseModel):
    questions: list[ClassifiedQuestion] = Field(default_factory=list)
    questions_ref: DataArtifactRef | None = None
    dropped_unknown_count: int = 0


class DatasetBundlePayload(BaseModel):  # 训练/评测文件均以路径传递
    train_path: str
    lf_val_path: str | None = None
    cotest_path: str
    test_path: str
    probe_path: str | None = None
    dataset_info_path: str
    train_dataset_name: str
    lf_val_dataset_name: str = ""
    class_distribution: dict[str, int]
    dataset_dir: str


class TrainRequestPayload(BaseModel):#trainer 节点，训练请求的格式，包含了训练所需要的所有信息，用哪个模型、什么数据、什么配置来训练
    model_name_or_path: str
    dataset_dir: str
    train_dataset_name: str
    config_template_path: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    action_metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticRequestPayload(BaseModel):
    reason: str
    probe_set_path: str
    champion_model_path: str


class TrainResultPayload(BaseModel):
    candidate_model_path: str
    train_log_path: str
    success: bool = False
    error_message: str = ""
    trainer_log_jsonl_path: str = ""
    training_loss_jsonl_path: str = ""
    all_results_path: str = ""
    trainer_state_path: str = ""
    train_results_path: str = ""


class DiagnosticResultPayload(BaseModel):
    candidate_model_path: str
    diagnostic_path: str
    worst_probe_path: str
    worst_probe_count: int
    target_bucket: str = ""


class SearchDAGNodePayload(BaseModel):
    node_id: str
    parent_node_ids: list[str]
    accuracy: float | None = None
    forgetting_score: float | None = None
    stability_score: float | None = None
    visit_count: int = 0
    value_estimate: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class SearchDAGEdgePayload(BaseModel):
    from_node_id: str
    to_node_id: str
    action_type: Literal["add_dataset", "reuse_buffer_data", "change_hyperparam", "mcts_action", "keep_branch"]
    action_summary: str
    round_id: int | None = None
    decision: Literal["promote", "provisional_promote", "keep_branch", "prune", "rollback"] | None = None
    reward: float | None = None
    action_metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayBufferEntryPayload(BaseModel):
    entry_id: str
    dataset_signature: str
    source_round: int
    success_score: float
    uniqueness_score: float = 1.0
    difficulty_score: float = 0.5
    coverage_score: float = 0.5
    question_id: str = ""
    question_text: str = ""
    gold_answer: str = ""
    rollout_gold_answer: str = ""
    train_output: str = ""
    target_style: Literal["answer", "cot"] = "answer"
    evaluation_method: Literal["gold", "llm_judge", "code_execution"] = "gold"
    needs_judge: bool = False
    source_dataset_id: str | None = None
    source_dataset_row_id: str | None = None
    source_dataset_split: str | None = None
    source_dataset_subset: str | None = None
    source_dataset_requested_split: str | None = None
    source_dataset_split_names: list[str] = Field(default_factory=list)
    source_dataset_columns: list[str] = Field(default_factory=list)
    source_dataset_first_row: dict[str, Any] = Field(default_factory=dict)
    source_dataset_schema: dict[str, Any] = Field(default_factory=dict)
    dynamic_difficulty: str = "unknown"
    origin_round_id: int | None = None
    pass_count: int | None = None
    rollout_count: int | None = None
    pass_rate: float | None = None
    bucket: str = "unknown"
    module: str = "unknown"
    used_in_rounds: list[int] = Field(default_factory=list)


class InspectionResultPayload(BaseModel):
    decision: Literal["promote", "provisional_promote", "keep_branch", "prune", "rollback", "diagnostic"]
    reason: str
    confidence: float
    should_store_to_replay_buffer: bool
    should_update_checkpoint: bool
    should_update_search_dag: bool


class EvalResultPayload(BaseModel):
    old_error_count: int
    old_error_rate: float
    new_skill_acc_before: float
    new_skill_acc_after: float
    new_skill_correct_before: int = 0
    new_skill_correct_after: int = 0
    new_skill_eval_count: int = 0
    cotest_acc_before: float = 0.0
    cotest_acc_after: float = 0.0
    old_ability_acc_before: float = 0.0
    old_ability_acc_after: float = 0.0
    old_ability_correct_before: int = 0
    old_ability_correct_after: int = 0
    old_ability_eval_count: int = 0
    new_ability_acc_before: float = 0.0
    new_ability_acc_after: float = 0.0
    new_ability_correct_before: int = 0
    new_ability_correct_after: int = 0
    new_ability_eval_count: int = 0
    forgetting_delta: float = 0.0
    probe_acc_after: float
    probe_acc_frozen: float | None = None
    probe_acc_champion: float | None = None
    external_probe_acc: float | None = None
    external_probe_acc_champion: float | None = None
    probe_easy_acc: float | None = None
    probe_medium_acc: float | None = None
    probe_hard_acc: float | None = None
    champion_probe_error_rate: float | None = None
    base_frozen_error_rate: float | None = None
    pass_old_skill_gate: bool
    pass_new_skill_gate: bool
    pass_probe_gate: bool
    pass_frozen_gate: bool = True
    should_stop: bool
    should_promote_candidate: bool
    training_summary: dict[str, Any] = Field(default_factory=dict)


class FormatErrorPayload(BaseModel):
    bad_message_type: str
    missing_fields: list[str]
    reason: str


# =============================================================================
# Unified Message Type
# =============================================================================


PayloadT = Union[
    GoalRequestPayload,
    SearchRequestPayload,
    SearchResultPayload,
    DatasetSchemaRequestPayload,
    DatasetSchemaResultPayload,
    DatasetReviewResultPayload,
    MaterializedDatasetPayload,
    FilterStrategyPayload,
    RolloutRequestPayload,
    RolloutTaskPayload,
    RolloutTaskResultPayload,
    RolloutResultPayload,
    FilterResultPayload,
    ClassificationResultPayload,
    DatasetBundlePayload,
    DiagnosticRequestPayload,
    TrainRequestPayload,
    TrainResultPayload,
    DiagnosticResultPayload,
    SearchDAGNodePayload,
    SearchDAGEdgePayload,
    ReplayBufferEntryPayload,
    EvalResultPayload,
    InspectionResultPayload,
    FormatErrorPayload,
]


class RoutedMessage(BaseModel):
    header: MessageHeader
    payload: PayloadT


def render_message_for_ui(message: RoutedMessage) -> str:
    return (
        f"发送方：{message.header.sender.value}\n"
        f"接收方：{message.header.receiver.value}\n"
        f"发送内容：{message.payload.model_dump()}\n"
        f"状态：{message.header.route_status.value}"
    )
