# EvoChampion 技术文档索引

这里是根 `README.md` 的渐进式展开层。根 README 负责快速上手和整体地图；本目录按模块解释实现细节、消息边界、关键状态、配置、产物和常见排错入口。

建议阅读顺序：

1. [00-overview.md](00-overview.md)：一次完整进化循环的全景、核心术语、目录结构和阅读路线。
2. [01-graph-messages-state.md](01-graph-messages-state.md)：LangGraph 图、router、消息协议、artifact ref、EvoState 和契约校验。
3. [02-bootstrap-prompt-routing.md](02-bootstrap-prompt-routing.md)：bootstrap、probe 初始化、checkpoint 恢复、prompt_designer 和首轮路由。
4. [03-data-admission.md](03-data-admission.md)：searcher、HF search、dataset_reviewer、schema agent、screening_entry、dataset_state。
5. [04-rollout-filter-difficulty.md](04-rollout-filter-difficulty.md)：cascade rollout、题目难度、rollout_aggregator、filter pre/post、quota 和补窗。
6. [05-policy-control.md](05-policy-control.md)：teacher、parameter_master、DAG/MCTS、LLM leaf decision、sampling plan、replay ratio。
7. [06-data-builder-classifier.md](06-data-builder-classifier.md)：classifier、split 构建、holdout registry、probe/test buffer、LLaMA-Factory 数据包。
8. [07-training-evaluation-inspection.md](07-training-evaluation-inspection.md)：trainer、LLaMA-Factory、evaluator、MathBench/OpenCompass、strategy_inspector、晋升/回滚。
9. [08-runtime-configuration.md](08-runtime-configuration.md)：环境变量、运行脚本、vLLM/Ray、HF cache、DeepSeek cleaner、服务器实验配置。
10. [09-artifacts-debugging.md](09-artifacts-debugging.md)：session 产物目录、关键日志、追踪文件、常见问题定位路径。

## 读者入口

- 想跑实验：先读根 `README.md`，再读 [08-runtime-configuration.md](08-runtime-configuration.md)。
- 想理解为什么选这批数据：读 [05-policy-control.md](05-policy-control.md) 和 [04-rollout-filter-difficulty.md](04-rollout-filter-difficulty.md)。
- 想排查数据异常：读 [03-data-admission.md](03-data-admission.md)、[06-data-builder-classifier.md](06-data-builder-classifier.md)、[09-artifacts-debugging.md](09-artifacts-debugging.md)。
- 想排查训练/评估异常：读 [07-training-evaluation-inspection.md](07-training-evaluation-inspection.md) 和 [08-runtime-configuration.md](08-runtime-configuration.md)。
