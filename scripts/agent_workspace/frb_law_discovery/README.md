# FRB 符号规律探索

本目录用于自动探索 FRB （Fast Radio Burst）偏振观测数据中的隐藏规律。

具体而言，Agent 应利用本目录中提供的数据和工具，尝试将原始的 FRB 观测数据处理成候选特征表，再结合符号回归和常规数据分析寻找稳定的、非平凡的、具有预测能力的符号规律。最终产出一系列数据中隐藏的有效候选模式，交给人类专家判断其物理意义和研究价值。


## 任务

你需要围绕 `data/` 提供的原始数据反复探索：
1. 阅读 `./data/README.md`，理解问题背景、数据来源。
2. 运行 `feature_extraction.py`，从 HDF5 和选择窗口中计算新的字段，生成 `saved/<exp_name>/output.csv` 和 `description.json`。
3. 运行 `run_sr_agent.py`，用 SRAgent 搜索符号公式。
4. 尝试修改 `feature_extraction.py` 的文件内容以探索新的特征；或尝试修改 `python run_sr_agent.py` 的调用参数以探索新的搜索策略。
5. 将有价值的候选规律写入 `patterns.md`。在探索时不要只追求高拟合分数，还要注意样本量、泄漏风险、公式复杂度、跨 FRB/望远镜稳定性和物理可解释性等等。


## 文件说明

- `README.md`: 本文件，说明目录结构、任务和工作要求。
- `data/`：原始 HDF5 数据和人工选择的时频窗口，阅读 `data/README.md` 以了解详细内容。注意：不得修改该目录中内容。
- `feature_extraction.py`：数据预处理脚本。运行以生成可供符号回归使用的数据字段，可以修改其中标记的部分以探索新的特征。
- `saved/`：运行 `feature_extraction.py` 产生的结果目录，包括 `output.csv`、`description.json`、`info.log`、`args.json` 等文件。每次运行时请使用新的 `--exp_name` 以进行区分。
- `run_sr_agent.py`: 本项目所开发的符号回归智能体（SRAgent）的运行脚本，运行时指定 feature_extraction.py 生成的 CSV 文件、目标字段、特征字段和问题描述等参数，SRAgent 将尝试搜索符号公式。
- `logs/`：运行 `run_sr_agent.py` 时产生的日志目录，记录 SRAgent 的搜索过程和结果。
- `draft_note.md`：Agent 自用的工作笔记，可以在其中写入任何对探索有益的内容。
- `patterns.md`：给人类看的结果摘要，只记录已经发现的候选规律。


## 附：SRAgent 说明

SRAgent (Symbolic Regression Agent) 是本项目开发的符号回归智能体。它通过 LLM（大语言模型）和工具调用的组合，尝试发现能够从输入特征预测目标字段的符号公式。


### 搜索循环机制

SRAgent 的搜索循环由四个参数控制：`R/C/L/K`，分别对应重启次数、独立分支数、每个分支的迭代深度和每轮采样的候选数：
- `R = max_restart_loop`：重启次数。后续 restart 会把历史最佳公式注入 prompt。
- `C = global_width`：每次 restart 下的独立对话分支数。
- `L = max_refinement_depth`：每个分支内的迭代深度。
- `K = local_sample_size`：每轮 prompt 向 LLM 采样的候选回复数。

搜索循环伪代码如下：

```python
R = max_restart_loop
C = global_width
L = max_refinement_depth
K = local_sample_size

topk = []
for r in range(R):          # Restart：用历史最好结果重新开始
    initial_prompt = build_initial_prompt(problem, topk[:restart_top_k])
    for c in range(C):          # Conversation：独立对话分支
        buffer = copy(initial_prompt)
        for l in range(L):  # Refinement：同一分支内的迭代轮次
            prompt = build_prompt(buffer)
            responses = request_llm(prompt, n=K)  # K 个局部采样
            results = run_tool_calls(responses)
            buffer = update_buffer(buffer, responses, results)
            topk = update_topk(topk, results)
            pareto_front = get_pareto_front(topk)
```


### 工具机制

SRAgent 通过 tool 调用完成数据分析、公式评估和结构搜索。通过以下命令查看可用工具：

```bash
sr-agent-tool list
```

通过以下命令手动调用某个工具分析数据（其中 context.npz 是数据上下文文件，调用 run_sr_agent.py 时会自动生成并保存至 logs/run_sr_agent/<exp_name>/context.npz 中）：

```bash
sr-agent-tool call <tool> --context <context.npz> --params '<json>'
```

常用工具大致如下：
- 数据分析类
    - `statistics_analysis`：计算变量或表达式的基本统计量。
    - `code_executor`：在受限沙盒中运行 Python 代码，用于探索数据结构、变换、残差等。
    - `predict_property`：预测单调性、凸性、周期性、乘法可分性等数学性质。
- 公式评估 / 拟合 / 搜索类
    - `evaluate_formula`：评估一个 nd2py 公式的拟合质量。
    - `evaluate_code`：评估一个 Python 定义的候选模型；适合 nd2py 公式表达不了的模式，例如依赖非数值字段、分段规则或字典映射的预测程序。
    - `polynomial_fit`：拟合多项式结构。
    - `call_sindy`：调用 SINDy 做稀疏回归。
    - `call_pysr`：调用 PySR 做遗传编程式符号回归。
- 通用类
    - `read_skill/create_skill/edit_skill`：读取或维护可复用搜索策略。
- 其它类
    - `ask_human`、`workspace_shell`、`workspace_code_executor`：交互或工作区工具，对本项目帮助不大，默认被排除。

根据你对工具功能的理解和 SRAgent 使用此工具的具体表现，可以在运行 `run_sr_agent.py` 时通过 `--tools` 参数指定工具子集，以此改善（也可能降低）搜索性能。


### 最佳公式更新机制

公式评估 / 拟合 / 搜索类工具在运行时可产生如下字段：

```python
{
    "formula": "...",
    "metrics": {"mse": ..., "complexity": ..., ...},
    "is_candidate": True,
}
```

其中，is_candidate=True 表示该公式作为用自变量预测因变量的有效公式，可以被搜索循环纳入 best formula / pareto front。
- best formula: 所有 is_candidate=True 的公式中，具有最佳 mse score 的公式
- pareto front: 所有 is_candidate=True 的公式中，具有最佳 mse-complexity-balance 的公式集合

最佳公式更新机制的伪代码如下：

```python
for each tool_result:
    if tool_result["is_candidate"]:
        record = {
            "formula": tool_result["formula"],
            **tool_result["metrics"],
            "node_id": current_search_node,
        }
        topk.push(priority=record["mse"], record=record)

best_formula = min(topk, key=mse)

pareto_front = []
best_complexity_so_far = inf
for record in sorted(topk, key=mse):
    if record["complexity"] < best_complexity_so_far:
        pareto_front.append(record)
        best_complexity_so_far = record["complexity"]
```

注意：受代码架构所限，mse score 等指标是在全样本上计算的，没有训练/测试拆分。因此高分公式需警惕过拟合和样本量过小等问题。


### evaluate_formula 与 evaluate_code

SRAgent 通常希望发现一种可以写成数学方程的模式，形如 `a * x + b`、`sin(x1) + x2**2`、`exp(-a * x)` 等。这类公式由 `nd2py` package 描述：它是一个类似 `sympy` 的轻量级符号处理库，支持常见数学函数、符号表达式求值，并提供在数据上通过 BFGS 自动拟合公式中未知参数的能力。

然而，`nd2py` 提供的数学方程表达能力并不完整。对于分段函数 / 查表规则 / 条件分支等更复杂的模式、涉及非数值字段的规则、或需要通过 BFGS 以外的特定算法估计公式中参数的情况，强行写成单个 `nd2py` 公式通常不自然，甚至无法表达。为此，SRAgent 提供了 `evaluate_code`：Agent 可以编写一个 `def model_func(data)` 和一个 `def predict_func(data, model)`，前者用自定义方式从数据中拟合或构造模型，后者用该模型对给定数据生成预测。这个拟合过程不局限于 `nd2py` 使用的 BFGS 参数优化，也可以是手写规则、分组统计、简单聚类、字典映射或其它 Python 逻辑。

两类候选都会返回 `formula`、`metrics` 和 `is_candidate`，从而参与 best formula 和 pareto front 的比较。区别在于复杂度定义：
- `evaluate_formula`：复杂度通常是数学表达式中的符号/节点数量，较容易解释。
- `evaluate_code`：Python 程序的复杂度不易精确定义。当前实现将复杂度近似为 `model_func` 和 `predict_func` 代码字符总数。虽然这会使得其复杂度远高于普通数学公式，考虑到 Python 程序的表达能力通常更强、拟合效果通常更好，因此在 pareto front 中仍然有可能被选中。


## Hints

- 调用 SRAgent 时注意，太低的预算（例如 --R 1 --C 1 --L 2 --K 1）不可能发现有效的规律，只会白白浪费时间和资源。建议至少使用 --R 2 --C 2 --L 10 --K 2 这样的组合，也可以通过增加 --R/--C/--L/--K 进一步增加搜索预算。
- 目标规律可能难以被数学公式描述，建议通过 `python run_sr_agent.py --ban_tools evaluate_formula` 禁用 `evaluate_formula`，以迫使 SRAgent 使用表达能力更强的 `evaluate_code` 提交结果。
