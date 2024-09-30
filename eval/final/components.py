# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from typing import List, NamedTuple
from kfp.dsl import component, Dataset, Input, Output, Artifact, Model, importer
from utils.consts import PYTHON_IMAGE, EVAL_IMAGE


# Much of this component is borrowed from instructlab/instructlab models/evaluate.py
# https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py
# TODO: package vllm, etc within base image
@component(
    base_image=EVAL_IMAGE, packages_to_install=["vllm", "tenacity", "lm-eval[api]"]
)
def run_mmlu_branch_mt_bench_branch_op(
    mmlu_branch_output: Output[Artifact],
    mt_bench_branch_output: Output[Artifact],
    candidate_model: str,
    base_model: str,
    tasks: Input[Dataset],
    taxonomy: Input[Dataset],
    base_branch: str,
    candidate_branch: str,
    model_dtype: str,
    few_shots: int,
    batch_size: int,
    device: str,
    merge_system_user_message: bool,
):
    import json
    import os
    import torch

    from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLU_TASKS
    from instructlab.eval.mt_bench import MTBenchBranchEvaluator

    # TODO: package these embedded functions within the base image

    def launch_local_vllm(
        model_path: str, gpu_count: int, retries: int = 60, delay: int = 5
    ):
        import subprocess
        import sys
        import time
        import requests

        # TODO: assign global var
        vllm_server = "http://localhost:8000/v1"

        if gpu_count > 0:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--tensor-parallel-size",
                str(gpu_count),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
            ]

        subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {vllm_server}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{vllm_server}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {vllm_server}.")
                    return
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {vllm_server} after {retries} retries."
        )

    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
    # Also, the base image does not include `pkill` cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
    def stop_local_vllm():
        import psutil

        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            cmdline = process.info.get("cmdline")
            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                print(
                    f"Found vLLM server process with PID: {process.info['pid']}, terminating..."
                )
                try:
                    process.terminate()  # Try graceful termination
                    process.wait(timeout=5)  # Wait a bit for it to terminate
                    if process.is_running():
                        print(
                            f"Forcefully killing vLLM server process with PID: {process.info['pid']}"
                        )
                        process.kill()  # Force kill if it's still running
                    print(
                        f"Successfully stopped vLLM server with PID: {process.info['pid']}"
                    )
                except psutil.NoSuchProcess:
                    print(f"Process with PID {process.info['pid']} no longer exists.")
                except psutil.AccessDenied:
                    print(
                        f"Access denied when trying to terminate process with PID {process.info['pid']}."
                    )
                except Exception as e:
                    print(
                        f"Failed to terminate process with PID {process.info['pid']}. Error: {e}"
                    )

    def branch_eval_summary_to_json(
        improvements: list[tuple[str, float]],
        regressions: list[tuple[str, float]],
        no_changes: list[str],
        new=None,
    ) -> str:
        """takes in results lists from mt_bench_branch and mmlu_branch benchmark evaluation
        returns a JSON object with diff between the branches
        """
        if len(improvements) > 0:
            improvements.sort(key=sort_score, reverse=True)

        if len(regressions) > 0:
            regressions.sort(key=sort_score)

        summary = {
            "improvements": [
                {"task": task, "delta": delta} for task, delta in improvements
            ],
            "regressions": [
                {"task": task, "delta": delta} for task, delta in regressions
            ],
            "no_changes": [{"task": task} for task in no_changes],
        }

        if new is not None and len(new) > 0:
            summary["new"] = [{"qna": qna} for qna in new]

        return json.dumps(summary, indent=4)

    # Copied from instructlab/instructlab
    # https://github.com/instructlab/instructlab/blob/f7b0d1587741bfebe62f9b7bfd1b82a94b59f809/src/instructlab/model/evaluate.py#L203
    def qa_pairs_to_qna_to_avg_scores(qa_pairs: list[dict]) -> dict[str, float]:
        """takes in a list of qa_pair dicts
        returns a dict of average scores per qna file
        """
        qna_to_scores: dict[str, list[float]] = {}
        for qa_pair in qa_pairs:
            qna_file = qa_pair["qna_file"]
            score = qa_pair["score"]
            scores = qna_to_scores.get(qna_file)
            if scores is None:
                qna_to_scores[qna_file] = [score]
            else:
                scores.append(score)
        qna_to_avg_scores = {}
        for qna, scores in qna_to_scores.items():
            qna_to_avg_scores[qna] = sum(scores) / len(scores)
        return qna_to_avg_scores

    def find_matching_directories(base_directory: str, pattern: str):
        import re
        import os

        matching_dirs = []
        regex = re.compile(pattern)

        for root, dirs, files in os.walk(base_directory):
            for directory in dirs:
                if regex.search(directory):
                    matching_dirs.append(os.path.join(root, directory))

        return matching_dirs

    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")

    # TODO: assign global var
    vllm_server = "http://localhost:8000/v1"

    # MT_BENCH_BRANCH

    # model_name is same as model_path with ilab setup
    candidate_model_name = candidate_model
    base_model_name = base_model

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    output_dir = ("/tmp/eval_output",)

    # TODO: candidate_branch must be in same repo, not a fork, or, can compare main branch against candidate, base models
    # ??
    base_branch = base_branch or "main"
    candidate_branch = candidate_branch or "main"

    # TODO: Fix logic for branch
    # ilab is set up to test branches from a single repo,
    # not a fork of a repo with a PR# comparing across forks.

    mt_bench_evaluators = [
        MTBenchBranchEvaluator(
            candidate_model_name,
            judge_model_name,
            taxonomy.path,
            candidate_branch,
            output_dir,
            merge_system_user_message=merge_system_user_message,
        ),
        MTBenchBranchEvaluator(
            base_model_name,
            judge_model_name,
            taxonomy.path,
            base_branch,
            output_dir,
            merge_system_user_message=merge_system_user_message,
        ),
    ]

    branches = [candidate_branch, base_branch]
    m_paths = [candidate_model, base_model]
    m_names = [candidate_model_name, base_model_name]
    qa_pairs_and_errors = []
    for i, evaluator in enumerate(mt_bench_evaluators):
        branch = branches[i]
        m_path = m_paths[i]
        m_name = m_names[i]

        print(
            f"Generating questions and reference answers from qna files for branch {branch}..."
        )
        launch_local_vllm(candidate_model, gpu_count)

        evaluator.gen_answers(vllm_server)

        stop_local_vllm()

        print(f"Evaluating answers for branch {branch}...")
        judgement = evaluator.judge_answers(judge_endpoint, api_key=judge_api_key)

        if len(judgement) == 3:
            qa_pairs = judgement[1]
            error_rate = judgement[2]
        else:
            qa_pairs = judgement[0]
            error_rate = judgement[1]

        qa_pairs_and_errors.append((qa_pairs, error_rate))

    candidate_qa_pairs, candidate_error_rate = qa_pairs_and_errors[0]
    base_qa_pairs, base_error_rate = qa_pairs_and_errors[1]

    candidate_qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(qa_pairs)
    base_qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(base_qa_pairs)

    improvements, regressions, no_changes, new_qnas = [], [], [], []
    for qna, avg_score in candidate_qna_to_avg_scores.items():
        base_avg_score = base_qna_to_avg_scores.get(qna)
        if base_avg_score is not None:
            if avg_score > base_avg_score:
                improvements.append((qna, round(avg_score - base_avg_score, 2)))
            elif avg_score == base_avg_score:
                no_changes.append(qna)
            else:
                regressions.append((qna, round(avg_score - base_avg_score, 2)))
        else:
            new_qnas.append((qna))

    error_rate = (candidate_error_rate + base_error_rate) / 2
    if error_rate > 0:
        error_rate = round(error_rate, 2)

    summary = branch_eval_summary_to_json(
        improvements, regressions, no_changes, new_qnas
    )

    mt_bench_data = {
        "report_title": "SKILLS EVALUATION REPORT",
        "model": candidate_model_path,
        "judge_model": judge_model_name,
        "overall_score": overall_score,
        "turn_scores": turn_scores,
        "qa_scores": qa_pairs,
        "error_rate": error_rate,
        "summary": summary,
    }

    with open(mt_bench_branch_output.path, "w") as f:
        json.dump(mt_bench_branch_data, f, indent=4)

    # MMLU_BRANCH

    # These are specific to ilab/eval
    pattern = r"node_datasets_"
    mmlu_tasks = ["mmlu_pr"]

    node_dataset_dirs = find_matching_directories(tasks.path, pattern)
    if node_dataset_dirs:
        tasks_dir = node_dataset_dirs[0]

        mmlu_branch_evaluators = [
            MMLUBranchEvaluator(
                candidate_model,
                tasks_dir,
                mmlu_tasks,
                few_shots=few_shots,
                batch_size=batch_size,
            ),
            MMLUBranchEvaluator(
                base_model,
                tasks_dir,
                mmlu_tasks,
                few_shots=few_shots,
                batch_size=batch_size,
            ),
        ]
        m_paths = [candidate_model, base_model]
        overall_scores = []
        individual_scores_list = []
        for i, evaluator in enumerate(mmlu_branch_evaluators):
            m_path = m_paths[i]
            launch_local_vllm(candidate_model, gpu_count)
            overall_score, individual_scores = evaluator.run(vllm_server)
            overall_scores.append(overall_score)
            individual_scores_list.append(individual_scores)
            stop_local_vllm()

        candidate_overall_score = overall_scores[0]
        base_overall_score = overall_scores[1]
        candidate_individual_scores = individual_scores_list[0]
        base_individual_scores = individual_scores_list[1]
        delta = round(candidate_overall_score - base_overall_score, 2)
        if delta >= 0:
            average = f"+{delta}"
        else:
            average = delta

        improvements, regressions, no_changes = [], [], []
        for task, score in candidate_individual_scores.items():
            base_score = base_individual_scores[task]
            s = score["score"]
            b_s = base_score["score"]
            d = round(s - b_s, 2)
            if s > b_s:
                improvements.append((task, d))
            elif b_s > s:
                regressions.append((task, d))
            else:
                no_changes.append(task)

        summary = branch_eval_summary_to_json(
            improvements, regressions, no_changes, new_tasks
        )
        mmlu_branch_data = {
            "report_title": "KNOWLEDGE EVALUATION REPORT",
            "candidate_model": candidate_model,
            "base_model": base_model,
            "average_score": average,
            "number_of_tasks": len(candidate_individual_scores),
            "individual_scores": [
                {task: round(score["score"], 2)}
                for task, score in candidate_individual_scores.items()
            ],
            "summary": summary,
        }

        with open(mmlu_branch_output.path, "w") as f:
            json.dump(mmlu_branch_data, f, indent=4)
    else:
        print("No MMLU tasks directories found, skipping MMLU_branch evaluation.")
