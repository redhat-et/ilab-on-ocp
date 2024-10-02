# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from typing import List, NamedTuple, Optional
from kfp.dsl import component, Input, Output, Artifact, Model, importer
from utils.consts import PYTHON_IMAGE

EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval"


@component(base_image=EVAL_IMAGE)
def run_mmlu_op(
    mmlu_output: Output[Artifact],
    models_path_prefix: str,
    mmlu_tasks_list: str,
    model_dtype: str,
    few_shots: int,
    batch_size: int,
    device: str = None,
    models_list: List[str] = None,
    models_folder: Optional[str] = None,
) -> NamedTuple("outputs", best_model=str, best_score=float):
    import json
    import os
    import torch
    from instructlab.eval.mmlu import MMLUEvaluator, MMLU_TASKS

    mmlu_tasks = mmlu_tasks_list.split(",") if mmlu_tasks_list else MMLU_TASKS

    if models_list is None and models_folder:
        models_list = os.listdir(models_folder)

    # Device setup and debug
    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )

    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")

    effective_device = (
        device if device is not None else ("cuda" if gpu_available else "cpu")
    )
    print(f"Running on device: {effective_device}")

    scores = {}
    all_mmlu_data = []

    for model_name in models_list:
        model_path = f"{models_path_prefix}/{model_name}"
        # Debug
        print(f"Model {model_name} is stored at: {model_path}")

        # Evaluation
        evaluator = MMLUEvaluator(
            model_path=model_path,
            tasks=mmlu_tasks,
            model_dtype=model_dtype,
            few_shots=few_shots,
            batch_size=batch_size,
            device=effective_device,
        )

        mmlu_score, individual_scores = evaluator.run()
        average_score = round(mmlu_score, 2)
        print(
            f"Model {model_name} is stored at: {model_path} with AVERAGE_SCORE: {average_score}"
        )

        mmlu_data = {
            "report_title": "KNOWLEDGE EVALUATION REPORT",
            "model": model_name,
            "average_score": average_score,
            "number_of_tasks": len(individual_scores),
            "individual_scores": [
                {task: round(score["score"], 2)}
                for task, score in individual_scores.items()
            ],
        }

        all_mmlu_data.append(mmlu_data)
        scores[model_path] = average_score

    with open(mmlu_output.path, "w") as f:
        json.dump(all_mmlu_data, f, indent=4)
    outputs = NamedTuple("outputs", best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    return outputs(best_model=best_model, best_score=best_score)


@component(base_image=PYTHON_IMAGE)
def load_mmlu_results_op(mmlu_output: Input[Artifact]) -> list:
    import json

    mmlu_score_list = []
    with open(mmlu_output.path, "r") as f:
        mmlu_score_list = json.load(f)

    print("MMLU Evaluation Data:")
    for mmlu_score in mmlu_score_list:
        print(json.dumps(mmlu_score, indent=4))

    return mmlu_score_list
