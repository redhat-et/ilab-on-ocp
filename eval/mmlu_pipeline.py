# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model, importer

EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval"
TOOLBOX_IMAGE = "registry.access.redhat.com/ubi9/toolbox"
CANDIDATE_S3_URI = "s3://sallyom-eval-e58df6b0-606b-4749-96a5-a105657cb068/models/instructlab/granite-7b-lab"

@component(base_image=EVAL_IMAGE)
def run_mmlu(
    candidate_model: Input[Model],
    mmlu_output: Output[Artifact],
    mmlu_tasks_list: str,
    model_dtype: str,
    few_shots: int,
    batch_size: int,
    device: str,
):
    import json
    import os
    import torch
    from instructlab.eval.mmlu import MMLUEvaluator, MMLU_TASKS

    model_name = os.path.basename(os.path.normpath(candidate_model.path))
    # Debug
    print(f"Model {model_name} is stored at: {candidate_model.path}")

    mmlu_tasks = mmlu_tasks_list.split(',') if mmlu_tasks_list else MMLU_TASKS

    # Device setup and debug
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if gpu_available else "No GPU available"

    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")

    effective_device = device if device is not None else ("cuda" if gpu_available else "cpu")
    print(f"Running on device: {effective_device}")

    # Evaluation
    evaluator = MMLUEvaluator(
        model_path=candidate_model.path,
        tasks=mmlu_tasks,
        model_dtype=model_dtype,
        few_shots=few_shots,
        batch_size=batch_size,
        device=effective_device,
    )

    mmlu_score, individual_scores = evaluator.run()
    mmlu_data = {
        "report_title": "KNOWLEDGE EVALUATION REPORT",
        "model": model_name,
        "average_score": round(mmlu_score, 2),
        "number_of_tasks": len(individual_scores),
        "individual_scores": [{task: round(score['score'], 2)} for task, score in individual_scores.items()]
    }

    with open(mmlu_output.path, 'w') as f:
        json.dump(mmlu_data, f, indent=4)

@component(base_image=TOOLBOX_IMAGE)
def load_mmlu_results(mmlu_output: Input[Artifact]) -> dict:
    import json

    with open(mmlu_output.path, 'r') as file:
        evaluation_data = json.load(file)

    print("Loaded Evaluation Data:")
    print(json.dumps(evaluation_data, indent=4))

    return evaluation_data

@pipeline(
    display_name="MMLU Evaluation Pipeline",
    name="mmlu_eval",
    description="Pipeline to run the MMLU evaluation script",
)

def mmlu_pipeline(
    # minimal subset of MMLU_TASKS
    mmlu_tasks_list: str = "mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy",
    model_dtype: str = "bfloat16",
    few_shots: int = 5,
    batch_size: int = 8,
    device: str = None,
):
    import_candidate_model_task = importer(
        artifact_uri=CANDIDATE_S3_URI,
        artifact_class=Model,
        # TODO: True/False?
        reimport=False
    )

    run_mmlu_task = run_mmlu(
        candidate_model=import_candidate_model_task.output,
        mmlu_tasks_list=mmlu_tasks_list,
        model_dtype=model_dtype,
        few_shots=few_shots,
        batch_size=batch_size,
        device=device,
    )

    load_mmlu_results_task = load_mmlu_results(
        mmlu_output=run_mmlu_task.output,
    )

    run_mmlu_task.set_accelerator_type('nvidia.com/gpu')
    run_mmlu_task.set_accelerator_limit(1)

    return

if __name__ == "__main__":
    compiler.Compiler().compile(mmlu_pipeline, "mmlu_pipeline.yaml")
