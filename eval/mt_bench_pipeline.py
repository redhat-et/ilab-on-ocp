# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from typing import NamedTuple
from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model, importer
from kfp.kubernetes import use_config_map_as_env, use_secret_as_env

EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval"
TOOLBOX_IMAGE = "registry.access.redhat.com/ubi9/toolbox"
CANDIDATE_S3_URI = "s3://sallyom-eval-e58df6b0-606b-4749-96a5-a105657cb068/models/instructlab/granite-7b-lab"
K8S_NAME = "kfp-model-server"

# TODO: This is a placeholder for launching vLLM with candidate model
@component(base_image=EVAL_IMAGE)
def serve_candidate_model(
    candidate_model: Input[Model]
) -> NamedTuple('outputs', candidate_model_name=str, candidate_server_url=str):
    import os

    candidate_model_name = os.path.basename(os.path.normpath(candidate_model.path))
    print(f"(real)Candidate Model Name: {candidate_model_name}")
    # TODO: REMOVE after implement vLLM with candidate model
    # Fake model-name for now, with judge_model_name
    candidate_model_name = os.getenv("JUDGE_MODEL_NAME") 
    print(f"(fake)Candidate Model Name: {candidate_model_name}")

    # TODO: Replace this with the candidate model endpoint
    # Launch_server call with https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L301C1-L327C1
    candidate_server_url = os.getenv("JUDGE_MODEL_ENDPOINT")
    outputs = NamedTuple('outputs', candidate_model_name=str, candidate_server_url=str)
    return outputs(candidate_model_name, candidate_server_url)

@component(base_image=EVAL_IMAGE)
def run_mt_bench(
    candidate_server_url: str,
    candidate_model_name: str,
    mt_bench_output: Output[Artifact],
    device: str,
    max_workers: str,
    merge_system_user_message: bool,
):
    import json
    import torch
    import os
    #from instructlab.eval.mt_bench import MTBenchEvaluator
    from instructlab.eval import mt_bench_answers, mt_bench_judgment

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if gpu_available else "No GPU available"

    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")

    effective_device = device if device is not None else ("cuda" if gpu_available else "cpu")

    print(f"Running on device: {effective_device}")

    judge_model_name = os.getenv("JUDGE_MODEL_NAME")
    judge_endpoint = os.getenv("JUDGE_MODEL_ENDPOINT")
    print(f"Candidate Server URL: {candidate_server_url}")
    print(f"Candidate Model Name: {candidate_model_name}")

    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    # TODO: Using evaluator results in connection errors, need to determine why.
    #       For now, using mt_bench_answers.generate_answers & mt_bench_judgment.generate_judgment
    #evaluator = MTBenchEvaluator(
    #    model_name=candidate_model_name,
    #    judge_model_name=judge_model_name,
    #    max_workers=max_workers,
    #    merge_system_user_message=merge_system_user_message
    #)

    judge_api_key = os.getenv("JUDGE_API_KEY", "")

    print("Generating answers...")
    # TODO: launch vLLM with candidate-model
    # Faking it by using judge_endpoint for generate, so need to pass the api_key & candidate_server_url is actually judge_server_url for now
    # Need to add a launch_server task with https://github.com/instructlab/instructlab/blob/main/src/instructlab/model/evaluate.py#L301C1-L327C1
    # with local vLLM, won't need to pass an api_key here
    #
    # See note above about issue with evaluator
    #evaluator.gen_answers(server_url=candidate_server_url, api_key=judge_api_key)
    mt_bench_answers.generate_answers(
        model_name=candidate_model_name,
        model_api_base=candidate_server_url,
        api_key=judge_api_key,
        output_dir="/tmp/eval_output",
        max_workers=max_workers
    )

    print("Judging answers...")
    # See note above about issue with evaluator
    #overall_score, qa_pairs, turn_scores = evaluator.judge_answers(server_url=judge_endpoint, api_key=judge_api_key)
    overall_score, qa_pairs, turn_scores, error_rate = mt_bench_judgment.generate_judgment(
        model_name = candidate_model_name,
        judge_model_name=judge_model_name,
        model_api_base=judge_endpoint,
        api_key=judge_api_key,
        output_dir="/tmp/eval_output",
        max_workers=max_workers,
        merge_system_user_message=merge_system_user_message
    )

    mt_bench_data = {
        "report_title": "SKILLS EVALUATION REPORT",
        "model": candidate_model_name,
        "judge_model": judge_model_name,
        "overall_score": overall_score,
        "turn_scores": turn_scores,
        "qa_scores": qa_pairs,
        "error_rate": error_rate,
    }

    with open(mt_bench_output.path, 'w') as f:
        json.dump(mt_bench_data, f, indent=4)

@component(base_image=TOOLBOX_IMAGE)
def load_mt_bench_results(mt_bench_output: Input[Artifact]) -> dict:
    import json

    with open(mt_bench_output.path, 'r') as file:
        mt_bench_data = json.load(file)
    print("Loaded MT_Bench Data:")
    print(json.dumps(mt_bench_data, indent=4))

    return mt_bench_data

@pipeline(
    display_name="MT_BENCH Evaluation Pipeline",
    name="mt_bench_eval",
    description="Pipeline to run the MT_BENCH evaluation",
)

def mt_bench_pipeline(
    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - `auto`
    # with `auto`, number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36 
    max_workers: str = "auto",
    merge_system_user_message: bool = False,
    device: str = None,

):
    import_candidate_model_task = importer(
        artifact_uri=CANDIDATE_S3_URI,
        artifact_class=Model,
        # TODO: True/False?
        reimport=False,
    )

    serve_candidate_model_task = serve_candidate_model(
        candidate_model=import_candidate_model_task.output,
    )

    run_mt_bench_task = run_mt_bench(
        candidate_server_url=serve_candidate_model_task.outputs['candidate_server_url'],
        candidate_model_name=serve_candidate_model_task.outputs['candidate_model_name'],
        max_workers=max_workers,
        device=device,
        merge_system_user_message=merge_system_user_message,
    )

    load_mt_bench_results_task = load_mt_bench_results(
        mt_bench_output=run_mt_bench_task.output,
    )

    run_mt_bench_task.set_accelerator_type('nvidia.com/gpu')
    run_mt_bench_task.set_accelerator_limit(1)

    # For example on K8S object to populate see ./kfp-model-server.yaml
    use_config_map_as_env(run_mt_bench_task, K8S_NAME, dict(judge_model_endpoint="JUDGE_MODEL_ENDPOINT", judge_model_name="JUDGE_MODEL_NAME"))
    use_config_map_as_env(serve_candidate_model_task, K8S_NAME, dict(judge_model_endpoint="JUDGE_MODEL_ENDPOINT", judge_model_name="JUDGE_MODEL_NAME"))
    use_secret_as_env(run_mt_bench_task, K8S_NAME, {"judge_api_key": "JUDGE_API_KEY"})

if __name__ == "__main__":
    compiler.Compiler().compile(mt_bench_pipeline, "mt_bench_pipeline.yaml")

