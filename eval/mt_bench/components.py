# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error
from typing import List, NamedTuple
from kfp.dsl import component, Input, Output, Artifact, Model, importer
from utils.consts import PYTHON_IMAGE

EVAL_IMAGE = "quay.io/sallyom/instructlab-ocp:eval"

@component(base_image=EVAL_IMAGE, packages_to_install=["vllm"]))
def run_mt_bench_op(
    models_path_prefix: str,
    models_list: List[str],
    mt_bench_output: Output[Artifact],
    merge_system_user_message: bool,
    max_workers: str = "auto",
    device: str = None,
) -> NamedTuple('outputs', best_model=str, best_score=float):

    # TODO: Ensure vLLM server is utilizing GPU
    def launch_vllm_server_background(model_path: str, retries: int = 60, delay: int = 5):
        import subprocess
        import time
        import requests

        command = f"nohup python3.11 -m vllm.entrypoints.openai.api_server --model {model_path} &"
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        server_url = "http://localhost:8000/v1"
        print(f"Waiting for vLLM server to start at {server_url}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{server_url}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {server_url}.")
                    return
            except requests.ConnectionError:
                pass

            print(f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})...")
            time.sleep(delay)
        
        raise RuntimeError(f"Failed to start vLLM server at {server_url} after {retries} retries.")

    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
    # Also, the base image does not include `pkill` cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
    def stop_vllm_server_by_name():
        import psutil

        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            cmdline = process.info.get("cmdline")
            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                print(f"Found vLLM server process with PID: {process.info['pid']}, terminating...")
                try:
                    process.terminate()  # Try graceful termination
                    process.wait(timeout=5)  # Wait a bit for it to terminate
                    if process.is_running():
                        print(f"Forcefully killing vLLM server process with PID: {process.info['pid']}")
                        process.kill()  # Force kill if it's still running
                    print(f"Successfully stopped vLLM server with PID: {process.info['pid']}")
                except psutil.NoSuchProcess:
                    print(f"Process with PID {process.info['pid']} no longer exists.")
                except psutil.AccessDenied:
                    print(f"Access denied when trying to terminate process with PID {process.info['pid']}.")
                except Exception as e:
                    print(f"Failed to terminate process with PID {process.info['pid']}. Error: {e}")


    import json
    import torch
    import os

    from instructlab.eval import mt_bench_answers, mt_bench_judgment

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    candidate_server_url = "http://localhost:8000/v1"

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if gpu_available else "No GPU available"

    print(f"GPU Available: {gpu_available}, {gpu_name}")

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
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    scores = {}
    all_mt_bench_data = []

    for model_name in models_list:
        print(f"Serving candidate model: {model_name}")
        model_path = f"{models_path_prefix}/{model_name}"
        
        # Launch the vLLM server and wait until it is ready
        launch_vllm_server_background(model_path)

        # model ID is the model_path value in vLLM
        print("Generating answers...")
        mt_bench_answers.generate_answers(
            model_name=model_path,
            model_api_base=candidate_server_url,
            output_dir="/tmp/eval_output",
            max_workers=max_workers
        )

        print("Judging answers...")
        overall_score, qa_pairs, turn_scores, error_rate = mt_bench_judgment.generate_judgment(
            model_name=model_path,
            judge_model_name=judge_model_name,
            model_api_base=judge_endpoint,
            api_key=judge_api_key,
            output_dir="/tmp/eval_output",
            max_workers=max_workers,
            merge_system_user_message=merge_system_user_message
        )

        stop_vllm_server_by_name()

        mt_bench_data = {
            "report_title": "SKILLS EVALUATION REPORT",
            "model": model_path,
            "judge_model": judge_model_name,
            "overall_score": overall_score,
            "turn_scores": turn_scores,
            "qa_scores": qa_pairs,
            "error_rate": error_rate,
        }

        all_mt_bench_data.append(mt_bench_data)
        scores[model_path] = overall_score

    with open(mt_bench_output.path, 'w') as f:
        json.dump(all_mt_bench_data, f, indent=4)

    outputs = NamedTuple('outputs', best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    return outputs(best_model=best_model, best_score=best_score)

@component(base_image=PYTHON_IMAGE)
def load_mt_bench_results_op(mt_bench_output: Input[Artifact]) -> list:
    import json

    mt_bench_score_list = []
    with open(mt_bench_output.path, 'r') as f:
         mt_bench_score_list = json.load(f)

    print("MT_Bench Evaluation Data:")
    for mt_bench_score in mt_bench_score_list:
        print(json.dumps(mt_bench_score, indent=4))

    return mt_bench_score_list
