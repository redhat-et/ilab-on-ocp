VLLM_SERVER = "http://localhost:8000/v1"


def launch_vllm(model_path: str, gpu_count: int, retries: int = 60, delay: int = 5):
    import subprocess
    import sys
    import time

    import requests

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

    print(f"Waiting for vLLM server to start at {VLLM_SERVER}...")

    for attempt in range(retries):
        try:
            response = requests.get(f"{VLLM_SERVER}/models")
            if response.status_code == 200:
                print(f"vLLM server is up and running at {VLLM_SERVER}.")
                return
        except requests.ConnectionError:
            pass

        print(
            f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
        )
        time.sleep(delay)

    raise RuntimeError(
        f"Failed to start vLLM server at {VLLM_SERVER} after {retries} retries."
    )


# This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
# Also, the base image does not include `pkill` cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
def stop_vllm():
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
