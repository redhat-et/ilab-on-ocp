# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member,missing-function-docstring
from typing import List

from kfp import dsl

from .consts import OC_IMAGE, PYTHON_IMAGE, TOOLBOX_IMAGE


@dsl.container_component
def kubectl_apply_op(manifest: str):
    return dsl.ContainerSpec(
        OC_IMAGE,
        ["/bin/sh", "-c"],
        [f'echo "{manifest}" | kubectl apply -f -'],
    )


@dsl.container_component
def kubectl_wait_for_op(
    condition: str,
    kind: str,
    name: str,
    # namespace: Optional[str] = None,
    # timeout: Optional[str] = None,
):
    return dsl.ContainerSpec(
        OC_IMAGE,
        ["/bin/sh", "-c"],
        [
            f"kubectl wait --for={condition} {kind}/{name} --timeout=24h",
        ],
    )


@dsl.container_component
def pvc_to_mt_bench_op(mt_bench_output: dsl.Output[dsl.Artifact], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {mt_bench_output.path}"],
    )


@dsl.container_component
def pvc_to_model_op(model: dsl.Output[dsl.Model], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {model.path}"],
    )


@dsl.component(use_venv=True)
def list_models_in_directory_op(models_folder: str) -> List[str]:
    import os

    models = os.listdir(models_folder)
    return models


@dsl.component(
    base_image=PYTHON_IMAGE, packages_to_install=["huggingface_hub"], use_venv=True
)
def huggingface_importer_op(repo_name: str, model_path: str = "/model"):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_name, cache_dir="/tmp", local_dir=model_path)
