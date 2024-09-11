# type: ignore
# pylint: disable=unused-argument,missing-function-docstring
from kfp import dsl
from ..consts import PYTHON_IMAGE


@dsl.component(base_image=PYTHON_IMAGE)
def kubectl_apply_op(manifest: str):
    return


@dsl.component(base_image=PYTHON_IMAGE)
def kubectl_wait_for_op(condition: str, kind: str, name: str):
    return

@dsl.component(base_image=PYTHON_IMAGE)
def huggingface_importer_op(model: dsl.Output[dsl.Model], repo_name: str):
    return

@dsl.component(base_image=PYTHON_IMAGE)
def pvc_to_artifact_op(model: dsl.Output[dsl.Artifact], pvc_path: str):
    return

@dsl.component(base_image=PYTHON_IMAGE)
def pvc_to_model_op(model: dsl.Output[dsl.Model], pvc_path: str):
    return
