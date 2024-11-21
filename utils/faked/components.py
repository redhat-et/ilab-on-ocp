# type: ignore
# pylint: disable=unused-argument,missing-function-docstring
from kfp import dsl

from ..consts import PYTHON_IMAGE


@dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
def model_to_pvc_op(model: dsl.Input[dsl.Model], pvc_path: str = "/model"):
    return


@dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
def pvc_to_mt_bench_op(mt_bench_output: dsl.Output[dsl.Artifact], pvc_path: str):
    return


@dsl.component(base_image=PYTHON_IMAGE, install_kfp_package=False)
def pvc_to_model_op(model: dsl.Output[dsl.Model], pvc_path: str):
    return
