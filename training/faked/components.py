# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring,unused-argument

from typing import NamedTuple

from kfp import dsl

from utils.consts import PYTHON_IMAGE


@dsl.component(base_image=PYTHON_IMAGE)
def pytorchjob_manifest_op(
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
) -> NamedTuple("outputs", manifest=str, name=str):
    Outputs = NamedTuple("outputs", manifest=str, name=str)
    return Outputs("", "")
