# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring,unused-argument

from typing import NamedTuple, Optional

from kfp import dsl

from utils.consts import PYTHON_IMAGE, TOOLBOX_IMAGE


@dsl.component(base_image=PYTHON_IMAGE)
def pytorchjob_manifest_op(
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
) -> NamedTuple("outputs", manifest=str, name=str):
    Outputs = NamedTuple("outputs", manifest=str, name=str)
    return Outputs("", "")


@dsl.component(base_image=PYTHON_IMAGE)
def data_processing_op(
    model_path: str = "/model",
    sdg_path: str = "/data/sdg",
    skills_path: str = "/data/skills",
    knowledge_path: str = "/data/knowledge",
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000,
):
    return


@dsl.container_component
def skills_processed_data_to_artifact_op(
    skills_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/skills",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {skills_processed_data.path}"],
    )


@dsl.container_component
def knowledge_processed_data_to_artifact_op(
    knowledge_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/knowledge",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {knowledge_processed_data.path}"],
    )
