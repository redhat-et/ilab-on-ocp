# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member,missing-function-docstring

from kfp import dsl

from .consts import PYTHON_IMAGE, RHELAI_IMAGE, TOOLBOX_IMAGE


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


@dsl.container_component
def model_to_pvc_op(model: dsl.Input[dsl.Model], pvc_path: str = "/model"):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {model.path}/* {pvc_path}"],
    )


@dsl.container_component
def ilab_importer_op(repository: str, release: str, base_model: dsl.Output[dsl.Model]):
    return dsl.ContainerSpec(
        RHELAI_IMAGE,
        ["/bin/sh", "-c"],
        [
            f"ilab --config=DEFAULT model download --repository {repository} --release {release} --model-dir {base_model.path}"
        ],
    )
