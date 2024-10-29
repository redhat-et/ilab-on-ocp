# type: ignore
# pylint: disable=unused-argument
from typing import Optional

from kfp import dsl

from utils.consts import PYTHON_IMAGE, TOOLBOX_IMAGE


@dsl.container_component
def git_clone_op(
    repo_branch: str,
    repo_pr: Optional[int],
    repo_url: Optional[str],
    taxonomy_path: str = "/data/taxonomy",
):
    return dsl.ContainerSpec(
        "registry.access.redhat.com/ubi9/toolbox",
        ["/bin/sh", "-c"],
        [
            f"git clone {repo_url} {taxonomy_path} && cd {taxonomy_path} && "
            + f'if [ -n "{repo_branch}" ]; then '
            + f"git fetch origin {repo_branch} && git checkout {repo_branch}; "
            + f'elif [ -n "{repo_pr}" ] && [ {repo_pr} -gt 0 ]; then '
            + f"git fetch origin pull/{repo_pr}/head:{repo_pr} && git checkout {repo_pr}; fi "
        ],
    )


@dsl.component(
    base_image=PYTHON_IMAGE,
    packages_to_install=[
        "git+https://github.com/redhat-et/ilab-on-ocp.git#subdirectory=sdg/faked/fixtures"
    ],
    use_venv=True,
)
def sdg_op(
    num_instructions_to_generate: int,
    pipeline: str,
    repo_branch: Optional[str],
    repo_pr: Optional[int],
    taxonomy_path: str = "/data/taxonomy",
    sdg_path: str = "/data/sdg",
):
    import shutil
    import sys
    from pathlib import Path

    shutil.copytree(Path(sys.prefix) / "sdg_fixtures", sdg_path, dirs_exist_ok=True)
    return


@dsl.container_component
def taxonomy_to_artifact_op(taxonomy: dsl.Output[dsl.Dataset], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {taxonomy.path}"],
    )


@dsl.container_component
def sdg_to_artifact_op(sdg: dsl.Output[dsl.Dataset], pvc_path: str):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {sdg.path}"],
    )
