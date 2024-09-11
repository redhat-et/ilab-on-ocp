# type: ignore
# pylint: disable=unused-argument
from typing import Optional
from kfp import dsl

IMAGE = "registry.access.redhat.com/ubi9/python-311:latest"

@dsl.component(base_image=IMAGE)
def git_clone_op(
    taxonomy: dsl.Output[dsl.Dataset],
    repo_branch: str,
    repo_pr: Optional[int],
    repo_url: Optional[str],
):
    return


@dsl.component(base_image=IMAGE, packages_to_install=["git+https://github.com/redhat-et/ilab-on-ocp.git#subdirectory=sdg/faked/fixtures"])
def sdg_op(
    num_instructions_to_generate: int,
    taxonomy: dsl.Input[dsl.Dataset],
    sdg: dsl.Output[dsl.Dataset],
    repo_branch: Optional[str],
    repo_pr: Optional[int],
):
    import sys
    from pathlib import Path
    import shutil

    shutil.copytree(Path(sys.prefix) / "sdg_fixtures", sdg.path, dirs_exist_ok=True)
    return
