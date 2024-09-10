# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member
from typing import List, Literal, Optional
import click
from kfp import dsl, compiler
from kfp.kubernetes import use_config_map_as_env, use_secret_as_env

K8S_NAME = "kfp-model-server"
MOCKED_STAGES = ['sdg', 'train', 'eval']

def pipeline_wrapper(mock: List[Optional[Literal[MOCKED_STAGES]]]):
    """Wrapper for KFP pipeline, which allows for mocking individual stages."""
    if 'sdg' in mock:
        from sdg.faked import git_clone_op, sdg_op
    else:
        from sdg import git_clone_op, sdg_op


    @dsl.pipeline(
        display_name="InstructLab",
        name="instructlab",
        description="InstructLab pipeline",
    )
    def pipeline(
        num_instructions_to_generate: int = 2,
        repo_url: str = "https://github.com/instructlab/taxonomy.git",
        repo_branch: Optional[str] = None,
        repo_pr: Optional[int] = None,
    ):
        git_clone_task = git_clone_op(
            repo_branch=repo_branch, repo_pr=repo_pr, repo_url=repo_url
        )

        sdg_task = sdg_op(
            num_instructions_to_generate=num_instructions_to_generate,
            taxonomy=git_clone_task.outputs["taxonomy"],
            repo_branch=repo_branch,
            repo_pr=repo_pr,
        )

        # For example on K8S object to populate see kfp-model-server.yaml
        use_config_map_as_env(sdg_task, K8S_NAME, dict(endpoint="endpoint", model="model"))
        use_secret_as_env(sdg_task, K8S_NAME, {"api_key": "api_key"})

        return
    return pipeline

@click.command()
@click.option('--mock', type=click.Choice(MOCKED_STAGES, case_sensitive=False), help="Mock part of the pipeline", multiple=True, default=[])
def cli(mock):

    p = pipeline_wrapper(mock)

    with click.progressbar(length=1, label="Generating pipeline") as bar:
        compiler.Compiler().compile(p, "pipeline.yaml")
        bar.update(1)


if __name__ == "__main__":
    cli()
