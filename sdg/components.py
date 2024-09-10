# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member
from typing import Optional
from kfp import dsl

IMAGE = "quay.io/tcoufal/ilab-sdg:latest"

@dsl.container_component
def git_clone_op(
    taxonomy: dsl.Output[dsl.Dataset],
    repo_branch: str,
    repo_pr: Optional[int],
    repo_url: Optional[str],
):
    return dsl.ContainerSpec(
        "registry.access.redhat.com/ubi9/toolbox",
        ["/bin/sh", "-c"],
        [
            f"git clone {repo_url} {taxonomy.path} && cd {taxonomy.path} && "
            + f'if [ ! -z "{repo_branch}" ]; then '
            + f"git fetch origin {repo_branch} && git checkout {repo_branch}; "
            + f'elif [ ! -z "{repo_pr}" ]; then '
            + f"git fetch origin pull/{repo_pr}/head:{repo_pr} && git checkout {repo_pr}; fi "
        ],
    )


@dsl.component(base_image=IMAGE)
def sdg_op(
    num_instructions_to_generate: int,
    taxonomy: dsl.Input[dsl.Dataset],
    sdg: dsl.Output[dsl.Dataset],
    repo_branch: Optional[str],
    repo_pr: Optional[int],
):
    import openai
    from instructlab.sdg import generate_data
    from instructlab.sdg.utils.taxonomy import read_taxonomy
    from os import getenv

    api_key = getenv("api_key")
    model = getenv("model")
    endpoint = getenv("endpoint")
    client = openai.OpenAI(base_url=endpoint, api_key=api_key)

    taxonomy_base = "main" if repo_branch or repo_pr else "empty"

    print("Generating syntetic dataset for:")
    print()
    print(read_taxonomy(taxonomy.path, taxonomy_base))

    # generate_data has a magic word for its taxonomy_base argument - `empty`
    # it allows generating from the whole repo, see:
    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230
    generate_data(
        client=client,
        num_instructions_to_generate=num_instructions_to_generate,
        output_dir=sdg.path,
        taxonomy=taxonomy.path,
        taxonomy_base=taxonomy_base,
        model_name=model,
    )
