# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member
from typing import Optional

from kfp import dsl

from utils.consts import RHELAI_IMAGE, TOOLBOX_IMAGE


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


@dsl.component(base_image=RHELAI_IMAGE, install_kfp_package=False)
def sdg_op(
    num_instructions_to_generate: int,
    pipeline: str,
    repo_branch: Optional[str],
    repo_pr: Optional[int],
    taxonomy_path: str = "/data/taxonomy",
    sdg_path: str = "/data/sdg",
    sdg_sampling_size: float = 1.0,
):
    from os import getenv, path

    import openai
    import yaml
    from instructlab.sdg import generate_data
    from instructlab.sdg.utils.taxonomy import read_taxonomy

    def set_precomputed_skills_data_ratio(sampling_size: float):
        skills_recipe = "/usr/share/instructlab/sdg/default_data_recipes/skills.yaml"
        if path.exists(skills_recipe):
            with open(skills_recipe, "r") as file:
                skills_yaml = yaml.load(file, Loader=yaml.Loader)

            skills_yaml["datasets"][0]["sampling_size"] = sampling_size

            with open(skills_recipe, "w", encoding="utf-8") as file:
                yaml.dump(skills_yaml, file)

    api_key = getenv("api_key")
    model = getenv("model")
    endpoint = getenv("endpoint")

    if sdg_ca_cert := getenv("SDG_CA_CERT_PATH"):
        import httpx

        custom_http_client = httpx.Client(verify=sdg_ca_cert)
        client = openai.OpenAI(
            base_url=endpoint, api_key=api_key, http_client=custom_http_client
        )
    else:
        client = openai.OpenAI(base_url=endpoint, api_key=api_key)

    taxonomy_base = "main" if repo_branch or (repo_pr and int(repo_pr) > 0) else "empty"

    print("Generating synthetic dataset for:")
    print()
    print(
        read_taxonomy(
            taxonomy_path, taxonomy_base, document_output_dir=f"{sdg_path}/documents"
        )
    )

    # sset_precomputed_skills_data_ratio(sampling_size=sdg_sampling_size)

    # generate_data has a magic word for its taxonomy_base argument - 'empty'
    # it allows generating from the whole repo, see:
    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230
    generate_data(
        client=client,
        num_instructions_to_generate=num_instructions_to_generate,
        output_dir=sdg_path,
        taxonomy=taxonomy_path,
        taxonomy_base=taxonomy_base,
        model_name=model,
        pipeline=pipeline,
        chunk_word_count=1000,
        server_ctx_size=4096,
    )


@dsl.container_component
def taxonomy_to_artifact_op(
    taxonomy: dsl.Output[dsl.Dataset], pvc_path: str = "/data/taxonomy"
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {taxonomy.path}"],
    )


@dsl.container_component
def sdg_to_artifact_op(sdg: dsl.Output[dsl.Dataset], pvc_path: str = "/data/sdg"):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {sdg.path}"],
    )
