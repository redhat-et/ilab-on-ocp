# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member
import typing
from typing import List, Literal, Optional

import click
from kfp import compiler, dsl
from kfp.kubernetes import (
    CreatePVC,
    DeletePVC,
    mount_pvc,
    set_image_pull_policy,
    set_image_pull_secrets,
    use_config_map_as_env,
    use_secret_as_env,
)

# TODO: remove me once the KFP package is part of the RHELAI image
RHELAI_SITE_PACKAGES = "/opt/app-root/lib64/python3.11/site-packages/"
K8S_NAME = "kfp-model-server"
JUDGE_CONFIG_MAP = "kfp-model-server"
JUDGE_SECRET = "judge-server"
MOCKED_STAGES = ["sdg", "train", "eval"]
PIPELINE_FILE_NAME = "pipeline.yaml"
SDG_PIPELINE = "simple"
IMAGE_PULL_SECRET = "redhat-et-ilab-botty-pull-secret"
STANDALONE_TEMPLATE_FILE_NAME = "standalone.tpl"
GENERATED_STANDALONE_FILE_NAME = "standalone.py"
DEFAULT_REPO_URL = "https://github.com/instructlab/taxonomy.git"
KFP_MODEL_SERVER_CM = "sdg/kfp-model-server.yaml"
BASE_MODEL = "ibm-granite/granite-7b-base"

# eval args
MMLU_TASKS_LIST = "mmlu_anatomy,mmlu_astronomy"
MODEL_DTYPE = "bfloat16"
FEW_SHOTS = 5
# BATCH_SIZE can also be an int, for example "8" is converted to an int in eval/final
BATCH_SIZE = "auto"
MAX_WORKERS = "auto"
MERGE_SYSTEM_USER_MESSAGE = False

# training args
NUM_EPOCHS_PHASE_1 = 2
NUM_EPOCHS_PHASE_2 = 2
EFFECTIVE_BATCH_SIZE = 3840
LEARNING_RATE = 1e-4
NUM_WARMUP_STEPS = 800
SAVE_SAMPLES = 0
MAX_BATCH_LEN = 20000
SEED = 42


def pipeline_wrapper(mock: List[Literal[MOCKED_STAGES]]):
    """Wrapper for KFP pipeline, which allows for mocking individual stages."""

    # Imports for SDG stage
    if mock is not None and "sdg" in mock:
        from sdg.faked import (
            git_clone_op,
            sdg_op,
            sdg_to_artifact_op,
            taxonomy_to_artifact_op,
        )
    else:
        from sdg import (
            git_clone_op,
            sdg_op,
            sdg_to_artifact_op,
            taxonomy_to_artifact_op,
        )

    # Imports for Training stage
    if mock is not None and "train" in mock:
        from training.faked import (
            data_processing_op,
            knowledge_processed_data_to_artifact_op,
            pytorchjob_manifest_op,
            skills_processed_data_to_artifact_op,
        )
        from utils.faked import (
            huggingface_importer_op,
            kubectl_apply_op,
            kubectl_wait_for_op,
            pvc_to_model_op,
            pvc_to_mt_bench_op,
        )
    else:
        from training import (
            data_processing_op,
            knowledge_processed_data_to_artifact_op,
            pytorchjob_manifest_op,
            skills_processed_data_to_artifact_op,
        )
        from utils import (
            huggingface_importer_op,
            kubectl_apply_op,
            kubectl_wait_for_op,
            pvc_to_model_op,
            pvc_to_mt_bench_op,
        )

    # Imports for evaluation
    from eval.final import run_final_eval_op
    from eval.mmlu import load_mmlu_results_op, run_mmlu_op

    ## from eval.mmlu import run_mmlu_op, load_mmlu_results_op
    from eval.mt_bench import run_mt_bench_op
    from utils import list_models_in_directory_op

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
        storage_class_name: str = "nfs-csi",
        base_model: str = BASE_MODEL,
        # minimal subset of MMLU_TASKS
        # mmlu_tasks_list: str = MMLU_TASKS_LIST,
        model_dtype: str = MODEL_DTYPE,
        few_shots: int = FEW_SHOTS,
        batch_size: str = BATCH_SIZE,
        max_workers: str = MAX_WORKERS,
        merge_system_user_message: bool = MERGE_SYSTEM_USER_MESSAGE,
        device: str = None,
        nproc_per_node: int = 3,
        nnodes: int = 2,
        num_epochs_phase_1: int = NUM_EPOCHS_PHASE_1,
        num_epochs_phase_2: int = NUM_EPOCHS_PHASE_2,
        effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        num_warmup_steps: int = NUM_WARMUP_STEPS,
        save_samples: int = SAVE_SAMPLES,
        sdg_pipeline: str = SDG_PIPELINE,
        max_batch_len: int = MAX_BATCH_LEN,
        seed: int = SEED,
    ):
        # SDG stage
        sdg_input_pvc_task = CreatePVC(
            pvc_name_suffix="-sdg",
            access_modes=["ReadWriteMany"],
            size="10Gi",
            storage_class_name=storage_class_name,
        )
        git_clone_task = git_clone_op(
            repo_branch=repo_branch,
            repo_pr=repo_pr if repo_pr and repo_pr > 0 else None,
            repo_url=repo_url,
        )
        mount_pvc(
            task=git_clone_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        git_clone_task.set_caching_options(False)

        sdg_task = sdg_op(
            num_instructions_to_generate=num_instructions_to_generate,
            pipeline=sdg_pipeline,
            repo_branch=repo_branch,
            repo_pr=repo_pr,
        )
        sdg_task.set_env_variable("HOME", "/tmp")
        sdg_task.set_env_variable("HF_HOME", "/tmp")
        # TODO: remove this once the KFP package is part of the RHELAI image
        sdg_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )
        use_config_map_as_env(
            sdg_task, K8S_NAME, dict(endpoint="endpoint", model="model")
        )
        use_secret_as_env(sdg_task, K8S_NAME, {"api_key": "api_key"})
        sdg_task.after(git_clone_task)
        mount_pvc(
            task=sdg_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        sdg_task.set_caching_options(False)

        # Upload "sdg" and "taxonomy" artifacts to S3 without blocking the rest of the workflow
        taxonomy_to_artifact_task = taxonomy_to_artifact_op()
        taxonomy_to_artifact_task.after(git_clone_task, sdg_task)
        mount_pvc(
            task=taxonomy_to_artifact_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        sdg_to_artifact_task = sdg_to_artifact_op()
        sdg_to_artifact_task.after(git_clone_task, sdg_task)
        mount_pvc(
            task=sdg_to_artifact_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )

        set_image_pull_secrets(sdg_task, [IMAGE_PULL_SECRET])

        # uncomment if updating image with same tag
        # set_image_pull_policy(sdg_task, "Always")

        # Training stage

        # We need to pass storage_class_name as "" to use the default StorageClass, if left empty, KFP uses "standard" StorageClass.
        # 'standard' !=  default StorageClass
        # https://github.com/kubeflow/pipelines/blob/1cded35cf5e93d8c8d32fefbddceb2eed8de9a0a/backend/src/v2/driver/driver.go#L1428-L1436
        # At least we made it a pipeline parameter
        model_pvc_task = CreatePVC(
            pvc_name_suffix="-model-cache",
            access_modes=["ReadWriteMany"],
            size="100Gi",
            storage_class_name=storage_class_name,
        )
        model_to_pvc_task = huggingface_importer_op(repo_name=base_model)
        # TODO: remove this once the KFP package is part of the RHELAI image
        model_to_pvc_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )
        model_to_pvc_task.set_caching_options(False)
        mount_pvc(
            task=model_to_pvc_task, pvc_name=model_pvc_task.output, mount_path="/model"
        )

        # Data processing
        data_processing_task = data_processing_op()
        # TODO: remove this once the KFP package is part of the RHELAI image
        data_processing_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )
        mount_pvc(
            task=data_processing_task,
            pvc_name=model_pvc_task.output,
            mount_path="/model",
        )
        mount_pvc(
            task=data_processing_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        data_processing_task.after(model_to_pvc_task, sdg_task)
        data_processing_task.set_caching_options(False)

        # Upload "skills_processed_data" and "knowledge_processed_data" artifacts to S3 without blocking the rest of the workflow
        skills_processed_data_to_artifact_task = skills_processed_data_to_artifact_op()
        skills_processed_data_to_artifact_task.after(data_processing_task)
        mount_pvc(
            task=skills_processed_data_to_artifact_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        skills_processed_data_to_artifact_task.set_caching_options(False)
        knowledge_processed_data_to_artifact_task = (
            knowledge_processed_data_to_artifact_op()
        )
        knowledge_processed_data_to_artifact_task.after(data_processing_task)
        mount_pvc(
            task=knowledge_processed_data_to_artifact_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        knowledge_processed_data_to_artifact_task.set_caching_options(False)

        output_pvc_task = CreatePVC(
            pvc_name_suffix="-output",
            access_modes=["ReadWriteMany"],
            size="100Gi",
            storage_class_name=storage_class_name,
        )

        # Using pvc_create_task.output as PyTorchJob name since dsl.PIPELINE_* global variables do not template/work in KFP v2
        # https://github.com/kubeflow/pipelines/issues/10453
        pytorchjob_manifest_task = pytorchjob_manifest_op(
            model_pvc_name=model_pvc_task.output,
            input_pvc_name=sdg_input_pvc_task.output,
            name_suffix=sdg_input_pvc_task.output,
            output_pvc_name=output_pvc_task.output,
            phase_num=1,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            num_epochs=num_epochs_phase_1,
            effective_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            num_warmup_steps=num_warmup_steps,
            save_samples=save_samples,
            max_batch_len=max_batch_len,
            seed=seed,
        )
        # TODO: remove this once the KFP package is part of the RHELAI image
        pytorchjob_manifest_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )
        pytorchjob_manifest_task.set_caching_options(False)

        kubectl_apply_task = kubectl_apply_op(
            manifest=pytorchjob_manifest_task.outputs["manifest"]
        )
        kubectl_apply_task.after(data_processing_task, model_to_pvc_task)
        kubectl_apply_task.set_caching_options(False)

        kubectl_wait_task = kubectl_wait_for_op(
            condition="condition=Succeeded",
            kind="pytorchjobs",
            name=pytorchjob_manifest_task.outputs["name"],
        )
        kubectl_wait_task.after(kubectl_apply_task)
        kubectl_wait_task.set_caching_options(False)

        # # MMLU Evaluation of models

        # models_list_task = list_models_in_directory_op(
        #     models_folder="/output/model/model/hf_format",
        # )
        # models_list_task.set_caching_options(False)

        # models_list_task.after(kubectl_wait_task)

        # mount_pvc(
        #     task=models_list_task,
        #     pvc_name=output_pvc_task.output,
        #     mount_path="/output/model",
        # )

        # run_mmlu_task = run_mmlu_op(
        #     models_list=models_list_task.output,
        #     models_path_prefix="/output/model/hf_format",
        #     mmlu_tasks_list=mmlu_tasks_list,
        #     model_dtype=model_dtype,
        #     few_shots=few_shots,
        #     batch_size=batch_size,
        #     device=device,
        # )

        # run_mmlu_task.set_caching_options(False)

        # mount_pvc(
        #     task=run_mmlu_task, pvc_name=output_pvc_task.output, mount_path="/output"
        # )

        # load_mmlu_results_task = load_mmlu_results_op(
        #     mmlu_output=run_mmlu_task.outputs["mmlu_output"],
        # )

        # run_mmlu_task.set_accelerator_type("nvidia.com/gpu")
        # run_mmlu_task.set_accelerator_limit(1)

        # #    Run training on MMLU best-model
        # #    Run final eval on best scored mt_bench candidate
        # #    For now, running mt_bench on same output models as training phase 1
        # #    TODO: Another training phase, using the best-model from MMLU as base

        #### Train 2

        pytorchjob_manifest_2_task = pytorchjob_manifest_op(
            model_pvc_name=model_pvc_task.output,
            input_pvc_name=sdg_input_pvc_task.output,
            name_suffix=sdg_input_pvc_task.output,
            output_pvc_name=output_pvc_task.output,
            phase_num=2,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            num_epochs=num_epochs_phase_2,
            effective_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            num_warmup_steps=num_warmup_steps,
            save_samples=save_samples,
            max_batch_len=max_batch_len,
            seed=seed,
        )
        # TODO: remove this once the KFP package is part of the RHELAI image
        pytorchjob_manifest_2_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )

        pytorchjob_manifest_2_task.set_caching_options(False)
        pytorchjob_manifest_2_task.after(kubectl_wait_task)

        mount_pvc(
            task=pytorchjob_manifest_2_task,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )

        kubectl_apply_2_task = kubectl_apply_op(
            manifest=pytorchjob_manifest_2_task.outputs["manifest"]
        )
        kubectl_apply_2_task.set_caching_options(False)

        kubectl_wait_2_task = kubectl_wait_for_op(
            condition="condition=Succeeded",
            kind="pytorchjobs",
            name=pytorchjob_manifest_2_task.outputs["name"],
        )
        kubectl_wait_2_task.after(kubectl_apply_2_task)
        kubectl_wait_2_task.set_caching_options(False)

        ###

        models_list_2_task = list_models_in_directory_op(
            models_folder="/output/phase_2/model/hf_format",
        )
        # TODO: remove this once the KFP package is part of the RHELAI image
        models_list_2_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )
        models_list_2_task.set_caching_options(False)
        models_list_2_task.after(kubectl_wait_2_task)
        mount_pvc(
            task=models_list_2_task,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )

        ###

        # MT_Bench Evaluation of models

        run_mt_bench_task = run_mt_bench_op(
            models_list=models_list_2_task.output,
            models_path_prefix="/output/phase_2/model/hf_format",
            max_workers=max_workers,
            merge_system_user_message=merge_system_user_message,
            device=device,
        )
        # TODO: remove this once the KFP package is part of the RHELAI image
        run_mt_bench_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )

        mount_pvc(
            task=run_mt_bench_task,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )
        run_mt_bench_task.set_env_variable("HOME", "/tmp")
        run_mt_bench_task.set_env_variable("HF_HOME", "/tmp")
        run_mt_bench_task.set_accelerator_type("nvidia.com/gpu")
        run_mt_bench_task.set_accelerator_limit(1)
        run_mt_bench_task.set_caching_options(False)
        use_config_map_as_env(
            run_mt_bench_task,
            JUDGE_CONFIG_MAP,
            dict(endpoint="JUDGE_ENDPOINT", model="JUDGE_NAME"),
        )
        set_image_pull_secrets(run_mt_bench_task, [IMAGE_PULL_SECRET])
        use_secret_as_env(run_mt_bench_task, JUDGE_SECRET, {"api_key": "JUDGE_API_KEY"})

        # uncomment if updating image with same tag
        # set_image_pull_policy(run_mt_bench_task, "Always")

        final_eval_task = run_final_eval_op(
            candidate_model="/output/phase_2/model/hf_format/candidate_model",
            # TODO: DO we need both candidate_branch and base_branch
            base_branch=repo_branch,
            candidate_branch=repo_branch,
            device=device,
            base_model_dir="/model/",
            max_workers=max_workers,
            merge_system_user_message=merge_system_user_message,
            model_dtype=model_dtype,
            few_shots=few_shots,
            batch_size=batch_size,
        )
        # TODO: remove this once the KFP package is part of the RHELAI image
        final_eval_task.set_env_variable(
            name="PYTHONPATH",
            value=f"$PYTHONPATH:{RHELAI_SITE_PACKAGES}",
        )

        mount_pvc(
            task=final_eval_task, pvc_name=output_pvc_task.output, mount_path="/output"
        )
        mount_pvc(
            task=final_eval_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/input",
        )
        mount_pvc(
            task=final_eval_task,
            pvc_name=model_pvc_task.output,
            mount_path="/model",
        )

        use_config_map_as_env(
            final_eval_task,
            JUDGE_CONFIG_MAP,
            dict(endpoint="JUDGE_ENDPOINT", model="JUDGE_NAME"),
        )

        final_eval_task.set_env_variable("HOME", "/tmp")
        final_eval_task.set_env_variable("HF_HOME", "/tmp")
        set_image_pull_secrets(final_eval_task, [IMAGE_PULL_SECRET])

        # uncomment if updating image with same tag
        # set_image_pull_policy(final_eval_task, "Always")

        use_secret_as_env(final_eval_task, JUDGE_SECRET, {"api_key": "JUDGE_API_KEY"})

        final_eval_task.after(run_mt_bench_task)
        final_eval_task.set_accelerator_type("nvidia.com/gpu")
        final_eval_task.set_accelerator_limit(1)

        output_model_task = pvc_to_model_op(
            pvc_path="/output/phase_2/model/hf_format/candidate_model",
        )
        output_model_task.after(run_mt_bench_task)
        mount_pvc(
            task=output_model_task,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )

        output_mt_bench_task = pvc_to_mt_bench_op(
            pvc_path="/output/mt_bench_data.json",
        )
        output_mt_bench_task.after(run_mt_bench_task)
        mount_pvc(
            task=output_mt_bench_task,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )

        output_pvc_delete_task = DeletePVC(pvc_name=output_pvc_task.output)
        output_pvc_delete_task.after(
            output_model_task, output_mt_bench_task, final_eval_task
        )

        sdg_pvc_delete_task = DeletePVC(pvc_name=sdg_input_pvc_task.output)
        sdg_pvc_delete_task.after(final_eval_task)

        model_pvc_delete_task = DeletePVC(pvc_name=model_pvc_task.output)
        model_pvc_delete_task.after(final_eval_task)

        return

    return pipeline


@click.option(
    "--mock",
    type=click.Choice(MOCKED_STAGES, case_sensitive=False),
    help="Mock part of the pipeline",
    multiple=True,
    default=[],
)
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context, mock):
    if ctx.invoked_subcommand is None:
        generate_pipeline(mock)


def generate_pipeline(mock):
    p = pipeline_wrapper(mock)

    with click.progressbar(length=1, label="Generating pipeline") as bar:
        compiler.Compiler().compile(p, PIPELINE_FILE_NAME)
        bar.update(1)


@cli.command(name="gen-standalone")
def gen_standalone():
    """
    Generates a standalone script that mimics the behavior of the pipeline.

    This function should be used when Kubeflow Pipelines are not available. It will generate a
    script that replicates the pipeline's functionality.

    Example usage: ''' $ python pipeline.py gen-standalone '''
    """
    from os import path

    import yaml
    from jinja2 import Template
    from jinja2.exceptions import TemplateSyntaxError

    click.echo("Generating pipeline YAML file...")
    try:
        generate_pipeline(mock=None)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.exceptions.Exit(1)

    # Load the YAML pipeline file which contains multiple documents
    with open(PIPELINE_FILE_NAME, "r", encoding="utf-8") as file:
        try:
            documents = list(yaml.safe_load_all(file))
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise click.exceptions.Exit(1)

    # The list of executor names to extract details from to generate the standalone script
    executors = {
        "exec-data-processing-op": 'data_processing_op(max_seq_len={MAX_SEQ_LEN}, max_batch_len={MAX_BATCH_LEN}, sdg_path="{DATA_PVC_SDG_PATH}", model_path="{DATA_PVC_MODEL_PATH}", skills_path="{PREPROCESSED_DATA_SKILLS_PATH}", knowledge_path="{PREPROCESSED_DATA_KNOWLEDGE_PATH}")',
        "exec-sdg-op": 'sdg_op(num_instructions_to_generate={num_instructions_to_generate}, pipeline="{sdg_pipeline}", repo_branch="{exec_git_clone_op_repo_branch}", repo_pr={exec_git_clone_op_repo_pr}, taxonomy_path="{TAXONOMY_DATA_PATH}", sdg_path="{DATA_PVC_SDG_PATH}")',
        "exec-git-clone-op": {},
        "exec-huggingface-importer-op": 'huggingface_importer_op(repo_name="{REPO_GRANITE_7B_IMAGE}", model_path="{DATA_PVC_MODEL_PATH}")',
        "exec-run-mt-bench-op": 'run_mt_bench_op(best_score_file="{MT_BENCH_SCORES_PATH}",output_path="{MT_BENCH_OUTPUT_PATH}",models_folder="{CANDIDATE_MODEL_PATH_PREFIX}",models_path_prefix="{CANDIDATE_MODEL_PATH_PREFIX}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE})',
        "exec-run-final-eval-op": 'run_final_eval_op(mmlu_branch_output="{MMLU_BRANCH_SCORES_PATH}", mt_bench_branch_output="{MT_BENCH_BRANCH_SCORES_PATH}", candidate_model="{CANDIDATE_MODEL_PATH}", taxonomy_path="{TAXONOMY_PATH}", sdg_path="{DATA_PVC_SDG_PATH}", base_branch="", candidate_branch="", device=None, base_model_dir="{DATA_PVC_MODEL_PATH}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE}, model_dtype="{MODEL_DTYPE}", few_shots={FEW_SHOTS}, batch_size="{BATCH_SIZE}")',
    }

    details = {}
    for executor_name, executor_input_param in executors.items():
        try:
            executor_name_camelize = executor_name.replace("-", "_")
            # replace "-" with "_" in executor_name to match the key in the details dictionary
            executor_details = get_executor_details(documents, executor_name)
            if executor_details is not None:
                details[executor_name_camelize + "_image"] = executor_details["image"]
                details[executor_name_camelize + "_command"] = (
                    change_dsl_function_to_normal_function(executor_details["command"])
                )
                if executor_name == "exec-git-clone-op":
                    details[executor_name_camelize + "_args"] = remove_template_markers(
                        executor_details["args"],
                        executor_name_camelize,
                        executor_input_param,
                    )
                else:
                    details[executor_name_camelize + "_args"] = executor_input_param

        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise click.exceptions.Exit(1)

    # Open the template file
    try:
        standalone_template_path = path.join(
            "standalone", STANDALONE_TEMPLATE_FILE_NAME
        )
        with open(standalone_template_path, "r", encoding="utf-8") as template_file:
            template_content = template_file.read()
    except FileNotFoundError as e:
        click.echo(
            f"Error: The template file '{standalone_template_path}' was not found.",
            err=True,
        )
        raise click.exceptions.Exit(1) from e
    except IOError as e:
        click.echo(
            f"Error: An I/O error occurred while reading '{standalone_template_path}': {e}",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # Prepare the Jinja2 Template
    try:
        template = Template(template_content)
    except TemplateSyntaxError as e:
        click.echo(
            f"Error: The template file '{standalone_template_path}' contains a syntax error: {e}",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # Render the template with dynamic values
    rendered_code = template.render(details)

    # Write the rendered code to a new Python file
    standalone_script_path = path.join("standalone", GENERATED_STANDALONE_FILE_NAME)
    with open(standalone_script_path, "w", encoding="utf-8") as output_file:
        output_file.write(rendered_code)

    click.echo(f"Successfully generated '{standalone_script_path}' script.")


def get_executor_details(
    documents: typing.List[typing.Dict[str, typing.Any]], executor_name: str
) -> dict | None:
    """
    Extracts the command, args, and image of a given executor container from the provided YAML
    documents.

    Args:
        documents (List[Dict[str, Any]]): List of YAML documents loaded as dictionaries.
        executor_name (str): The name of the executor to search for.

    Returns:
        dict: A dictionary containing the 'command', 'args', and 'image' of the executor container
        if found, otherwise raise en error.
    """
    spec = "deploymentSpec"
    deployment_spec_found = False
    for doc in documents:
        deployment_spec = doc.get(spec)
        if not deployment_spec:
            continue
        else:
            deployment_spec_found = True
        for executors_value in deployment_spec.values():
            for executor, executor_value in executors_value.items():
                if executor == executor_name:
                    container = executor_value.get("container", {})
                    if not all(
                        key in container for key in ("command", "args", "image")
                    ):
                        raise ValueError(
                            f"Executor '{executor_name}' does not have the required "
                            "'command', 'args', or 'image' fields."
                        )
                    return {
                        "command": container["command"],
                        "args": container["args"],
                        "image": container["image"],
                    }
        print(f"Executor '{executor_name}' not found in the provided {spec} document.")
        return None
    if not deployment_spec_found:
        raise ValueError(
            "The provided documents do not contain a 'deploymentSpec' key."
        )


def remove_template_markers(
    rendered_code: list, executor_name: str, executor_input_param: str
) -> list:
    """
    Removes the Jinja2 template markers from each element of the rendered code list.

    Args:
        rendered_code (list): The list of rendered code elements containing Jinja2 template markers.

    Returns:
        list: The list of rendered code elements with Jinja2 template markers removed.

    Examples with an executor name of 'exec':
        Input: ["{{$.inputs.parameters['repo_name']}}", "{{$.inputs.parameters['model']}}"]
        Output: ["{exec_repo_name}", "{exec_model}"]

    """
    import json
    import re

    pattern = r"\{\{\$\.inputs\.parameters\['([^']+)'\]\}\}"
    rendered_code = [
        re.sub(pattern, r"{%s_\1}" % executor_name, element)
        for element in rendered_code
    ]

    # TODO: find a better approach
    # Only useful for git_clone_op at the moment
    # additionally remove {{$.outputs.artifacts[\'taxonomy\'].path}}
    pattern = r"\{\{\$\.outputs\.artifacts\['([^']+)'\]\.path\}\}"
    rendered_code = [
        re.sub(pattern, r"{TAXONOMY_PATH}", element) for element in rendered_code
    ]

    # Replace '{{$}}' with input_param
    pattern = r"\{\{\$\}\}"
    rendered_code = [
        re.sub(pattern, json.dumps(executor_input_param), element)
        for element in rendered_code
    ]

    return rendered_code


def change_dsl_function_to_normal_function(rendered_code: list):
    replacements = {
        "dsl.Input[dsl.Dataset]": "str",
        "dsl.Input[dsl.Model]": "str",
        "dsl.Input[dsl.Artifact]": "str",
        "dsl.Output[dsl.Dataset]": "str",
        "dsl.Output[dsl.Model]": "str",
        "Output[Artifact]": "str",
        "Input[Dataset]": "str",
        "import kfp": "",
        "from kfp import dsl": "",
        "from kfp.dsl import *": "",
    }

    import re

    # Regular expression to match ".path" but not "os.path"
    path_pattern = re.compile(r"(?<!os)\.path")

    def remove_path_not_os_path(line):
        return path_pattern.sub("", line)

    rendered_code = [remove_path_not_os_path(line) for line in rendered_code]

    for old, new in replacements.items():
        rendered_code = [line.replace(old, new) for line in rendered_code]
    return rendered_code[-1].strip()


if __name__ == "__main__":
    cli()
