# type: ignore
# pylint: disable=no-value-for-parameter,import-outside-toplevel,import-error,no-member
from typing import List, Literal, Optional
import click
from kfp import dsl, compiler
from kfp.kubernetes import (
    use_config_map_as_env,
    use_secret_as_env,
    CreatePVC,
    DeletePVC,
    mount_pvc,
)

# For now, all external models are the same mistral, but won't be always
K8S_NAME = "kfp-model-server"
JUDGE_CONFIG_MAP = "kfp-model-server"
JUDGE_SECRET = "judge-server"
MOCKED_STAGES = ["sdg", "train", "eval"]


def pipeline_wrapper(mock: List[Literal[MOCKED_STAGES]]):
    """Wrapper for KFP pipeline, which allows for mocking individual stages."""

    # Imports for SDG stage
    if "sdg" in mock:
        from sdg.faked import git_clone_op, sdg_op
    else:
        from sdg import git_clone_op, sdg_op

    # Imports for Training stage
    if "train" in mock:
        from training.faked import pytorchjob_manifest_op
        from utils.faked import (
            kubectl_apply_op,
            kubectl_wait_for_op,
            huggingface_importer_op,
            pvc_to_artifact_op,
            pvc_to_model_op
        )
        from utils import artifact_to_pvc_op
    else:
        from training import data_processing_op, pytorchjob_manifest_op
        from utils import (
            kubectl_apply_op,
            kubectl_wait_for_op,
            artifact_to_pvc_op,
            huggingface_importer_op,
            pvc_to_artifact_op,
            pvc_to_model_op
        )

    # Imports for MMLU, MT_BENCH stage
    # TODO: Add mock/fake components
    from utils import list_models_in_directory_op
    from eval.mmlu import run_mmlu_op, load_mmlu_results_op
    from eval.mt_bench import run_mt_bench_op, load_mt_bench_results_op

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
        storage_class_name: str = "ocs-external-storagecluster-ceph-rbd",
        base_model: str = "ibm-granite/granite-7b-base",
        # minimal subset of MMLU_TASKS
        mmlu_tasks_list: str = "mmlu_anatomy,mmlu_astronomy",
        model_dtype: str = "bfloat16",
        few_shots: int = 5,
        batch_size: int = 8,
        max_workers: str = "auto",
        merge_system_user_message: bool = False,
        device: str = None,
    ):

        # SDG stage
        git_clone_task = git_clone_op(
            repo_branch=repo_branch, repo_pr=repo_pr, repo_url=repo_url
        )

        sdg_task = sdg_op(
            num_instructions_to_generate=num_instructions_to_generate,
            taxonomy=git_clone_task.outputs["taxonomy"],
            repo_branch=repo_branch,
            repo_pr=repo_pr,
        )
        use_config_map_as_env(
            sdg_task, K8S_NAME, dict(endpoint="endpoint", model="model")
        )
        use_secret_as_env(sdg_task, K8S_NAME, {"api_key": "api_key"})


        # Training stage

        # We need to pass storage_class_name as "" to use the default StorageClass, if left empty, KFP uses "standard" StorageClass.
        # 'standard' !=  default StorageClass
        # https://github.com/kubeflow/pipelines/blob/1cded35cf5e93d8c8d32fefbddceb2eed8de9a0a/backend/src/v2/driver/driver.go#L1428-L1436
        # At least we made it a pipeline parameter
        model_pvc_task = CreatePVC(
            pvc_name_suffix="-model-cache",
            access_modes=["ReadWriteOnce"],
            size="50Gi",
            storage_class_name=storage_class_name,
        )
        model_to_artifact = huggingface_importer_op(repo_name=base_model)
        model_to_pvc_task = artifact_to_pvc_op(
            data=model_to_artifact.outputs["model"], pvc_path="/model"
        )
        model_to_pvc_task.set_caching_options(False)
        mount_pvc(
            task=model_to_pvc_task, pvc_name=model_pvc_task.output, mount_path="/model"
        )

        #Data processing 
        data_processing_task = data_processing_op(
            sdg = sdg_task.outputs["sdg"],
            model = model_to_artifact.outputs["model"]
        )

        sdg_input_pvc_task = CreatePVC(
            pvc_name_suffix="-sdg",
            access_modes=["ReadWriteOnce"],
            size="1Gi",
            storage_class_name=storage_class_name,
        )
        sdg_to_pvc_task = artifact_to_pvc_op(
            data=data_processing_task.outputs["processed_data"], pvc_path="/data"
        )
        sdg_to_pvc_task.set_caching_options(False)
        mount_pvc(
            task=sdg_to_pvc_task, pvc_name=sdg_input_pvc_task.output, mount_path="/data"
        )

        output_pvc_task = CreatePVC(
            pvc_name_suffix="-output",
            access_modes=["ReadWriteOnce"],
            size="50Gi",
            storage_class_name=storage_class_name,
        )

        # Using pvc_create_task.output as PyTorchJob name since dsl.PIPELINE_* global variables do not template/work in KFP v2
        # https://github.com/kubeflow/pipelines/issues/10453
        pytorchjob_manifest_task = pytorchjob_manifest_op(
            model_pvc_name=model_pvc_task.output,
            input_pvc_name=sdg_input_pvc_task.output,
            name_suffix=sdg_input_pvc_task.output,
            output_pvc_name=output_pvc_task.output,
        )
        pytorchjob_manifest_task.set_caching_options(False)

        kubectl_apply_task = kubectl_apply_op(
            manifest=pytorchjob_manifest_task.outputs["manifest"]
        )
        kubectl_apply_task.after(sdg_to_pvc_task, model_to_pvc_task)
        kubectl_apply_task.set_caching_options(False)

        kubectl_wait_task = kubectl_wait_for_op(
            condition="condition=Succeeded",
            kind="pytorchjobs",
            name=pytorchjob_manifest_task.outputs["name"],
        )
        kubectl_wait_task.after(kubectl_apply_task)
        kubectl_wait_task.set_caching_options(False)

        sdg_pvc_delete_task = DeletePVC(pvc_name=sdg_input_pvc_task.output)
        sdg_pvc_delete_task.after(kubectl_wait_task)

        model_pvc_delete_task = DeletePVC(pvc_name=model_pvc_task.output)
        model_pvc_delete_task.after(kubectl_wait_task)

        # MMLU Evaluation of models

        models_list_task = list_models_in_directory_op(
            models_folder="/output/model/model/hf_format",
        )

        models_list_task.after(kubectl_wait_task)

        mount_pvc(
            task=models_list_task, pvc_name=output_pvc_task.output, mount_path="/output/model"
        )

        run_mmlu_task = run_mmlu_op(
            models_list=models_list_task.output,
            models_path_prefix = "/output/model/model/hf_format",
            mmlu_tasks_list=mmlu_tasks_list,
            model_dtype=model_dtype,
            few_shots=few_shots,
            batch_size=batch_size,
            device=device,
        )

        mount_pvc(
            task=run_mmlu_task, pvc_name=output_pvc_task.output, mount_path="/output/model"
        )

        load_mmlu_results_task = load_mmlu_results_op(
            mmlu_output=run_mmlu_task.outputs['mmlu_output'],
        )

        run_mmlu_task.set_accelerator_type('nvidia.com/gpu')
        run_mmlu_task.set_accelerator_limit(1)

        #    Run training on MMLU best-model
        #    Run final eval on best scored mt_bench candidate
        #    For now, running mt_bench on same output models as training phase 1
        #    TODO: Another training phase, using the best-model from MMLU as base

        #    This is currently a duplicate of the above models_list_task,
        #    it's a placeholder for the listing of models from training phase 2
        phase_two_models_list_task = list_models_in_directory_op(
            models_folder="/output/model/model/hf_format",
        )

        phase_two_models_list_task.after(load_mmlu_results_task)

        mount_pvc(
            task=phase_two_models_list_task, pvc_name=output_pvc_task.output, mount_path="/output/model"
        )

        run_mt_bench_task = run_mt_bench_op(
            # TODO: make a second models_list_task from the 2nd phase of training
            models_list=phase_two_models_list_task.output,
            models_path_prefix = "/output/model/model/hf_format",
            max_workers = max_workers,
            merge_system_user_message = merge_system_user_message,
            device = device,
        )

        mount_pvc(
            task=run_mt_bench_task, pvc_name=output_pvc_task.output, mount_path="/output/model"
        )

        # For now run on same models from same training run as MMLU
        run_mt_bench_task.after(phase_two_models_list_task)

        run_mt_bench_task.set_accelerator_type('nvidia.com/gpu')
        run_mt_bench_task.set_accelerator_limit(1)


        use_config_map_as_env(
            run_mt_bench_task, JUDGE_CONFIG_MAP, dict(endpoint="JUDGE_ENDPOINT", model="JUDGE_NAME")
        )

        use_secret_as_env(run_mt_bench_task, JUDGE_SECRET, {"api_key": "JUDGE_API_KEY"})


        # Technically `output_model_task` and `output_data_task` can happen before evaluation,
        # however the PVC can only be mounted once, so, setting these to _after_ so the eval proceeds.
        output_model_task = pvc_to_artifact_op(
            pvc_path="/output/data",
            )
        #output_model_task.after(kubectl_wait_task)
        output_model_task.after(run_mt_bench_task)
        output_model_task.set_caching_options(False)

        mount_pvc(
            task=output_model_task, pvc_name=output_pvc_task.output, mount_path="/output/data"
        )

        output_data_task = pvc_to_model_op(
            pvc_path="/output/model",
            )
        #output_data_task.after(kubectl_wait_task)
        output_data_task.after(run_mt_bench_task)

        mount_pvc(
            task=output_data_task, pvc_name=output_pvc_task.output, mount_path="/output/model"
        )
        output_pvc_delete_task = DeletePVC(pvc_name=output_pvc_task.output)
        output_pvc_delete_task.after(output_model_task, output_data_task, run_mmlu_task, run_mt_bench_task)

        return

    return pipeline


@click.command()
@click.option(
    "--mock",
    type=click.Choice(MOCKED_STAGES, case_sensitive=False),
    help="Mock part of the pipeline",
    multiple=True,
    default=[],
)
def cli(mock):

    p = pipeline_wrapper(mock)

    with click.progressbar(length=1, label="Generating pipeline") as bar:
        compiler.Compiler().compile(p, "pipeline.yaml")
        bar.update(1)


if __name__ == "__main__":
    cli()
