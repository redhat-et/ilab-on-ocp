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
    use_config_map_as_env,
    use_secret_as_env,
    use_secret_as_volume,
)

TEACHER_CONFIG_MAP = "teacher-server"
TEACHER_SECRET = "teacher-server"
JUDGE_CONFIG_MAP = "judge-server"
JUDGE_SECRET = "judge-server"
MOCKED_STAGES = ["sdg", "train", "eval"]
PIPELINE_FILE_NAME = "pipeline.yaml"
IMPORTER_PIPELINE_FILE_NAME = "importer-pipeline.yaml"
STANDALONE_TEMPLATE_FILE_NAME = "standalone.tpl"
GENERATED_STANDALONE_FILE_NAME = "standalone.py"
DEFAULT_REPO_URL = "https://github.com/instructlab/taxonomy.git"


def ilab_pipeline_wrapper(mock: List[Literal[MOCKED_STAGES]]):
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
            pytorch_job_launcher_op,
            skills_processed_data_to_artifact_op,
        )
        from utils.faked import (
            model_to_pvc_op,
            pvc_to_model_op,
            pvc_to_mt_bench_op,
        )
    else:
        from training import (
            data_processing_op,
            knowledge_processed_data_to_artifact_op,
            pytorch_job_launcher_op,
            skills_processed_data_to_artifact_op,
        )
        from utils import (
            model_to_pvc_op,
            pvc_to_model_op,
            pvc_to_mt_bench_op,
        )

    # Imports for evaluation
    from eval.final import run_final_eval_op
    from eval.mt_bench import run_mt_bench_op

    @dsl.pipeline(
        display_name="InstructLab",
        name="instructlab",
        description="InstructLab pipeline",
    )
    def pipeline(
        # SDG phase
        sdg_repo_url: str = "https://github.com/instructlab/taxonomy.git",
        sdg_repo_branch: Optional[str] = None,
        sdg_repo_pr: Optional[
            int
        ] = None,  # FIXME: https://issues.redhat.com/browse/RHOAIRFE-467
        sdg_base_model: str = "s3://<BUCKET>/<PATH_TO_MODEL>",
        sdg_scale_factor: int = 30,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L125
        sdg_pipeline: str = "full",  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L122
        sdg_max_batch_len: int = 5000,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L334
        sdg_sample_size: float = 1.0,  # FIXME: Not present in default config. Not configurable upstream at this point, capability added via https://github.com/instructlab/sdg/pull/432
        # Training phase
        train_nproc_per_node: int = 2,  # FIXME: Not present in default config. Arbitrary value chosen to demonstrate multi-node multi-gpu capabilities. Needs proper reference architecture justification.
        train_nnodes: int = 2,  # FIXME: Not present in default config. Arbitrary value chosen to demonstrate multi-node multi-gpu capabilities. Needs proper reference architecture justification.
        train_num_epochs_phase_1: int = 7,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L364
        train_num_epochs_phase_2: int = 10,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L377
        train_effective_batch_size_phase_1: int = 128,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L357
        train_effective_batch_size_phase_2: int = 3840,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L371
        train_learning_rate_phase_1: float = 2e-05,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L360
        train_learning_rate_phase_2: float = 6e-06,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L374
        train_num_warmup_steps_phase_1: int = 1000,  # https://github.com/instructlab/training/blob/v0.6.1/src/instructlab/training/main_ds.py#L874
        train_num_warmup_steps_phase_2: int = 1000,  # https://github.com/instructlab/training/blob/v0.6.1/src/instructlab/training/main_ds.py#L874
        train_save_samples: int = 250000,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L393
        train_max_batch_len: int = 5000,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L334
        train_seed: int = 42,  # https://github.com/instructlab/training/blob/v0.6.1/src/instructlab/training/main_ds.py#L901
        # MT Bench
        mt_bench_max_workers: str = "auto",  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L74
        mt_bench_merge_system_user_message: bool = False,  # https://github.com/instructlab/instructlab/blob/v0.21.2/src/instructlab/model/evaluate.py#L474
        # Final evaluation
        final_eval_max_workers: str = "auto",  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L74
        final_eval_few_shots: int = 5,  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L56
        final_eval_batch_size: str = "auto",  # https://github.com/instructlab/instructlab/blob/v0.21.2/tests/testdata/default_config.yaml#L52
        final_eval_merge_system_user_message: bool = False,  # https://github.com/instructlab/instructlab/blob/v0.21.2/src/instructlab/model/evaluate.py#L474
        # Other options
        k8s_storage_class_name: str = "standard",  # FIXME: https://github.com/kubeflow/pipelines/issues/11396, https://issues.redhat.com/browse/RHOAIRFE-470
    ):
        """InstructLab pipeline

        Args:
            sdg_repo_url: SDG parameter. Points to a taxonomy git repository
            sdg_repo_branch: SDG parameter. Points to a branch within the taxonomy git repository. If set, has priority over sdg_repo_pr
            sdg_repo_pr: SDG parameter. Points to a pull request against the taxonomy git repository
            sdg_base_model: SDG parameter. LLM model used to generate the synthetic dataset
            sdg_scale_factor: SDG parameter. The total number of instructions to be generated.
            sdg_pipeline: SDG parameter. Data generation pipeline to use. Available: 'simple', 'full', or a valid path to a directory of pipeline workflow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.
            sdg_max_batch_len: SDG parameter. Maximum tokens per gpu for each batch that will be handled in a single step.
            sdg_sample_size: SDG parameter. Represents the sdg skills recipe sampling size as percentage in decimal form.

            train_nproc_per_node: Training parameter. Number of GPUs per each node/worker to use for training.
            train_nnodes: Training parameter. Number of nodes/workers to train on.
            train_num_epochs_phase_1: Training parameter for in Phase 1. Number of epochs to run training.
            train_num_epochs_phase_2: Training parameter for in Phase 2. Number of epochs to run training.
            train_effective_batch_size_phase_1: Training parameter for in Phase 1. The number of samples in a batch that the model should see before its parameters are updated.
            train_effective_batch_size_phase_2: Training parameter for in Phase 2. The number of samples in a batch that the model should see before its parameters are updated.
            train_learning_rate_phase_1: Training parameter for in Phase 1. How fast we optimize the weights during gradient descent. Higher values may lead to unstable learning performance. It's generally recommended to have a low learning rate with a high effective batch size.
            train_learning_rate_phase_2: Training parameter for in Phase 2. How fast we optimize the weights during gradient descent. Higher values may lead to unstable learning performance. It's generally recommended to have a low learning rate with a high effective batch size.
            train_num_warmup_steps_phase_1: Training parameter for in Phase 1. The number of steps a model should go through before reaching the full learning rate. We start at 0 and linearly climb up to train_learning_rate.
            train_num_warmup_steps_phase_2: Training parameter for in Phase 2. The number of steps a model should go through before reaching the full learning rate. We start at 0 and linearly climb up to train_learning_rate.
            train_save_samples: Training parameter. Number of samples the model should see before saving a checkpoint.
            train_max_batch_len: Training parameter. Maximum tokens per gpu for each batch that will be handled in a single step.
            train_seed: Training parameter. Random seed for initializing training.

            mt_bench_max_workers: MT Bench parameter. Number of workers to use for evaluation with mt_bench or mt_bench_branch. Must be a positive integer or 'auto'.
            mt_bench_merge_system_user_message: MT Bench parameter. Boolean indicating whether to merge system and user messages (required for Mistral based judges)

            final_eval_max_workers: Final model evaluation parameter for MT Bench Branch. Number of workers to use for evaluation with mt_bench or mt_bench_branch. Must be a positive integer or 'auto'.
            final_eval_few_shots: Final model evaluation parameter for MMLU. Number of question-answer pairs provided in the context preceding the question used for evaluation.
            final_eval_batch_size: Final model evaluation parameter for MMLU. Batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory.
            final_eval_merge_system_user_message: Final model evaluation parameter for MT Bench Branch. Boolean indicating whether to merge system and user messages (required for Mistral based judges)

            k8s_storage_class_name: A Kubernetes StorageClass name for persistent volumes. Selected StorageClass must support RWX PersistentVolumes.
        """

        # SDG stage
        sdg_input_pvc_task = CreatePVC(
            pvc_name_suffix="-sdg",
            access_modes=["ReadWriteMany"],
            size="10Gi",
            storage_class_name=k8s_storage_class_name,
        )
        git_clone_task = git_clone_op(
            repo_branch=sdg_repo_branch,
            repo_pr=sdg_repo_pr if sdg_repo_pr and sdg_repo_pr > 0 else None,
            repo_url=sdg_repo_url,
        )
        mount_pvc(
            task=git_clone_task,
            pvc_name=sdg_input_pvc_task.output,
            mount_path="/data",
        )
        git_clone_task.set_caching_options(False)

        sdg_task = sdg_op(
            num_instructions_to_generate=sdg_scale_factor,
            pipeline=sdg_pipeline,
            repo_branch=sdg_repo_branch,
            repo_pr=sdg_repo_pr,
            sdg_sampling_size=sdg_sample_size,
        )
        sdg_task.set_env_variable("HOME", "/tmp")
        sdg_task.set_env_variable("HF_HOME", "/tmp")
        use_config_map_as_env(
            sdg_task, TEACHER_CONFIG_MAP, dict(endpoint="endpoint", model="model")
        )
        use_secret_as_env(sdg_task, TEACHER_SECRET, {"api_key": "api_key"})
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

        # uncomment if updating image with same tag
        # set_image_pull_policy(sdg_task, "Always")

        # Training stage
        model_source_s3_task = dsl.importer(
            artifact_uri=sdg_base_model, artifact_class=dsl.Model
        )

        model_pvc_task = CreatePVC(
            pvc_name_suffix="-model-cache",
            access_modes=["ReadWriteMany"],
            size="100Gi",
            storage_class_name=k8s_storage_class_name,
        )

        model_to_pvc_task = model_to_pvc_op(model=model_source_s3_task.output)
        model_to_pvc_task.set_caching_options(False)
        mount_pvc(
            task=model_to_pvc_task, pvc_name=model_pvc_task.output, mount_path="/model"
        )

        # Data processing
        data_processing_task = data_processing_op(max_batch_len=sdg_max_batch_len)
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
        data_processing_task.set_env_variable("XDG_CACHE_HOME", "/tmp")

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
            storage_class_name=k8s_storage_class_name,
        )

        # Training 1
        # Using pvc_create_task.output as PyTorchJob name since dsl.PIPELINE_* global variables do not template/work in KFP v2
        # https://github.com/kubeflow/pipelines/issues/10453
        training_phase_1 = pytorch_job_launcher_op(
            model_pvc_name=model_pvc_task.output,
            input_pvc_name=sdg_input_pvc_task.output,
            name_suffix=sdg_input_pvc_task.output,
            output_pvc_name=output_pvc_task.output,
            phase_num=1,
            nproc_per_node=train_nproc_per_node,
            nnodes=train_nnodes,
            num_epochs=train_num_epochs_phase_1,
            effective_batch_size=train_effective_batch_size_phase_1,
            learning_rate=train_learning_rate_phase_1,
            num_warmup_steps=train_num_warmup_steps_phase_1,
            save_samples=train_save_samples,
            max_batch_len=train_max_batch_len,
            seed=train_seed,
        )
        training_phase_1.after(data_processing_task, model_to_pvc_task)
        training_phase_1.set_caching_options(False)

        #### Train 2
        training_phase_2 = pytorch_job_launcher_op(
            model_pvc_name=model_pvc_task.output,
            input_pvc_name=sdg_input_pvc_task.output,
            name_suffix=sdg_input_pvc_task.output,
            output_pvc_name=output_pvc_task.output,
            phase_num=2,
            nproc_per_node=train_nproc_per_node,
            nnodes=train_nnodes,
            num_epochs=train_num_epochs_phase_2,
            effective_batch_size=train_effective_batch_size_phase_2,
            learning_rate=train_learning_rate_phase_2,
            num_warmup_steps=train_num_warmup_steps_phase_2,
            save_samples=train_save_samples,
            max_batch_len=train_max_batch_len,
            seed=train_seed,
        )

        training_phase_2.set_caching_options(False)
        training_phase_2.after(training_phase_1)

        mount_pvc(
            task=training_phase_2,
            pvc_name=output_pvc_task.output,
            mount_path="/output",
        )

        # MT_Bench Evaluation of models

        run_mt_bench_task = run_mt_bench_op(
            models_folder="/output/phase_2/model/hf_format",
            max_workers=mt_bench_max_workers,
            merge_system_user_message=mt_bench_merge_system_user_message,
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
        run_mt_bench_task.after(training_phase_2)
        use_config_map_as_env(
            run_mt_bench_task,
            JUDGE_CONFIG_MAP,
            dict(endpoint="JUDGE_ENDPOINT", model="JUDGE_NAME"),
        )
        use_secret_as_env(run_mt_bench_task, JUDGE_SECRET, {"api_key": "JUDGE_API_KEY"})

        # uncomment if updating image with same tag
        # set_image_pull_policy(run_mt_bench_task, "Always")

        final_eval_task = run_final_eval_op(
            candidate_model="/output/phase_2/model/hf_format/candidate_model",
            # TODO: DO we need both candidate_branch and base_branch
            base_branch=sdg_repo_branch,
            candidate_branch=sdg_repo_branch,
            base_model_dir="/model/",
            max_workers=final_eval_max_workers,
            merge_system_user_message=final_eval_merge_system_user_message,
            few_shots=final_eval_few_shots,
            batch_size=final_eval_batch_size,
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
        output_model_task.set_caching_options(False)
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


def import_base_model_pipeline_wrapper(mock: List[Literal[MOCKED_STAGES]]):
    from utils import ilab_importer_op

    @dsl.pipeline(
        display_name="InstructLab - base model importer",
        name="instructlab-base-importer",
        description="Helper pipeline to the InstructLab pipeline which allows users to seed/import a new base model",
    )
    def pipeline(
        # hf_token_secret: str = "", # FIXME: Don't use hardcoded secret/configmap names once fixed upstream: https://github.com/kubeflow/pipelines/issues/11395
        # oci_pull_secret: str = "", # FIXME: Don't use hardcoded secret/configmap names once fixed upstream: https://github.com/kubeflow/pipelines/issues/11395
        repository: str = "docker://registry.redhat.io/rhelai1/granite-7b-starter",
        release: str = "latest",
    ):
        """InstructLab - base model importer.

        Args:
            repository: Hugging Face or OCI repository of the model to download. OCI repository must have a docker:// prefix
            release: The revision of the model to download - e.g. a branch, tag, or commit hash for Hugging Face repositories and tag or commit hash for OCI repositories.
            hf_token_secret: Name of existing Kubernetes secret which contains HF_TOKEN value for Hugging Face repositories. Mandatory for all repositories besides those which belong to the "instructlab" organization.
            oci_pull_secret: Name of existing Kubernetes secret of .dockerconfigjson type for OCI repository authentication.
        """
        importer_task = ilab_importer_op(repository=repository, release=release)

        # FIXME: Don't use hardcoded secret/configmap names once fixed upstream: https://github.com/kubeflow/pipelines/issues/11395
        # FIXME: Make env variables optional once implemented upstream: https://github.com/kubeflow/pipelines/issues/11401
        # This pipeline is currently unusable outside of ocp-beta-test.nerc.mghpcc.org cluster, `ilab` namespace due to the hardcoded names...
        use_secret_as_env(
            importer_task, "hugging-face-token", dict(HF_TOKEN="HF_TOKEN")
        )
        importer_task.set_env_variable(
            "REGISTRY_AUTH_FILE", "/mnt/containers/.dockerconfigjson"
        )
        use_secret_as_volume(
            importer_task, "7033380-ilab-pull-secret", mount_path="/mnt/containers"
        )

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
    ilab_pipeline = ilab_pipeline_wrapper(mock)
    import_base_model_pipeline = import_base_model_pipeline_wrapper(mock)

    pipelines = [
        (ilab_pipeline, PIPELINE_FILE_NAME),
        (import_base_model_pipeline, IMPORTER_PIPELINE_FILE_NAME),
    ]

    with click.progressbar(pipelines, label="Generating pipeline") as bar:
        for pipeline_func, pipeline_file in bar:
            compiler.Compiler().compile(pipeline_func, pipeline_file)


@cli.command(name="run")
@click.option(
    "--mock",
    type=click.Choice(MOCKED_STAGES, case_sensitive=False),
    help="Mock part of the pipeline",
    multiple=True,
    default=[],
)
@click.option("-e", "--experiment", help="Set KFP experiment name.")
@click.option("-r", "--run", "run_name", help="Set KFP run name.")
@click.option(
    "-p",
    "--param",
    help="Override default parameters in KEY=VALUE format. Default parameters are suitable for dev cluster - the MOC cluster, `ilab` namespace.",
    multiple=True,
)
def run(mock, experiment, run_name, param):
    """
    Run the pipeline immediately against current kubernetes context (cluster and namespace).

    Command sets expected dev-cluster friendly default values when submitting.
    """
    from utils.kfp_client import get_kfp_client

    client = get_kfp_client()

    dev_arguments = {
        "k8s_storage_class_name": "nfs-csi",
        "sdg_base_model": "s3://ilab-pipeline-b1d4c2b1-ab00-4e7f-b985-697bda3df385/instructlab-base-importer/648f36d0-e3f0-43b8-8adb-530576beb675/ilab-importer-op/model/granite-7b-starter",
        "train_num_epochs_phase_1": 2,
        "train_num_epochs_phase_2": 2,
        "train_num_warmup_steps_phase_1": 100,
        "train_num_warmup_steps_phase_2": 100,
        "train_learning_rate_phase_1": 1e-4,
        "train_learning_rate_phase_2": 1e-4,
        "sdg_sample_size": 0.0002,
    }

    try:
        parsed_params = dict(item.split("=") for item in param)
    except ValueError as e:
        raise click.BadOptionUsage(
            "param", "Parameters are required to be passed in KEY=VALUE format"
        ) from e

    arguments = {**dev_arguments, **parsed_params}
    client.create_run_from_pipeline_func(
        pipeline_func=ilab_pipeline_wrapper(mock),
        experiment_name=experiment,
        run_name=run_name,
        arguments=arguments,
    )


@cli.command(name="gen-standalone")
def gen_standalone():
    """
    Generates a standalone script that mimics the behavior of the pipeline.

    This function should be used when Kubeflow Pipelines are not available. It will generate a
    script that replicates the pipeline's functionality.

    Example usage: ''' $ python pipeline.py gen-standalone '''
    """
    from os import chmod, path

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
        "exec-sdg-op": 'sdg_op(num_instructions_to_generate={num_instructions_to_generate}, pipeline="{sdg_pipeline}", repo_branch="{exec_git_clone_op_repo_branch or ""}", repo_pr={exec_git_clone_op_repo_pr or 0}, taxonomy_path="{TAXONOMY_DATA_PATH}", sdg_path="{DATA_PVC_SDG_PATH}", sdg_sampling_size={sdg_sampling_size})',
        "exec-git-clone-op": {},
        "exec-run-mt-bench-op": 'run_mt_bench_op(best_score_file="{MT_BENCH_SCORES_PATH}",output_path="{MT_BENCH_OUTPUT_PATH}",models_folder="{CANDIDATE_MODEL_PATH_PREFIX}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE})',
        "exec-run-final-eval-op": 'run_final_eval_op(mmlu_branch_output="{MMLU_BRANCH_SCORES_PATH}", mt_bench_branch_output="{MT_BENCH_BRANCH_SCORES_PATH}", candidate_model="{CANDIDATE_MODEL_PATH}", taxonomy_path="{TAXONOMY_PATH}", sdg_path="{DATA_PVC_SDG_PATH}", base_branch="", candidate_branch="", base_model_dir="{DATA_PVC_MODEL_PATH}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE}, few_shots={FEW_SHOTS}, batch_size="{BATCH_SIZE}")',
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
    # Make the rendered file executable
    chmod(standalone_script_path, 0o755)

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
