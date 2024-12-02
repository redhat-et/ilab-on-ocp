# type: ignore
import warnings

from kfp import Client
from kubernetes.client import CustomObjectsApi
from kubernetes.client.configuration import Configuration
from kubernetes.client.exceptions import ApiException
from kubernetes.config import list_kube_config_contexts
from kubernetes.config.config_exception import ConfigException
from kubernetes.config.kube_config import load_kube_config


def get_kfp_client():
    config = Configuration()
    try:
        load_kube_config(client_configuration=config)
        token = config.api_key["authorization"].split(" ")[-1]
    except (KeyError, ConfigException) as e:
        raise ApiException(
            401, "Unauthorized, try running `oc login` command first"
        ) from e
    Configuration.set_default(config)

    _, active_context = list_kube_config_contexts()
    namespace = active_context["context"]["namespace"]

    dspas = CustomObjectsApi().list_namespaced_custom_object(
        "datasciencepipelinesapplications.opendatahub.io",
        "v1alpha1",
        namespace,
        "datasciencepipelinesapplications",
    )

    try:
        dspa = dspas["items"][0]
    except IndexError as e:
        raise ApiException(404, "DataSciencePipelines resource not found") from e

    try:
        if dspa["spec"]["dspVersion"] != "v2":
            raise KeyError
    except KeyError as e:
        raise EnvironmentError(
            "Installed version of Kubeflow Pipelines does not meet minimal version criteria. Use KFPv2 please."
        ) from e

    try:
        host = dspa["status"]["components"]["apiServer"]["externalUrl"]
    except KeyError as e:
        raise ApiException(
            409,
            "DataSciencePipelines resource is not ready. Check for .status.components.apiServer",
        ) from e

    with warnings.catch_warnings(action="ignore"):
        return Client(existing_token=token, host=host)
