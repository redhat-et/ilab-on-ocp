# InstructLab Standalone Tool Integration Tests

### Prerequisites

* Admin access to an OpenShift cluster

* Installed OpenDataHub or RHOAI, enabled all Distributed Workload and Serving components

* OpenShift contains StorageClass with ReadWriteMany access mode

* OpenAI compliant Judge model deployed and served on an endpoint reachable from OpenShift

* Installed Go 1.21

### Sample Judge Model Deployment

* The sample manifest for deploying judge-model can be found here - `tests/standalone/e2e/resources/judge_model_deployment.yaml`

## Required environment variables

### Environment variables to download SDG and upload trained model

* `AWS_DEFAULT_ENDPOINT` - Storage bucket default endpoint
* `AWS_ACCESS_KEY_ID` - Storage bucket access key
* `AWS_SECRET_ACCESS_KEY` - Storage bucket secret key
* `AWS_STORAGE_BUCKET` - Storage bucket name
* `SDG_OBJECT_STORE_DATA_KEY` - Path in the storage bucket where SDG bundle is located
* `SDG_SERVING_MODEL_API_KEY` - Teacher model api key
* `SDG_NAME` - Teacher model name
* `SDG_ENDPOINT` - Teacher model endpoint
* `SDG_CA_CERT` - Name of the configmap that contains the CA Cert bundle for the Teacher model
* `SDG_CA_CERT_CM_KEY` - The Configmap key that contains the CA cert bundle specified via `SDG_CA_CERT`
* `SDG_CA_CERT_FROM_OPENSHIFT` - Set to `true` if the CA Cert can be fetched from Openshift. This will automatically utilize `kube-root-ca.crt` that is provisioned within every namespace. Take precedence over `SDG_CA_CERT` and `SDG_CA_CERT_CM_KEY`.
* `SDG_SAMPLING_SIZE` (Optional) - Adjusts the sampling used for the skills data recipe during SDG phase, should be a percentage in decimal form. Default is `0.0002`.

### Environment variables for connection to Judge model

* `JUDGE_ENDPOINT` - Endpoint where the Judge model is deployed to (it should end with `/v1`)
* `JUDGE_NAME` - Name of the Judge model
* `JUDGE_API_KEY` - API key needed to access the Judge model
* `JUDGE_CA_CERT_FROM_OPENSHIFT` (Optional) - If Judge model is deployed in the same OpenShift instance and the OpenShift certificate is insecure then set this env variable to `true`. It will indicate to the test to set OpenShift CA certificate as trusted certificate.

### Misc environment variables

* `TEST_NAMESPACE` (Optional) - Specify test namespace which should be used to run the tests
* `TEST_ILAB_STORAGE_CLASS_NAME` (Optional) - Specify name of StorageClass which supports ReadWriteMany access mode. If not specified then test assumes StorageClass `nfs-csi` to exist.
* `RHELAI_WORKBENCH_IMAGE` (Optional) - Specify Workbench image to be used to run Standalone tool. If not specified then test uses Workbench image `quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8`.
Provided image should contain `click==8.1.7` and `kubernetes==26.1.0` packages.
* `TEST_RUN_TIMEOUT` (Optional) - Specify the timeout for the test. Requires a [parsed duration string](https://pkg.go.dev/time#ParseDuration). Default is 10 hours.

## Running Tests

Execute tests like standard Go unit tests.

```bash
go test -run TestInstructlabTrainingOnRhoai -v -timeout 180m ./standalone/e2e/
```
