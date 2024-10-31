## Mixtral serving
Mixtral is required as part of the Instructlab process in various places. The following will describe how to provide Mixtral and the LORA adapters.

### Secret
Because we neet to run oras inside of a container to download the various artifacts we must provide a .dockerconfigjson to the Kubernetes job with authentication back to registry.redhat.io.
It is suggested to use a Service account. https://access.redhat.com/terms-based-registry/accounts is the location to create a service account.

Create a secret based off of the service account.

secret.yaml

```
apiVersion: v1
kind: Secret
metadata:
  name: 7033380-ilab-pull-secret
data:
  .dockerconfigjson: sadfassdfsadfasdfasdfasdfasdfasdfasdf=
type: kubernetes.io/dockerconfigjson
```

Create the secret

```
oc create -f secret.yaml
```

### Kubernetes Job
Depending on the name of your secret the file `../mixtral_pull/pull_kube_job.yaml` will need to be modified.

```
...redacted...
      - name: docker-config
        secret:
          secretName: 7033380-ilab-pull-secret
...redacted...
```

With the secretName now reflecting your secret the job can be launched.

```
kubectl create -f ./mixtral_pull
```

This will create 3 different containers downloading various things using oras.

## Knative
The `knative-serving` configMap may need to be updated to ensure that pvcs can be used. It appears in newer versions of knative this is resolved. Ensure the following values are set in the knative-serving configMap.

```
  kubernetes.podspec-persistent-volume-claim: enabled
  kubernetes.podspec-persistent-volume-write: enabled
```

### Mixtral serving
This will make no sense but it is the only way discovered so far to ensure that a token is generated to work with the model. Using the RHODS model serving UI define a model to be served named mixtral. Ensure external access and token are selected as the TOKEN is the piece not yet discovered when using just the CLI.

We will now use the PVC from the previous step to serve the model and replace the runtime defined in the UI.

```
kubectl apply -f ./mixtral_serve/runtime.yaml
```

Modify the inference service and copy the entire spec field from ./mixtral_serve/inference.yaml

```
oc edit inferenceservice mixtral
```

```
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      args:
      - --dtype=bfloat16
      - --tensor-parallel-size=4
      - --enable-lora
      - --max-lora-rank=64
      - --lora-dtype=bfloat16
      - --fully-sharded-loras
      - --lora-modules
      - skill-classifier-v3-clm=/mnt/skills
      - text-classifier-knowledge-v3-clm=/mnt/knowledge
      modelFormat:
        name: vLLM
      name: ""
      resources:
        limits:
          cpu: "4"
          memory: 40Gi
          nvidia.com/gpu: "4"
        requests:
          cpu: "4"
          memory: 40Gi
          nvidia.com/gpu: "4"
      runtime: mixtral
    tolerations:
    - effect: NoSchedule
      key: nvidia.com/gpu
      operator: Exists
```


Follow the log of the kserve-container and wait for the the following log entry

```
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Testing
To interact with the model grab the inference endpoint from the RHOAI UI and the token.

```
oc get secret -o yaml default-name-mixtral-sa | grep token: | awk -F: '{print $2}' | tr -d ' ' | base64 -d
```

Export that value as a variable named TOKEN

```
export TOKEN=BLOBOFLETTERSANDNUMBERS
```

Using curl you can ensure that the model is accepting connections
```
curl -X POST "https://mixtral-labels.apps.hulk.octo-emerging.redhataicoe.com/v1/completions" -H  "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" -d '{"model": "mixtral", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0 }'


{"id":"cmpl-ecd5bd72a947438b805e25134bbdf636","object":"text_completion","created":1730231625,"model":"mixtral","choices":[{"index":0,"text":" city that is known for its steep","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7}}%
```
