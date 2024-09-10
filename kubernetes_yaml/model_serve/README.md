# Model Serving
To serve a model a data connection to an object store is required. Values must be base64 encoded when using the s3.yaml file that is included.

NOTE: When using NooBaa no Region is required.

To encyrpt your object storage credentials perform the following against all required fields.

```
echo "AWSACCESSKEYSECETVALUE" | base64
```

Once all values that relate to the object storage are defined create the secret and define the specific namespace to deploy the secret.

```
oc create -f s3.yaml -n $NAMESPACE
```

Once the secret has been defined the model serve can use the secret.

Modify the following values in your InferenceService yaml to define the path to the model and the name of the model that will be run.

```
      runtime: mistral-7b-instruct-v02
      storage:
        key: object-store
        path: models/Mistral-7B-Instruct-v0.2
```


Create the InferenceService in the same namespace in which the secret was defined.
```
oc create -f mistral-7b-instruct-v02.yaml -n $NAMESPACE
```
