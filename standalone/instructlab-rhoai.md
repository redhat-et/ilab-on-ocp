# Distributed InstructLab Training on RHOAI

This file documents an experimental install of InstructLab on Red Hat OpenShift AI.

## Pre-requisites

* An OpenShift cluster with
    * Sufficient GPUs available for training.
        * 4x NVIDIA A100 GPUs
    * Red Hat - Authorino installed
    * Red Hat Openshift Serverless installed
* An OpenShift AI installation, with the Training Operator and kserve components set to `Managed`
    * A data science project/namespace
* A [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) that supports dynamic provisioning with [ReadWriteMany](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes) access mode (see step 3 below).
* An AWS S3 compatible object store. Alternative object storage solutions such as Ceph, Nooba and MinIO are also compatible.
* SDG data generated that has been uploaded to an object store (see step 1 below).
* The `oc` command line tool binary installed locally to create and manage kubernetes resources.


## Steps

Before running the training and evaluation steps we must complete the following:

* Step 1 - Prepare data and push to object store
* Step 2 - Create judge model server
* Step 3 (Optional) - Setup NFS StorageClass
* Step 4 - Run InstructLab distributed training


### Step 1 - Prepare data and push to object store

* Create a tarball with the data (SDG-data), [model](https://huggingface.co/ibm-granite/granite-7b-base/tree/main) and [taxonomy](https://github.com/instructlab/taxonomy) and push them to your object store.

  ```
  $ mkdir -p s3-data/{data,model,taxonomy}
  ```

* To generate SDG data, add your required skill in taxonomy directory tree where it suits
  ```
  $ cd s3-data
  $ ilab config init
  $ ilab data generate --taxonomy-base=empty
  ```

* Download ilab model repository in s3-data model direct
  ```
  $ ilab model download --repository ibm-granite/granite-7b-base
  $ cp -r <path-to-model-downloaded-dir>/ibm-granite/granite-7b-base/* s3-data/model
  ```

* Clone taxonomy repository (this could be any taxonomy repo and any branches in the given repo)
  ```
  $ cd s3-data
  $ git clone https://github.com/instructlab/taxonomy.git taxonomy
  ```

* Generate tar archive
  ```
  $ cd s3-data
  $ tar -czvf rhelai.tar.gz *
  ```

* Upload the created tar archive to your object store.


    `Note` : The standalone.py script will check for the format required using a sample pattern, for example -


      ```
      model/config.json
      model/tokenizer.json
      model/tokenizer_config.json
      model/*.safetensors
      data/skills_recipe_*.yaml
      data/knowledge_recipe_*.yaml
      data/skills_train_*.jsonl
      data/knowledge_train_*.jsonl
      taxonomy/knowledge
      taxonomy/foundational_skills
      ```


### Step 2 - Create Judge model server

The judge model is used for model evaluation.

* Create a service account to be used for token authentication

    ```
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: <model-server-service-account-name>
      Namespace: <data-science-project-name/namespace>
    ```

* Upload [Prometheus-eval](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0) (Judge-Model) to the same object storage as before .

    * This [script](https://github.com/opendatahub-io/ilab-on-ocp/blob/main/kubernetes_yaml/model_downloader/container_file/download_hf_model.py) can be used for uploading the model to object storage.
    * Update the MODEL and MODEL_PATH parameters if using a different model to that in the example.

  Example:

    ```
    export MODEL=prometheus-eval/prometheus-8x7b-v2.0 \
    S3_ENDPOINT=<s3-bucket-endpoint> \
    AWS_ACCESS_KEY=<s3-bucket-access-key> \
    AWS_SECRET_KEY=<s3-bucket-secret-key> \
    AWS_BUCKET_NAME=<bucket-name> \
    S3_FOLDER=prometheus-eval \
    MODEL_PATH=model \
    HF_TOKEN=<hugging-face-auth-token>
    ```

* Navigate to the OpenShift AI dashboard
    * Choose Data Science Projects from the left hand menu and choose your data science project/namespace.
    * Choose the data connections tab, and click on the Add data connection button. Enter the details of your S3 bucket (object store) and click Add data connection.

    `Note`: Before following the next step - Ensure that the CapabilityServiceMeshAuthorization status is True in DSCinitialization resource.

    * Create a model server instance
        * Navigate to Data Science Projects and then the Models tab
        * On the right hand side select ‘Deploy model’ under Single-model serving platform
        * Under Serving runtime choose the serving runtime `vLLM Serving Runtime for Kserve`.
        * Check the `Make deployed models available through an external route` box.
        * Under token authentication check the `Require token authentication` box, select the service account that we have created above.
        * Choose the existing data connection created earlier.
        * Click deploy.

* Create a secret containing the judge model serving details

    ```
    apiVersion: v1
    kind: Secret
    metadata:
      name: judge-serving-details
    type: Opaque
    stringData:
      JUDGE_NAME:               # Name of the judge model or deployment
      JUDGE_ENDPOINT:           # Model serving endpoint, Sample format - `https://<deployed-model-server-endpoint>/v1`
      JUDGE_API_KEY:            # Deployed model-server auth token
      JUDGE_CA_CERT:            # Configmap containing CA cert for the judge model (optional - required if using custom CA cert), Example - `kube-root-ca.crt`
      JUDGE_CA_CERT_CM_KEY:     # Name of key inside configmap (optional - required if using custom CA cert), Example - `ca.crt`
    ```

* If using a custom CA certificate you must provide the relevant data in a ConfigMap. The config map name and key are then provided as a parameter to the standalone.py script as well as in the `judge-serving-details` secret above.

### Step 3 (Optional) - Setup NFS StorageClass

> [!CAUTION]
> The image provided here is for test purposes only.
> Users must provide a production ready storageclass with ReadWriteMany capability.

  This step is needed when the cluster doesn't have a storage provisioner capable of provisioning PersistentVolumeClaim with ReadWriteMany capability.

  * Installing the NFS CSI driver
    ```
    $ curl -skSL https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/v4.9.0/deploy/install-driver.sh | bash -s v4.9.0 --`
    ```

  * For deploying an in-cluster NFS server

    ```
    oc new-project nfs
    oc apply -f ./nfs-server-deployment.yaml
    ```

    `Note`:  Check the root PersistentVolumeclaim that'll be created and the requested storage.

  * For creating NFS storage-class
    ```
    oc apply -f ./nfs-storage-class.yaml
    ```

  * This will create the required resources in the cluster, including the required StorageClass.


### Step 4 - Run InstructLab distributed training

Now we can continue to set up the required resources in our cluster:

  The following resources will be created:

  1. ConfigMap
  2. Secret
  3. ClusterRole
  4. ClusterRoleBinding
  5. Pod

 * create a configMap that contains the [standalone.py script](standalone.py)

      ```
      $ curl -OL https://raw.githubusercontent.com/opendatahub-io/ilab-on-ocp/refs/heads/phase-1/standalone/standalone.py

      $ oc create configmap -n <data-science-project-name/namespace> standalone-script --from-file ./standalone.py
      ```

 * Create a secret resource that contains the credentials for your Object Storage (AWS S3 Bucket)

    `Note`: encode these credentials in Base-64 form and then add it to the secret yaml file below:

      ```
      apiVersion: v1
      kind: Secret
      metadata:
        name: sdg-object-store-credentials
      type: Opaque
      data:
        bucket:                     # The object store bucket containing SDG+Model+Taxonomy data. (Name of S3 bucket)
        access_key:                 # The object store access key (AWS Access key ID)
        secret_key:                 # The object store secret key (AWS Secret Access Key)
        data_key:                   # The name of the tarball that contains SDG data.
        endpoint:                   # The object store endpoint
        region:                     # The region for the object store.
        verify_tls:                 # Verify TLS for the object store.
      ```
      Apply the yaml file to the cluster

 * Create a ServiceAccount, ClusterRole and ClusterRoleBinding

      Provide access to the service account running the standalone.py script for accessing and manipulating related resources.

      ```
      apiVersion: rbac.authorization.k8s.io/v1
      kind: ClusterRole
      metadata:
        namespace: <data-science-project-name/namespace>
        name: secret-access-role
      rules:
        - apiGroups: [""]
          resources: ["pods", "pods/log", "configmaps", "persistentvolumeclaims", "secrets","events"]
          verbs: ["get", "list", "watch", "create", "update", "delete"]

        - apiGroups: ["batch"]
          resources: ["jobs"]
          verbs: ["get", "list", "create", "watch"]

        - apiGroups: ["kubeflow.org"]
          resources: ["pytorchjobs"]
          verbs: ["get", "list", "create", "watch"]
      ---
      apiVersion: rbac.authorization.k8s.io/v1
      kind: ClusterRoleBinding
      metadata:
        name: secret-access-binding
      subjects:
      - kind: ServiceAccount
        name: <workbench-service-account-name> # created above in Step-2
        namespace: <data-science-project-name/namespace>
      roleRef:
        kind: ClusterRole
        name: secret-access-role
        apiGroup: rbac.authorization.k8s.io
      ```
      Apply the yaml to the cluster.

      These are the required [RBAC configuration](https://github.com/opendatahub-io/ilab-on-ocp/tree/main/standalone#rbac-requirements-when-running-in-a-kubernetes-job)s which we are applying on the ServiceAccount.



* Create the workbench pod and run the standalone.py script

  * In this step the standalone.py script will be utilised. The script runs a pytorchjob that utilises Fully Sharded Data Parallel (FSDP), sharing the load across available resources (GPUs).
  * Prepare the pod yaml like below including this [workbench image](https://quay.io/repository/opendatahub/workbench-images/manifest/sha256:7f26f5f2bec4184af15acd95f29b3450526c5c28c386b6cb694fbe82d71d0b41).
  * This pod will access and run the standalone.py script from the configmap that we created earlier.
  * `Note` that the value for `judge-serving-model-api-key`should match the jwt auth token generated when setting up the judge serving model (step 2 above).


  ```
  apiVersion: v1
  kind: Pod
  metadata:
    name: ilab-pod
    namespace:  &lt;data-science-project-name/namespace>
  spec:
    serviceAccountName: <service-account-name>      # created above in Step-2
    containers: \
    - name: workbench-container \
      image: quay.io/opendatahub/workbench-images@sha256:7f26f5f2bec4184af15acd95f29b3450526c5c28c386b6cb694fbe82d71d0b41
      env:
      - name: SDG_OBJECT_STORE_ENDPOINT
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: endpoint
      - name: SDG_OBJECT_STORE_BUCKET
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: bucket
      - name: SDG_OBJECT_STORE_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: access_key
      - name: SDG_OBJECT_STORE_SECRET_KEY
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: secret_key
      - name: SDG_OBJECT_STORE_REGION
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: region
      - name: SDG_OBJECT_STORE_DATA_KEY
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: data_key
      - name: SDG_OBJECT_STORE_VERIFY_TLS
        valueFrom:
          secretKeyRef:
            name: sdg-object-store-credentials
            key: verify_tls
      volumeMounts:
      - name: script-volume
        mountPath: /home/standalone.py
        subPath: standalone.py
      command: ["python3", "/home/standalone.py", "run",
                "--namespace", "<data-science-project-name/namespace>",
                "--judge-serving-model-secret", "<created-judge-model-details-k8s-secret>",
                "--sdg-object-store-secret", "<created-object-store-credentials-secret-name>",
                "--storage-class", "<created-storage-class-name>",
                "--nproc-per-node" , '1',
                "--force-pull"]
    volumes:
    - name: script-volume
      configMap:
        name: standalone-script
  ```
  Apply the yaml to the cluster.
