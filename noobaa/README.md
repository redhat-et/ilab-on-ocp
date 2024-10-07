# Noobaa tuning
Due to the need to upload large files the following tuning parameters must be applied.

Patch the storagecluster for this example the name of the storage cluster is `ocs-external-storagecluster`
```
oc patch -n openshift-storage storagecluster ocs-external-storagecluster \\n    --type merge \\n    --patch '{"spec": {"resources": {"noobaa-core": {"limits": {"cpu": "3","memory": "4Gi"},"requests": {"cpu": "3","memory": "4Gi"}},"noobaa-db": {"limits": {"cpu": "3","memory": "4Gi"},"requests": {"cpu": "3","memory": "4Gi"}},"noobaa-endpoint": {"limits": {"cpu": "3","memory": "4Gi"},"requests": {"cpu": "3","memory": "4Gi"}}}}}'
```

Also patch the backingstore
```
oc patch BackingStore object-backing-store --type='merge' -p '{\n  "spec": {\n    "pvPool": {\n      "resources": {\n        "limits": {\n          "cpu": "1000m",\n          "memory": "4000Mi"\n        },\n        "requests": {\n          "cpu": "500m",\n          "memory": "500Mi"\n        }\n      }\n    }\n  }\n}'
```

This should cause the relevant Noobaa pod to be restarted with the new requests and limits to be defined.
