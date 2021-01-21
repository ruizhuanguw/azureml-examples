# Using Synapse in Azure machine learning

## Create synapse resources

Follow up the documents to create Synapse workspace and resource-setup.sh is available for you to create the resources.

- Create from [Portal](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace)
- Create from [Cli](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace-cli)

Follow up the documents to create Synapse spark pool

- Create from [Portal](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-apache-spark-pool-portal)
- Create from [Cli](https://docs.microsoft.com/en-us/cli/azure/ext/synapse/synapse/spark/pool?view=azure-cli-latest)

## Link Synapse Workspace

Make sure you are the owner of synapse workspace so that you can link synapse workspace into AML.
You can run resource-setup.py to link the synapse workspace and attach compute

```python
from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core import LinkedService, SynapseWorkspaceLinkedServiceConfiguration
synapse_link_config = SynapseWorkspaceLinkedServiceConfiguration(
    subscription_id="<subscription id>",
    resource_group="<resource group",
    name="<synapse workspace name>"
)

linked_service = LinkedService.register(
    workspace=ws,
    name='<link name>',
    linked_service_config=synapse_link_config)

```

## Attach synapse spark pool as AzureML compute

```python

from azureml.core.compute import SynapseCompute, ComputeTarget
spark_pool_name = "<spark pool name>"
attached_synapse_name = "<attached compute name>"

attach_config = SynapseCompute.attach_configuration(
        linked_service,
        type="SynapseSpark",
        pool_name=spark_pool_name)

synapse_compute=ComputeTarget.attach(
        workspace=ws,
        name=attached_synapse_name,
        attach_configuration=attach_config)

synapse_compute.wait_for_completion()
```

## Set up permission

Grant Spark admin role to system assigned identity of the linked service so that you can submit experiment run or pipeline run from AML workspace to synapse spark pool.

You can get the system assigned identity information by running

```python
print(linked_service.system_assigned_identity_principal_id)
```

- Launch synapse studio of the synapse workspace and grant linked service MSI "Synapse Apache Spark administrator" role.

- In azure portal grant linked service MSI "Storage Blob Data Contributor" role of the primary adlsgen2 account of synapse workspace to use the library management feature.
