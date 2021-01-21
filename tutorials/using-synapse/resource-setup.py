from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core import LinkedService, SynapseWorkspaceLinkedServiceConfiguration
synapse_link_config = SynapseWorkspaceLinkedServiceConfiguration(
    subscription_id=ws.subscription_id,
    resource_group=ws.resource_group,
    name="azuremlexamples"
)

linked_service = LinkedService.register(
    workspace=ws,
    name='synapselink',
    linked_service_config=synapse_link_config)

from azureml.core.compute import SynapseCompute, ComputeTarget
spark_pool_name = "sparkpool"
attached_synapse_name = "synapsecompute"

attach_config = SynapseCompute.attach_configuration(
        linked_service,
        type="SynapseSpark",
        pool_name=spark_pool_name)

synapse_compute=ComputeTarget.attach(
        workspace=ws,
        name=attached_synapse_name,
        attach_configuration=attach_config)

synapse_compute.wait_for_completion()