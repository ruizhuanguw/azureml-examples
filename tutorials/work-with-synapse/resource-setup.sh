az storage account create \
    --name azuremlexamples4synapse \
    --resource-group azureml-examples \
    --location eastus \
    --sku Standard_RAGRS \
    --kind StorageV2 \
    --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a \
    --enable-hierarchical-namespace true

az storage fs create --name test \
                     --account-name azuremlexamples4synapse \
                     --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a

az synapse workspace create --file-system test \
                            --location eastus \
                            --name azuremlexamples \
                            --resource-group azureml-examples \
                            --sql-admin-login-user sqladmin \
                            --sql-admin-login-password Passw0rd \
                            --storage-account azuremlexamples4synapse \
                            --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a

az synapse workspace firewall-rule create --name allowAll \
                                          --workspace-name azuremlexamples \
                                          --resource-group azureml-examples \
                                          --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a \
                                          --start-ip-address 0.0.0.0 \
                                          --end-ip-address 255.255.255.255

az synapse spark pool create --name sparkpool \
                             --workspace-name azuremlexamples 
                             --resource-group azureml-examples \
                             --subscription 6560575d-fa06-4e7d-95fb-f962e74efd7a \
                             --spark-version 2.4 \
                             --node-count 3 \
                             --node-size Medium \
                             --enable-auto-pause true \
                             --enable-auto-scale false