# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset, Datastore

subscription_id = '389384f8-9747-48b4-80a2-09f64d0a0dd7'
resource_group = 'BizQA-WUS3-RG-GpuClusterA100'
workspace_name = 'BizQA-Dev-WUS3-AML'


workspace = Workspace(subscription_id, resource_group, workspace_name)
  
datastore = Datastore.get(workspace, "workspaceblobstore")
dataset = Dataset.File.from_files(path=(datastore, 'chai/datasets/wikikg90m-v2'))

dataset.download(target_path='/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed/chai_candidates/', overwrite=True)