from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
import os
from azure.ai.ml.entities import Environment
from azure.ai.ml import load_component
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

cpu_compute_target = "aml-cluster"
train_src_dir = "./components/train"
data_prep_src_dir = "./components/data_prep"


def create_client():
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="59a62e46-b799-4da2-8314-f56ef5acf82b",
        resource_group_name="rg-azuremltraining",
        workspace_name="dummy-workspace"
    )
    return ml_client


def create_data(ml_client):
    web_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

    credit_data = Data(
        path=web_path,
        type=AssetTypes.URI_FILE,
        name="christof_creditcard_defaults",
        tags={"creator":"christof"}
    )

    ml_client.data.create_or_update(credit_data)

    credit_data = ml_client.data.create_or_update(credit_data)
    print(
        f"Dataset with name {credit_data.name} was registered to workspace, the dataset version is {credit_data.version}"
    )
    return credit_data


def create_environment(ml_client):
    custom_env_name = "christof_environment"


    pipeline_job_env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="./dependencies/conda.yaml",
        name="christof_environment",
        tags={"creator":"christof"}
    )

    ml_client.environments.create_or_update(pipeline_job_env)

    return pipeline_job_env



def create_data_prep(ml_client, data_prep_src_dir, pipeline_job_env):
    data_prep_component = command(
        name="data_prep_credit_defaults",
        display_name="Data preparation for training",
        description="reads a .xl input, split the input to train and test",
        inputs={
            "data": Input(type="uri_folder"),
            "test_train_ratio": Input(type="number"),
        },
        outputs=dict(
            train_data=Output(type="uri_folder", mode="rw_mount"),
            test_data=Output(type="uri_folder", mode="rw_mount"),
        ),
        # The source folder of the component
        code=data_prep_src_dir,
        command="""python data_prep.py \
                --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
                --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
                """,
        environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    )

    data_prep_component = ml_client.create_or_update(data_prep_component.component)

    # Create (register) the component in your workspace
    print(
        f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
    )

    return data_prep_component


def create_train(ml_client, train_src_dir):

    # Loading the component from the yml file
    train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

    train_component = ml_client.create_or_update(train_component)

    # Create (register) the component in your workspace
    print(
        f"Component {train_component.name} with Version {train_component.version} is registered"
    )

    return train_component
    
def create_pipeline(ml_client, credit_data, data_prep_component, train_component):
    
    @dsl.pipeline(
        compute=cpu_compute_target,
        description="E2E data_perp-train pipeline",
    )
    def credit_defaults_pipeline(
        pipeline_job_data_input,
        pipeline_job_test_train_ratio,
        pipeline_job_learning_rate,
        pipeline_job_registered_model_name,
    ):
        # using data_prep_function like a python call with its own inputs
        data_prep_job = data_prep_component(
            data=pipeline_job_data_input,
            test_train_ratio=pipeline_job_test_train_ratio,
        )

        # using train_func like a python call with its own inputs
        train_job = train_component(
            train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
            test_data=data_prep_job.outputs.test_data,  # note: using outputs from previous step
            learning_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
            registered_model_name=pipeline_job_registered_model_name,
        )

        # a pipeline returns a dictionary of outputs
        # keys will code for the pipeline output identifier
        return {
            "pipeline_job_train_data": data_prep_job.outputs.train_data,
            "pipeline_job_test_data": data_prep_job.outputs.test_data,
        }

    registered_model_name = "christof_credit_defaults_model"

    # Let's instantiate the pipeline with the parameters of our choice
    pipeline = credit_defaults_pipeline(
        pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
        pipeline_job_test_train_ratio=0.25,
        pipeline_job_learning_rate=0.05,
        pipeline_job_registered_model_name=registered_model_name
    )

    # submit the pipeline job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        # Project's name
        # Modify the experiment name to include your name!
        experiment_name="e2e_registered_components_christof",
    )

    print(pipeline_job.studio_url)


def process():

    ml_client = create_client()
    credit_data = create_data(ml_client)
    pipeline_job_env = create_environment(ml_client)
    data_prep_component = create_data_prep(ml_client, data_prep_src_dir, pipeline_job_env)
    train_component = create_train(ml_client, train_src_dir)
    create_pipeline(ml_client, credit_data, data_prep_component, train_component)


if __name__ == "__main__":
    process()