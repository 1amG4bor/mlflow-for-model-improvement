# MLflow for model improvement

###### Dependencies
![](https://img.shields.io/badge/Python-3.8-blue)
![](https://img.shields.io/badge/TensorFlow-2.6.0-orange)
![](https://img.shields.io/badge/Keras-2.6.0-red)
![](https://img.shields.io/badge/MLflow-1.20.2-lightblue)
![](https://img.shields.io/badge/Docker-20.10.7-success)

## 1). Introduction

This project aims to show how MLflow can be used to analyze, configure and improve our model to perform better
 whether it is a Deep Neural Network (DNN), Convolutional Neural Network, or a simple Feed-forward Neural Network.

>:warning: **This repository contains the source code and slides for the online Meet-up session that speaks of**:
> - what is MLOps
> - what is ML pipeline
> - how to use MLflow to improve our models.**
>
> The session was held on November 25, 2021 organized on the Community-Z platform.

[**About the "Talk" on Community-Z portal**](https://community-z.com/events/model_improvement_with_ml_flow)

[**Recording of the Meetup** - On YouTube](https://www.youtube.com/watch?v=_2ScOR83CE4)


## 2). How to run

Commands for creating the environment, running the model creation, training, and evaluation, and last but not least
 deploying and using the model.
 
- Checking that the required dependencies are installed<br>
`python --version & conda --version & docker --version`
- Build and run the docker containers that serve the containerized MLflow server on localhost<br>
`docker-compose up -d --build`
---
- Script to create, train, and evaluate the model<br>
`python src\train_eval.py --dataset=my-dataset --epoch=25 --name=MyModel -run-name=FirstTest --create`
- Script to load and evaluate a pre-trained model<br>
`python src\train_eval.py --dataset=my-dataset --name=MyPretrainedModel`

*For configuration options see below at "CLI options" or run the command with the flag `-h` or `--help`*

---
- Serve model from S3<br>
`mlflow models serve -m "s3://**bucket-name**/**folder-path**/**experiment_id**/**run_id**/artifacts/model" -p
 **port-number** -h **host**`<br>
e.g to serve the model on localhost: `mlflow models serve -m "s3://my-bucket/ml-runs/1/01010myrunid01010/artifacts/model
" -p 9000 -h 0.0.0.0`
- Serve model from the model repository<br>
`mlflow models serve -m "models:/my-model-name/Production" -p 9000 -h 0.0.0.0`
- Script to predict on served model<br>
`python src\predict.py --labels class-1 class-2 class-3 --classification-type=my-classification-type`<br>
e.g: `python src\predict.py --labels "apple" "banana" "peach"  --classification-type="fruit"`
- Deploy MLflow model in docker container<br>
 `mlflow models build-docker -m "runs:/<my-run-id>/model" -n "<my-image-name>"`

## 3). How to configure

Two configuration files need to be filled properly to be able to run the script, besides that many CLI (command-line
 interface) options are there that help to manage the script during the training process.

##### Options:
- **`.env` file**: Provides the values that are needed to build up and run the docker containers where the MLflow server will run. 
- **`config.yaml`**: Contains configurations that manage the data source, the model structure, deployment related
 information, and the MLflow basic config as well.
- **CLI options**: Collection of configuration values that manage how the script runs.

---

### .env file

Includes all the pieces of information that are necessary to build up the database which will store the logs, such as
 parameters, hyper-parameters, etc. These logs will be fetched by the MLflow UI. This file contains the AWS keys too
  that are required to reach AWS S3 to store and fetch the logged models.

### YAML Configuration

It has four parts, three of them are responsible for configuring the phases of the Machine Learning Pipeline, and the
 last one is the base MLflow configuration.   

##### Sections:
- **datasource**: values to define the source of the dataset to download and info how to extract it if locally does not exist
- **model**: group of values to define the model structure and how it should be built up
    - **convolution**: configure the convolutional layers to extract the features
    - **dense**: configure the dense layers to utilize the extracted features and provide predictions
    - **optimizer**: define the optimizer function for the training process
    - **save_format**: output format for saving the trained model
- **deployment**: configuration that defines values for the deployment stage
- **mlflow**: define where the MLflow server run and the group name of the experiments

### CLI options

The CLI arguments extend the configuration options that the YAML configuration provides thereby you can add additional
configurations that affect the workflow.

- ###### Data source `YAML configuration` `CLI arguments`
    - **download dataset**: with a YAML configuration and command-line arguments, the dataset is downloaded and
     cached automatically (will not download again on the next run)
    - **load existing dataset**: command-line arguments make it possible to choose any dataset that is stored locally
     *(~/.keras/datasets/...)*
    - **create filtered sub-dataset**: command-line arguments make it easy to create a filtered dataset that helps
     you fine-tune your model with larger and larger data sets instead of using the full dataset from the beginning
    - **combine the listed options above**: you can download and create a sub-dataset OR choose an existing dataset
     and create filtered one in one single step 
- ###### Training process `CLI arguments`
    - **batch-size**: can be set for the training (default: 64) - *hyperparameter that defines the number of training
     examples utilized in one iteration before updating the internal model parameters* 
    - **epoch**: can be set for the training (default: 20) - *hyperparameter that defines the number of times that the
     learning algorithm will work through the entire training dataset*
    - **test split**: percentage value that defines how split the full dataset for training and testing
    - **validation split**: percentage value that defines how split the training dataset for training and validation
  
Example CLI configurations: 
- Download the dataset specified in YAML config and extract it to a folder named, e.g: 'my_dataset'
    <br>**`python train_eval -d -ds my_dataset`** (-d | --download, -ds | --dataset)
- Create a filtered dataset based on an existing one with the specified classes
    <br>**`python train_eval -ds fruits -sub less_fruits_dataset apple banana`** (-sub | --subset)
- Train model on existing dataset with given batch size and epoch
    <br>**`python train_eval -ds my_dataset -b 48 -ep 100`** (-b | --batch, -ep | --epoch)
- Train model on existing dataset with custom data segregation and deploy limit
    <br>**`python train_eval -ds my_dataset -ts 30 -vs 10 -dl 95`** (-ts | --test-split, -vs | --validation-split, -dl | --deploy-limit)
    

## 3). Related articles and documentations

- The article *"Deploy MLflow with docker compose"*[^1] helped me in the implementation where I could borrow
 the configuration of how to set up the dockerized environment.
- The documentation of **MLflow**[^2][^3] and **TensorFlow**[^4] are quite verbose with a lot of examples that make it
 easy to use them.
- The Glossary section is based on **ml-cheatsheet - glossary**[^5]
- Introduction to MLflow - **Videos about the components of MLflow**[^5]

***

### References:

[^1]: Guillaume Androz (Jan 10, 2020). [Deploy MLflow with docker compose](https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039) *`towardsdatascience.com`*
[^2]: *MLflow - Quickstart*: https://www.mlflow.org/docs/latest/quickstart.html
[^3]: *MLflow - Tracking*: https://www.mlflow.org/docs/latest/tracking.html
[^4]: *TensorFlow API documentation - python*: https://www.tensorflow.org/api_docs/python/tf <br>
data.Dataset - *API for input pipelines*: https://www.tensorflow.org/api_docs/python/tf/data/Dataset <br>
Keras, models - *API for model creation*: https://www.tensorflow.org/api_docs/python/tf/keras/models <br>
Keras, layers - *How to build layers* : https://www.tensorflow.org/api_docs/python/tf/keras/layers <br>
Keras, callbacks - *Performed actions at various stages of training*: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks <br>
[^5]: https://ml-cheatsheet.readthedocs.io/en/latest/glossary.html <br>
[^6]: https://databricks.com/discover/managing-machine-learning-lifecycle <br>

***

### Social links 

[![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/1amG4bor)
[![](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabortanacs)
[]()
[]()