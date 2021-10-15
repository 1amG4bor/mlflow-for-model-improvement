# MLflow for model improvement
This project aims to show how MLflow can be used to analyze, configure and improve our model to perform better
 whether it is a Deep Neural Network (DNN), Convolutional Neural Network, or a simple Feed-forward Neural Network.

>:warning: **This repository contains the source code for the meet-up session that speak of how to use MLflow to improve
 our models.**<br>
>
>![](https://img.shields.io/static/v1?label=Session%20date&message=November%2025,%202021&color=success&logo=meetup) <br>
>![](https://img.shields.io/static/v1?label=Link%20to%20event&message=Coming%20soon..&color=red&&logo=meetup)
  
## 1). Used technologies:
- Python 3.8
- TensorFlow-2 (Keras)
- MLflow
- Docker

## 2). Configuration

There are 2 configuration file that need to be filled properly to able to run script.

- **.env**: file provides the values that is needed to build up and run the docker containers where the MLflow server
 will run
- **src/config.yaml**: contains configurations that manage how the script run 
    - download: group of values to define the dataset source for download and how to extract it, if locally does not
     exist
    - model: group of values to define how the model should be built up
  

- #### Data source `YAML configuration` `CLI arguments`
    - **download dataset**: with a YAML configuration and command-line arguments, the dataset is downloaded and
     cached automatically (will not download again on the next run)
    - **load existing dataset**: command line arguments make it possible to choose any dataset that stored locally
     *(~/.keras/datasets/...)*
    - **create filtered sub-dataset**: command line arguments make it easy to create a filtered dataset that helps
     you fine-tune your model with larger and larger data sets
    - **combine the listed options above**: you can download and create a sub-dataset OR choose an existing dataset
     and create filtered one in one single step 
- #### Training process `CLI arguments`
    - **batch-size**: can be set for the training (default: 64) - *hyperparameter that defines the number of training
     examples utilized in one iteration before updating the internal model parameters* 
    - **epoch**: can be set for the training (default: 20) - *hyperparameter that defines the number of times that the
     learning algorithm will work through the entire training dataset*
    - **validation split**: number between 0 and 1 that defines how split the training data for training and for
     validation
- #### Model creation `YAML configuration`
    - **TBD**
     
### How to configure

CLI arguments for more information can be listed with the following command: **`python <train_eval.py> -h`**

Some basic configuration: 
- Download the dataset specified in YAML config and extract to a folder named 'my_dataset'
    <br>**`python train_eval -d -ds my_dataset`** (-d | --download, -ds | --dataset')
- Create a filtered dataset based on an existing one with the specified classes
    <br>**`python train_eval -ds flower_photos -sub less_flowers_dataset roses tulips`** (-sub | --subset')
- Train our dataset with given batch size and epoch
    <br>**`python train_eval -ds my_dataset -b 48 -ep 100`** (-b | --batch, -ep | --epoch')

YAML configuration is: **`src/config.yml`**

## 3). Related articles and documentations

- The article *"Deploy MLflow with docker compose"*[^1] helped me in the implementation where I could borrow
 the configuration how to setup the dockerized environment.
- The documentation of **MLflow**[^2][^3] and **TensorFlow**[^4] are quite verbose with lot of example that make it
 easy to 

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
Estimator: High level tools for working with models: https://www.tensorflow.org/api_docs/python/tf/estimator
