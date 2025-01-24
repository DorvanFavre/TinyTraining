# TinyTraining

## Pipeline
Independinaly of the target device, the task, the dataset, the model and the training technics, a common pipeline remains the same. 

- The process is divided in two phases. The offline phses which can be done on a host computer and consist of preparing the model and the dataset. The online phase which is done on the target device and consist of traning and evaluating the model.

- The implementation differ according to the traning framework. For now, two traning framework are available anmely ORT and LiteRT.

- In the offline phase, there is three steps. Preparing the model, preapring the dataset and preparing the artifacts.

- In the online phase, there are two steps: Traning the model and then inferencing.


## Folder structure

- **Offline Phase**
  - **Framework**
    - ORT
      - **Task**
        - MNIST calssification
          - **Models**
            - LeNet-5
    - LiteRT
- **Online Phase**

# Use cases

**MINST using ORT**
1. OfflinePhase/ORT/PrepareModel/MNIST.py
2. OfflinePhase/ORT/PrepareDataset/MNIST.py
3. OfflinePhase/ORT/PrepareArtifacts/full_backpropagation.py
4. OnlinePhase/ORT/Training/train.py

**Mobilenet**


## Benchmarking
Each time a pipeline is executed, the result along side the configuration of the pipline is saved and thus a dashboard with comparisons is available.