# Noah_DL_Hybrid_parameterization
Optimizing Thermal and Hydrological Processing Simulation on the Qinghai-Tibet Plateau by Integrating Deep Learning and Land Surface Model.

## Outline

1. [Project Structure](#project-structure)

## Project Structure

```plaintext
project-root/
│  README.md
│  
├─Parameter_Generator
│  │  LSMTransformer.py
│  │  PGDataset.py
│  │  PGNetwork.py
│  │  PG_TRAINING_UWC_WANDB.ipynb
│  │  
│  ├─GENERATOR
│  │      README.md
│  │      
│  └─PG_DATASET
│      │  static_propertise.csv
│      │  
│      ├─CMFD2FLOAT
│      │      README.md
│      │      
│      ├─GRID_NPY
│      │      README.md
│      │      
│      └─GRID_NPY_QQ
│              README.md
│              
└─Transformer_TEST
    │  LSMDataset.py
    │  LSMLoss.py
    │  LSMTransformer.py
    │  SURROGATE_TRAINING_STC_WANDB.ipynb
    │  SURROGATE_TRAINING_UWC_WANDB.ipynb
    │  
    ├─SURROGATE
    │      README.md
    │      
    └─TEMP
            README.md
```
- **Parameter_Generator**: the define of model and dataset and training code for parameter generator are in this directory. 
    - **LSMTransformer.py**: self-defined model of surrogate
    - **PGDataset.py**: self-defined dataset to load cmfd, static propertice and SMCI(GROUND TRUTH) of study area
    - **PGNetwork.py**: self-defined model of parameter generator
    - **PG_TRAINING_UWC_WANDB.ipynb**: a training sample using wandb to train the parameter generator tested by UWC
    - **GENERATOR**: save the well-trained model in this directory.
    - **PG_DATASET**: the dataset of input and ground truth in this directory.
- **Transformer_Test**: the define of model and dataset and training code for surrogate are in this directory.
    - **LSMDataset.py**: self-defined dataset to load cmfd and simulation of Noah as ground truth of study area
    - **LSMLoss.py**: self-defined loss to add physics constraint
    - **LSMTransformer.py**: self-defined model of surrogate
    - **SURROGATE_TRAINING_STC_WANDB.ipynb**: a training sample using wandb to train the surrogate tested by STC
    - **SURROGATE_TRAINING_UWC_WANDB.ipynb**: a training sample using wandb to train the surrogate tested by UWC
    - **SURROGATE**: save the well-trained model in this directory.
    - **TEMP**: the datatset of input and ground truth in this directory.



















