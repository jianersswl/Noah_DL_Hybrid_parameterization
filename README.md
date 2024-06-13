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
