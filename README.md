# Multimodal Misinformation Detection using Large Vision-Language Models

This repository is the official implementation of [Multimodal Misinformation Detection using Large Vision-Language Models](https://doi.org/10.1145/3627673.3679826) published in CIKM 2024.

## Requirements

To install requirements:
```
conda create -n LLM4FV python=3.8.10
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3.1 -c pytorch
pip install -r requirements.txt
```

## Dataset


- MOCHEG: You can download dataset [here](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform).

    For more information on the dataset format and structure, please refer to their Please check their [official repository](https://github.com/VT-NLP/Mocheg).
- Factify: The dataset can be downloaded [here](https://drive.google.com/drive/folders/1MPgUN6xAnocENZ5fefny3eGOnD7KPZ5M?usp=sharing).

    Additional details can be found on their [official website](https://competitions.codalab.org/competitions/35153).

## Evaluation
To run the pipeline and evaluate on datasets, run:
```
bash eval.sh
```
## Credit
This repository is built by [Sahar Tahmasebi](https://github.com/sahartahmasebi). 

## Contributing

Our work is licenced under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project includes code that is licensed under the Apache License 2.0, from the [MOCHEG](https://github.com/VT-NLP/Mocheg) repository by [Barry Menglong Yao](https://github.com/Barry-Menglong-Yao). The original code has been modified to suit the needs of this project.
