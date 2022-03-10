This repository contains the source code for the paper "Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis", which was presented at AAAI 2023.

The experiments are found the notebook `gnn_dissect.ipynb`, which can be run in Google Colab Pro+. This trains the models on the datasets studied in the paper and extracts concepts for them.

Further analysis on the concepts of each dataset can be found in the notebooks in the `src` directory. Running them requires placing the trained models in the `models` directory and the extracted concepts in the `concepts` directory.

### Citing

```
@article{xuanyuan2022global,
  title={Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis},
  author={Xuanyuan, Han and Barbiero, Pietro and Georgiev, Dobrik and Magister, Lucie Charlotte and Li{\'o}, Pietro},
  journal={arXiv preprint arXiv:2208.10609},
  year={2022}
}
```