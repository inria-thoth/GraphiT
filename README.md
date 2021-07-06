# GraphiT: Encoding Graph Structure in Transformers

This repository implements GraphiT, described in the following paper:

>Grégoire Mialon*, Dexiong Chen*, Margot Selosse*, Julien Mairal.
[GraphiT: Encoding Graph Structure in Transformers][1].
<br/>*Equal contribution

## Short Description about GraphiT

![Figure from paper](figures/figure1.png)

GraphiT is an instance of transformers designed for graph-structured data. It takes as input a graph seen as a set of its node features, and integrates the graph structure via i) relative positional encoding using kernels on graphs and ii) encoding local substructures around each node, e.g, short paths, before adding it to the node features. GraphiT is able to outperform Graph Neural Networks in different graph classification and regression tasks, and offers promising visualization capabilities for domains where interpretability is important, e.g, in chemoinformatics.

## Installation

Environment:
```
numpy=1.18.1
scipy=1.3.2
Cython=0.29.23
scikit-learn=0.22.1
matplotlib=3.4
networkx=2.5
python=3.7
pytorch=1.6
torch-geometric=1.7
```

The train folds and model weights for visualization are already provided at the correct location. Datasets will be downloaded via Pytorch geometric.

To begin with, run:
```
cd GraphiT
. s_env
```

To install GCKN, you also need to run:
```
make
```

You also need to create a `cache` folder to store computed positional encoding
```
mkdir -p cache/pe
```

## Training GraphiT on graph classification and regression tasks

All our experimental scripts are in the folder `experiments`. So to start with, run `cd experiments`.

#### Classification

To train GraphiT on NCI1 with diffusion kernel, run:
```bash
python run_transformer_cv.py --dataset NCI1 --fold-idx 1 --pos-enc diffusion --beta 1.0
```

Here `--fold-idx` can be varied from 1 to 10 to train on a specified training fold. To test a selected model, just add the `--test` flag.

To include Laplacian positional encoding into input node features, run:
```bash
python run_transformer_cv.py --dataset NCI1 --fold-idx 1 --pos-enc diffusion --beta 1.0 --lappe --lap-dim 8
```

To include GCKN path features into input node features, run:
```bash
python run_transformer_gckn_cv.py --dataset NCI1 --fold-idx 1 --pos-enc diffusion --beta 1.0 --gckn-path 5
```

###### Reproduction of our classification results

To reproduce our experimental results, you need to perform grid search to select the best model and retrain it. We have prepared a script to perform grid search and testing on a single machine for MUTAG with GCKN and adjacency encoding as an example. The results for other datasets and other encodings can be easily obtained by adapting the script. 
```bash
cd scripts
bach -x cv_gckn_transformer.sh
```

You can modify the above script based on your server to conduct grid search on multiple machines. Once all experiments have been done, you can visualize the final results with
```bash
python results_gckn_transformer.py
```

#### Regression

To train GraphiT on ZINC, run:
```bash
python run_transformer.py --pos-enc diffusion --beta 1.0
```

To include Laplacian positional encoding into input node features, run:
```bash
python run_transformer.py --pos-enc diffusion --beta 1.0 --lappe --lap-dim 8
```

To include GCKN path features into input node features, run:
```bash
python run_transformer_gckn.py --pos-enc diffusion --beta 1.0 --gckn-path 8
```

## Visualizing attention scores

To visualize attention scores for GraphiT trained on Mutagenicity, run:
```
cd experiments
python visu_attention.py --idx-sample 10
```
To visualize Nitrothiopheneamide-methylbenzene, choose 10 as sample index.
To visualize Aminofluoranthene, choose 2003 as sample index.
If you want to test for other samples (i.e, other indexes), make sure that the model correctly predicts mutagenicity (class 0) for this sample.

## Citation

To cite GraphiT, please use the following Bibtex snippet:
```
@misc{mialon2021graphit,
      title={GraphiT: Encoding Graph Structure in Transformers}, 
      author={Gr\'egoire Mialon and Dexiong Chen and Margot Selosse and Julien Mairal},
      year={2021},
      eprint={2106.05667},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[1]: https://arxiv.org/abs/2106.05667
