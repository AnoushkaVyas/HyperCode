# HyperCode

In this project, we are going to use hyperbolic graph models to encode ASTs to increase the performance of models like CodeT5 in the tasks of code understanding and generation. Our algorithm, referred as HyperCode, provides an end-to-end learning of a graph model and a language model for the code matching task. The report of the project is given here [link](https://github.com/AnoushkaVyas/HyperCode/blob/main/HyperCode.pdf).

## Installation

```
conda conda env create -f hypercode.yml
conda activate hypercode
```

## Training

```
python main.py \
    --data_name 'amazon_s' \
    --data_path 'data' \
    --outdir 'output/amazon_s' \
    --pretrained_embeddings 'data/amazon_s/amazon_s.emd' \
    --n_epochs 10 \
    --n_layers 4 \
    --n_heads 4 \
    --gcn_option 'no_gcn' \
    --node_edge_composition_func 'mult' \
    --ft_input_option 'last4_cat' \
    --path_option 'shortest' \
    --ft_n_epochs 10 \
    --num_walks_per_node 1 \
    --max_length 6 \
    --walk_type 'dfs' \
    --is_pre_trained
 ```
