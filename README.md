# HyperCode

In this project, we are going to use hyperbolic graph models to encode ASTs to increase the performance of models like CodeT5 in the tasks of code understanding and generation. Our algorithm, referred as HyperCode, provides an end-to-end learning of a graph model and a language model for the code matching task. The report of the project is given here [link](https://github.com/AnoushkaVyas/HyperCode/blob/main/HyperCode.pdf). The project also uses the library [GraphZoo](https://github.com/AnoushkaVyas/GraphZoo).

## Installation

```
conda conda env create -f hypercode.yml
conda activate hypercode
pip install graphzoo
```

## Training

```
python main.py \
    --output_dir 'saved_models/pretrain/' \
    --train_batch_size 5 \
    --eval_batch_size 5 \
    --test_batch_size 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.0005 \
    --adam_epsilon 1e-8 \
    --num_train_epochs 100 \
    --validate_every 5 \
    --seed 42
 ```
 
 
