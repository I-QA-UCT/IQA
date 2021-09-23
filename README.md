# QAIT Task Extension - Interactive Question Answering in Text-Based Environments
--------------------------------------------------------------------------------
Extension to Code for EMNLP 2019 paper "Interactive Language Learning by Question Answering".

## To install dependencies
```
sudo apt update
conda create -p ~/venvs/qait python=3.6
source activate ~/venvs/qait
pip install --upgrade pip
pip install numpy==1.16.4
pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
pip install -U spacy
python -m spacy download en
pip install tqdm h5py visdom pyyaml
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install gym==0.15.4 <!-- (Dependency Issue with Original Code) -->
pip install transformers==4.9.2
```

## Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it.

## Note
Adapted from Chen et al.:
```
/decision_transformer/model_qait.py # contains base classes for Decision Transformer and QA model
/decision_transformer/trainer_qait.py # methods for training classes in model_qait.py
/decision_transformer/experiment.py # driver class - experiment details are specified here with command line args.
```

Minor changes made to Yuan et al.'s QAit codebase, namely:
```
agent.py # Added functionality to take in Decision Transformer and QA model
evaluate.py # can be used to evaluate DT or DT-BERT
reward_helper.py # functionality created for intermediate rewards
```

## Validation Set
Run the following python command to generate validation set:
```
python IQA/create_eval_set.py --games [number of games in set]
```

## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## Generate Random Rollouts
```
python record_trajectories.py -env [filename of dataset] -sui [sufficient information reward threshold for trajectory]
```
For example, here new offline trajectories and read into /fixed_map/attribute-500-qa.json. These are generated with a cutoff of 0.0 indicating all trajectories, despite some not having gained any reward, of the random agent will be recorded.
```
python record_trajectories.py -env "fixed_map/attribute-500-qa" -sui 0.0
```

## To Train Decision Transformer
Using command line arguments, we specify parameters of DT:
```
python decision_transformer/experiment.py --dataset "[dataset name]" --env "[experiment name]" --device "['cuda' or 'cpu']" --state [number of tokens in state] --embed_dim [embedding dimension] --K [number ] --batch_size [batch size] --max_iter [number of iterations in training] --dropout [dropout] --eval_per_iter [specify every how many episodes to evaluate on] --random_map [is the dataset random or fixed map] -w [log to wandb] --question_type "[specify question type]"
```
For example:
```
python decision_transformer/experiment.py --dataset "location-500_split" --env "location-500-random_map" --device "cuda" --state 180 --embed_dim 256 --n_layer 2 --K 50 --batch_size 128 --max_iter 5000 --dropout 0.5 --eval_per_iter 250 --random_map True -w True -qt "location"
```

## To Train Question Answerer
Similarly to the Decision Transformer training, here is an example of training the QA model for attribute questions:
```
python decision_transformer/experiment.py --dataset "attribute-500"  --model_type "qa" --device "cuda" --env "attribute-500-fixed_map-qa-module" --batch_size 16 --vocab_size 1654 --state_context_window 512 --num_workers 5 --max_iters 30 -qt "attribute" --pretrained_model "bert" -lr 1e-5 -load "attribute-500-fixed_map"
```

## To Evaluate on Test Set
``` 
python evaluate.py -m [Decision Transformer env name] -qa [QA model env name (OPTIONAL)]
```
For example:
```
python evaluate.py -qa existence-500-random_map-qa-module -m existence-500-random_map
```

## Citation

Please use the following bibtex entry:
```
@article{yuan2019qait,
  title={Interactive Language Learning by Question Answering},
  author={Yuan, Xingdi and C\^ot\'{e}, Marc-Alexandre and Fu, Jie and Lin, Zhouhan and Pal, Christopher and Bengio, Yoshua and Trischler, Adam},
  booktitle={EMNLP},
  year={2019}
}
```

## License

[MIT](./LICENSE)
