# QAIT Task Extension - Interactive Question Answering in Text-Based Environments
--------------------------------------------------------------------------------
Extension to Code for EMNLP 2019 paper "Interactive Language Learning by Question Answering".
Implementation of policy-based agent using REINFORCE with baseline. 
Implementation of ICM module to use as regularization technique for semantic encoding.

## Created Files
- check_max.py - used for checking training logs to find saved model and best results

- plot.py - used for plotting training curves

## Modified Files
- model.py - added ActorCritic, ICM_Inverse, ICM_Forward, ICM_Feature, and ICM classes. ActorCritic is the policy and baseline networks. All the ICM classes are used for the environment dynamics model. 

- command_generation_memory.py - add single episode memory storage to not involve the shuffling and batching of data and just store a single episode of data.

- evaluate.py - add other accuracy metrics including F1 score, precision, and recall.

- agent.py - Add methods to make use of ActorCritic and ICM Classes. This includes action selection, policy and baseline network updates, ICM network updates.

- train.py - add code to make new agent supported. This includes new storage of relevant data, calculation of intrinsic rewards, update method for agent depending on architecture selected.

## Running Changes
Alot of effort went into making all the original code unchanged with regards to execution and due to this all code is implemented with respect to the config file meaning all new changes and modifications can simply be turned off and the original methods can run. All changes are implemented in conjunction with existing code so mixing and matching methods are possible within the config file.

## Docs
Navigate to docs/build/index.html for a docs website.

## To install dependencies
```
sudo apt update
conda create -p ~/venvs/qait python=3.6
conda activate ~/venvs/qait
pip install --upgrade pip
pip install numpy==1.16.4
pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
pip install -U spacy
python -m spacy download en
pip install tqdm pyyaml
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install gym==0.15.4
conda install h5py 
pip install wandb sklearn plotly pandas jsonlines
```

## Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it.


## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## To Train
```
python train.py ./

-d <training location> (just use "./")
-l <flag to log to wandb> (requires logging into wandb)
```

## Citation
If this is used, please give acknowledgements and cite the original QAit paper.

The bibtex for the original QAit paper is:

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
