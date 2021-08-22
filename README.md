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
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch (Or later versions work fine)
pip install gym==0.15.4 (Dependency Issue with Original Code)
conda install h5py (Dependency Issue with Original Code)
pip install wandb (Only for logging to wandb)
```

## GAT Dependencies
```
conda install pytorch-geometric -c rusty1s -c conda-forge
pip install nltk==3.3
pip install matplotlib==2.2.3
pip install transformers==4.9.2
```

## Stanford OpenIE Downloads
Download CoreNLP and English model jar from [https://stanfordnlp.github.io/CoreNLP/](CoreNLP website). Unzip everything. Place the English model jar in the CoreNLP directory, and place the CoreNLP directory in IQA directory. 

NOTE: The CoreNLP directory should be labelled "stanford-corenlp-4.2.2".


## Other Download Files
The NLTK punkt dataset needed to be downloaded in order to tokenize state descriptions for the KG class. It can be downloaded by opening the python interpreter with the command "python", and running the following:

```
>>> import nltk
>>> nltk.download(punkt)
```

## Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it.


## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## To Train
```
python train.py ./
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