# QAit Task - Graph Attention Network Extension - Interactive Question Answering in Text-Based Environments
--------------------------------------------------------------------------------
Extension to Code for EMNLP 2019 paper "Interactive Language Learning by Question Answering".

## Code Contribution

Descriptions of all added files, classes and functions mentioned in the following sections can be found in code.

### Added Files

- knowledge_graph.py

- bert_embedder.py

### Added Classes in Existing Files

- model.py - Added the GATContainer and GAT classes. 

- layers.py - Added the Transformer and LayerNorm classes.


### Other Modified Files

Any references to files and classes defined in the above two sections or their functions and variables ensure overall integration between the GAT components and the existing QAit architecture. The main references to these files, classes, functions and variables in the existing files are:

- model.py - Added necessary code to integrate GAT output into the action_scorer (also known as the Command Generator) and answer_question functions.

- layers.py - Added necessary code to integrate GAT output into the AnswerPointer class.

- train.py - Edit code to include the Knowledge Graph (KG) and Graph Attention Network (GAT) in each training step. This includes adding the KG state into the replay buffer (for batching) and resetting the KG after a new game.

- agent.py - Edit code to include the GAT and its components in the agent's architecture. This includes when performing actions (act, act_greedy, act_random and get_ranks functions), answering questions (answer_question and answer_question_act_greedy functions) and calculating loss (get_qa_loss and get_dqn_loss functions).

- qa_memory.py - Added two variables to be stored into the replay buffer (for batching purposes), namely the number of entities in the graph and the edge indices of the graph.

- command_generation_memory.py - Same as qa_memory.py

- evaluation.py - Edit code to reset the KG after a new game.

## Installation

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
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install gym==0.15.4
conda install h5py
conda install pytorch-geometric -c rusty1s -c conda-forge
pip install nltk==3.3
pip install transformers==4.9.2
```

## Downloads

### Stanford OpenIE
Download CoreNLP and the English model jar file from [CoreNLP website](https://stanfordnlp.github.io/CoreNLP/). Unzip everything. Place the English model jar file in the downloaded CoreNLP directory.


### Tokenizer
The NLTK punkt dataset needs to be downloaded to tokenise state descriptions in the Knowledge Graph class. It can be downloaded by opening the python interpreter with the command "python" and running the following:

```
>>> import nltk
>>> nltk.download(punkt)
```

### Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it and place it in the main directory.


### Pretrained Word Embeddings
Download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip the file and place it in the main directory.

## Running

### OpenIE

In a separate terminal, open the downloaded CoreNLP directory. The OpenIE extractor can then be run by running the following command:

```
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### To Train

In the main directory, the following command can be run to start training:

```
python train.py ./
```

## Configurations

The config.yaml file contains all the settings and customisation a user can make to QAit and the GAT. This includes the number of episodes, question type, number of games trained on, component dimensions and other architecture specifications. The port number of OpenIE can also be specified (in the README, it is run on port 9000).

The config.yaml file in the repository is set to have the same configurations as the original QAit paper.

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