import torch
from transformers import AutoTokenizer, AutoModel
​
from torch import nn
​
​
class BertEmbedder(nn.Module):
    pass

if __name__ == "__main__":
    # embedder = BertEmbedder("mini", ["embeddings", "pooler", "LayerNorm", "layer.0"])
    # embedder.embed("hello world . I am Groot!")
    # embedder.embed(["hello world . I am Groot!", "yoyoyo"])