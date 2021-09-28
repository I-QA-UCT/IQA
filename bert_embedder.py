from transformers import AutoTokenizer, AutoModel

from torch import nn
import os


class BertEmbedder(nn.Module):
    """
    A pre-trained BERT embedder class. 
    """

    def __init__(self, size, bert_fine_tune_layers, device):
        super().__init__()

        self.device = device
        self.size = size
        self.bert_fine_tune_layers = bert_fine_tune_layers

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model_name = "prajjwal1/bert-" + self.size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print("bert config:",  self.model.config.__dict__)
        self.dims = self.model.config.hidden_size
        self.set_trainable_params()
        print("Loaded bert model with", self.dims, "dims and ", num_params(self), "trainable params")

    def set_trainable_params(self):
        def is_in_fine_tune_list(name):
            if name == "":  # full model is off by default
                return False

            for l in self.bert_fine_tune_layers:
                if l in name:
                    return True
            return False
            
        for param in self.model.parameters():
            """all params are turned off. then we selectively reactivate grads"""
            param.requires_grad = False
            
        for n, m in self.model.named_modules():
            if not is_in_fine_tune_list(n):
                continue
            for param in m.parameters():
                param.requires_grad = True

    def embed(self, string):
        encoding = self.tokenizer(string, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)

        if input_ids.size(-1) > 512:
            raise TooManyTokens("too many tokens:", input_ids.size(-1))
        out = self.model(input_ids=input_ids)

        last_hidden_state = out["last_hidden_state"]
        return last_hidden_state
    
    def forward(self, string, **kwargs):
        emb = self.embed(string, **kwargs)
        return emb
    

class TooManyTokens(Exception):
    pass


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)