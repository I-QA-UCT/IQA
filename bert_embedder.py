from transformers import AutoTokenizer, AutoModel

from torch import nn
import os


class BertEmbedder(nn.Module):
    """
    A pre-trained BERT embedder (introduced in https://arxiv.org/abs/1908.08962) wrapper class.
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
        """
        Sets the tuneable parameters inthe BERT model. In the case of this extention, the set of trainable parameters are set to none.
        """
        def is_in_fine_tune_list(name):
            if name == "":  # full model is off by default
                return False

            for l in self.bert_fine_tune_layers:
                if l in name:
                    return True
            return False
            
        for param in self.model.parameters():
            #all params are turned off. then we selectively reactivate grads
            param.requires_grad = False
            
        for n, m in self.model.named_modules():
            if not is_in_fine_tune_list(n):
                continue
            for param in m.parameters():
                param.requires_grad = True

    def embed(self, string):
        """
        Embeds a natural lanuage word/phrase into a vector representation.
        :param string: The natural langauge word/phrase.
        :return last_hidden_state: The vector representation.
        """
        encoding = self.tokenizer(string, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)

        if input_ids.size(-1) > 512:
            raise TooManyTokens("too many tokens:", input_ids.size(-1))
        out = self.model(input_ids=input_ids)

        last_hidden_state = out["last_hidden_state"]
        return last_hidden_state

class TooManyTokens(Exception):
    pass


def num_params(model):
    """
    Counts the trainable parameters in a model.
    :param model: The BERT model.
    :return: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)