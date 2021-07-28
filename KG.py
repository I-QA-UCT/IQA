import networkx as nx
import json
import requests
import numpy as np

def openIE(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D", #{"annotators": "openie"}
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response

class SupplementaryKG(object):

    def __init__(self):
        self.graph_state = nx.DiGraph()
        
        self.vocab, self.actions, self.vocab_er = self.load_files()

    def load_files(self):
        vocab = {}
        i = 0
        with open('vocabularies/word_vocab.txt', 'r') as file_output:
            for i, line in enumerate(file_output):
                vocab[line.strip()] = i

        actions = eval(open('act2id.txt', 'r').readline())

        entities = {}
        with open("entity2id.tsv", 'r') as file_output:
            for line in file_output:
                entity, entity_id = line.split('\t')
                entities[entity.strip()] = int(entity_id.strip())

        relations = {}
        with open("relation2id.tsv", 'r') as file_output:
            for line in file_output:
                relation, relation_id = line.split('\t')
                relations[relation.strip()] = int(relation_id.strip())
        
        entity_relation_dict = {'entity': entities, 'relation': relations}
        
        return vocab, actions, entity_relation_dict

    def update_state(self, state, previous_action=None):
        pass

if __name__ == '__main__':
    test = SupplementaryKG()
