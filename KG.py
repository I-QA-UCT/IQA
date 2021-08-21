from collections import OrderedDict
import networkx as nx
import json
import requests
import numpy as np
from nltk import sent_tokenize, word_tokenize
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from bert_embedder import BertEmbedder
import os 

def init_openIE():
    try:
        cmd = 'cd stanford-corenlp-4.2.2 && java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > /dev/null 2>&1 &'
        os.system(cmd)
    except:
        print("Error setting up Stanford OpenIE.")
    print("Stanford Open IE initialised and listening on port 9000.")

def openIE(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D", #{"annotators": "openie"}
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response

class SupplementaryKG(object):

    def __init__(self, use_cuda, use_bert, bert_size, device):
    
        self.use_bert = use_bert
        self.vocab, self.vocab_er = self.load_files()
        self.visible_state = "" #What states and relations are currently visible to the agent
        self.room = "" #Room at the centre of KG
        
        self.graph_state = nx.DiGraph()
        self.adj_matrix = None #Matrix of adjacent nodes in the graph (used as part of attention representation for GAT)
        self.graph_state_rep = []  #Representation attention between entities 

        self.use_cuda = use_cuda
        self.entities = OrderedDict()
        self.entity_nums = 0

        self.device = device
        self.bert = BertEmbedder(bert_size, [], self.device)
        self.state_ent_emb = None
        self.embeds = []

        init_openIE()

    def load_files(self):
        vocab = {}
        i = 0
        with open('vocabularies/word_vocab.txt', 'r') as file_output:
            for i, line in enumerate(file_output):
                vocab[line.strip()] = i

        if self.use_bert:
            return vocab, None
        else:
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
            
            return vocab, entity_relation_dict

    def set_entities(self, path):
        file_in = open(path, 'r')
        self.entities = eval(file_in.readline())
        self.entity_nums = len(self.entities.keys())

    def visualize(self):
        pos = nx.spring_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=200, font_size=10)
        plt.show()

    def update_state(self, visible_state, prev_action=None):

        if prev_action == 'restart':
            self.graph_state.clear()

        #Format visible state, and set to self
        visible_state = visible_state.split('-')
        if len(visible_state) > 1:
            visible_state = visible_state[2]
        self.visible_state = str(visible_state)
        
        rules = []
        try:
            #Run visible state through Standford OpenIE and extract triple into list of rules
            sents = openIE(self.visible_state)['sentences']
            for obv in sents:
                triple = obv['openie']
                for tr in triple:
                    subject, relation, object = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower()

                    #Change subject to be singularised i.e. from 1st to 2nd person
                    if subject == 'we':
                        subject = 'you'
                        if relation == 'are in':
                            relation = "'ve entered"

                    if subject == 'it':
                        break

                    rules.append((subject, relation, object))
        except:
             print("Error: OpenIE")

        # print(rules)
        prev_remove = []
        room = ""
        room_set = False 
        remove = []    
        link = []

        #Iterate through all relations, set current room, add links and add all relations to previously removed or removed list
        for rule in rules:
            subject, relation, object = rule
            if 'entered' in relation or 'are in' in relation:
                prev_remove.append(relation)
                if not room_set: #Set current room if not already set
                    room = object
                    room_set = True
            if 'should' in relation:
                prev_remove.append(relation) 
            if 'see' in relation or 'make out' in relation:
                link.append((relation, object)) #Add link between relation and object to list
                remove.append(relation)
        
        prev_room = self.room
        self.room = room
        add_rules = []
        directions = ['north', 'south', 'east', 'west']

        #Provide previous room in KG if previous action is available
        if prev_action is not None:
            for dir in directions:
                if dir in prev_action and self.room != "":
                    add_rules.append((prev_room, dir + ' of', room))
        prev_room_subgraph = None
        prev_you_subgraph = None

        for sent in sent_tokenize(self.visible_state):
            if 'exit' in sent or 'entranceway' in sent:
                for dir in directions:
                    if dir in sent:
                        rules.append((self.room, 'has', 'exit to ' + dir)) 
                    if prev_room != "":
                        #Get previous room and previous "you" subgraphs
                        graph_copy = self.graph_state.copy()

                        if ('you', prev_room) in graph_copy.edges:
                            graph_copy.remove_edge('you', prev_room)

                        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]

                        for con_c in con_cs:
                            if prev_room in con_c.nodes:
                                prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
                            if 'you' in con_c.nodes:
                                prev_you_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
        
        #Add liks and not removed rules to the the add_rule list
        for l in link:
            add_rules.append((room, l[0], l[1]))
        
        for rule in rules:
            subject, relation, object = rule
            if relation not in remove:
                add_rules.append(rule)
        
        edges = list(self.graph_state.edges)

        # print("add", add_rules)

        #Remove edges from KG that are no longer needed
        for edge in edges:
            relation = self.graph_state[edge[0]][edge[1]]['rel']
            if relation in prev_remove:
                self.graph_state.remove_edge(*edge) 
        
        #Remove previous "you" edges from KG
        if prev_you_subgraph is not None:
            self.graph_state.remove_edges_from(prev_you_subgraph.edges)
        
        #Update KG to include new edges and nodes
        if self.use_bert:
           for rule in add_rules:
                u = '_'.join(str(rule[0]).split())
                v = '_'.join(str(rule[2]).split())
                if u not in self.entities.keys():
                    self.entities[u] = self.entity_nums
                    self.entity_nums += 1

                if v not in self.entities.keys():
                    self.entities[v] = self.entity_nums
                    self.entity_nums += 1
                
                if u != 'it' and v != 'it':
                        self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])
        else:
            for rule in add_rules:
                u = '_'.join(str(rule[0]).split())
                v = '_'.join(str(rule[2]).split())
                if u in self.vocab_er['entity'].keys() and v in self.vocab_er['entity'].keys():
                    if u != 'it' and v != 'it':
                        self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])

        # print("pre", self.graph_state.edges)

        if prev_room_subgraph is not None:
            self.graph_state.add_edges_from(prev_room_subgraph.edges)

        return
    
    def state_ent_emb_bert(self):

        entities = list(self.entities.keys())
        num_current = len(self.embeds)
        for i in range(num_current, len(entities)):
            graph_node_text = entities[i].replace('_', ' ')
            node_embedding = self.bert.embed(graph_node_text).squeeze(0)

            #Summarizer
            node_embedding= node_embedding.mean(dim=0) 
            self.embeds.append(node_embedding)
        
        self.state_ent_emb = torch.nn.Embedding.from_pretrained(torch.stack(self.embeds), freeze=True)
    
    def get_state_representation_bert(self):
        
        self.adj_matrix = [[],[]]

        for source, target in self.graph_state.edges:

            source = '_'.join(str(source).split()) #Make source and target nodes look the same OpenIE entities
            target = '_'.join(str(target).split())

            source_id = self.entities[source]    
            target_id = self.entities[target]
            self.adj_matrix[0].append(source_id) 
            self.adj_matrix[1].append(target_id)

        edge_index = torch.tensor(self.adj_matrix, dtype=torch.long, device = self.device)

        self.state_ent_emb_bert()

        data = Data(x=self.state_ent_emb.weight, edge_index=edge_index).to(self.device)

        return data

    def get_state_representation(self):
        
        result = []
        self.adj_matrix = np.zeros((len(
            self.vocab_er['entity']), len(self.vocab_er['entity']))) #Set matrix to zeros

        for source, target in self.graph_state.edges:
            source = '_'.join(str(source).split()) #Make source and target nodes look the same vocab_er
            target = '_'.join(str(target).split())

            #Ignore words not in discovered by agent in entity_relation_collection.py
            if source not in self.vocab_er['entity'].keys() or target not in self.vocab_er['entity'].keys():
                break

            source_id = self.vocab_er['entity'][source]    
            target_id = self.vocab_er['entity'][target]      
            self.adj_matrix[source_id][target_id] = 1 #Update matrix representation to reflect relation between source and target

            result.append(self.vocab_er['entity'][source])
            result.append(self.vocab_er['entity'][target])

        return list(set(result))

    def step(self, visible_state, prev_action=None):
        self.update_state(visible_state, prev_action)
        if self.use_bert:
            self.graph_state_rep = self.get_state_representation_bert()
        else:
            self.graph_state_rep = self.get_state_representation(), torch.IntTensor(self.adj_matrix, device =self.device)


if __name__ == '__main__':

    test = SupplementaryKG()
