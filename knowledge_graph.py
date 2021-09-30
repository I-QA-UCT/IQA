from collections import OrderedDict
import networkx as nx
import json
import requests
import numpy as np
from nltk import sent_tokenize
import torch
from torch_geometric.data import Data
from bert_embedder import BertEmbedder


class KnowledgeGraph(object):

    def __init__(self, bert_size, device, port):
    
        self.visible_state = ""     #Observation description
        self.room = ""              #Current room the agent is in
        
        self.graph_state = nx.DiGraph() #The actual graph object
        self.edge_indices = None        #Edge indices to be used by attention mechanism in GAT
        self.graph_state_rep = None     #Representation of current KG - will include edge indices and vector representation of entities

        self.entities = OrderedDict()   
        self.entity_nums = 0

        self.device = device
        self.bert = BertEmbedder(bert_size, [], self.device)
        self.embeds = []
        self.bert_lookup = {}

        if bert_size == 'tiny':
            self.bert_size_int = 128
        elif bert_size == 'mini':
            self.bert_size_int = 256
        elif bert_size == 'base':
            self.bert_size_int = 768
        else:
            self.bert_size_int = 512 #Small or medium

        self.port = port

    def openIE(self,sentence):
        """
        Submits an HTTP request to OpenIE.
        :param sentence: The natural lanugage sentence to be parsed.
        :return response: OpenIE response object which includes entities and relations.
        """
        url = "http://localhost:" + str(self.port) + "/"
        querystring = {
            "properties": "%7B%22annotators%22%3A%20%22openie%22%7D", #{"annotators": "openie"}
            "pipelineLanguage": "en"}
        response = requests.request("POST", url, data=sentence, params=querystring)
        response = json.JSONDecoder().decode(response.text)
        return response

    def update_state(self, visible_state, prev_action=None):
        """
        Updates an agent's KG using OpenIE and a set of heuristics. Adapted from Ammanabrolu et al. <https://arxiv.org/pdf/1812.01628.pdf>
        :param visible_state: The observation description provided by TextWorld.
        :param prev_action: The previous action performed by the agent, if any.
        """
        #Format visible state, and set to self
        visible_state = visible_state.split('-')
        if len(visible_state) > 1:
            visible_state = visible_state[2]
        self.visible_state = str(visible_state)

        rules = []

        #Run visible state through Standford OpenIE and extract tuple into list of rules
        sents = self.openIE(self.visible_state)['sentences']
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

        #Add relation to previous room in KG if previous action was movement
        if prev_action is not None:
            for dir in directions:
                if dir in prev_action and self.room != "":
                    add_rules.append((prev_room, dir + ' of', room))
        prev_room_subgraph = None
        prev_you_subgraph = None

        #Identify entities relating to the previous room the agent was in
        for sent in sent_tokenize(self.visible_state):
            if 'exit' in sent or 'entranceway' in sent:
                for dir in directions:
                    if dir in sent:
                        rules.append((self.room, 'has', 'exit to ' + dir)) 
                    if prev_room != "":
                       
                        graph_copy = self.graph_state.copy()

                        if ('you', prev_room) in graph_copy.edges:
                            graph_copy.remove_edge('you', prev_room)

                        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]

                        for con_c in con_cs:
                            if prev_room in con_c.nodes:
                                prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
                            if 'you' in con_c.nodes:
                                prev_you_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
        
        #Add the entities and relations into list
        for l in link:
            add_rules.append((room, l[0], l[1]))
        
        for rule in rules:
            subject, relation, object = rule
            if relation not in remove:
                add_rules.append(rule)
        
        edges = list(self.graph_state.edges)

        #Remove edges from KG that are no longer needed
        for edge in edges:
            relation = self.graph_state[edge[0]][edge[1]]['rel']
            if relation in prev_remove:
                self.graph_state.remove_edge(*edge) 


        #Remove previous "you" edges from KG
        if prev_you_subgraph is not None:
            self.graph_state.remove_edges_from(prev_you_subgraph.edges)
    
        #Update KG to include new edges and nodes
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
    
        #Add previous room entites and relations to graph
        if prev_room_subgraph is not None:
            for edge in list(prev_room_subgraph.edges):
                self.graph_state.add_edge(edge[0], edge[1], rel=prev_room_subgraph[edge[0]][edge[1]]['rel'])

        return
    
    def reset_state(self):
        """
        Resets the KG to be completely empty. This is done when an agent starts a new game.
        """
        self.graph_state.clear()
        self.entities.clear()
        self.entity_nums =0
        self.embeds.clear()

    def state_ent_emb_bert(self):
        """
        Populates a list of embeddings to reflect current entities in the graph. Invokes the BERT embedder to get a vector representation of entities in the KG.
        """
        entities = list(self.entities.keys())
        num_current = len(self.embeds)

        #Only find embeddings for new entities
        for i in range(num_current, len(entities)):
            #Check if new entity's embedding isn't already in the BERT embedding dictionary (reduces calls to BERT embedder)
            if entities[i] in self.bert_lookup:
                self.embeds.append(self.bert_lookup[entities[i]])
            else:
                graph_node_text = entities[i].replace('_', ' ')
                node_embedding = self.bert.embed(graph_node_text).squeeze(0)
                node_embedding= node_embedding.mean(dim=0)
                self.embeds.append(node_embedding)
                self.bert_lookup[entities[i]] = node_embedding
    
    def get_state_representation(self):
        """
        Constructs a state representation of the current KG to be used by the GAT.
        :return data: Data object containing entities' vector representation and a list of edge indices.
        """
        self.edge_indices = [[],[]]

        for source, target in self.graph_state.edges:

            source = '_'.join(str(source).split()) #replace underscores from entity names
            target = '_'.join(str(target).split())

            source_id = self.entities[source]    
            target_id = self.entities[target]
            self.edge_indices[0].append(source_id) 
            self.edge_indices[1].append(target_id)

        edge_index = torch.tensor(self.edge_indices, dtype=torch.long, device = self.device)

        self.state_ent_emb_bert() #Update current state embeddings using BERT
        if len(self.embeds) == 0:
            self.embeds = [torch.zeros(self.bert_size_int, device = self.device)]
        data = Data(x=torch.stack(self.embeds), edge_index=edge_index).to(self.device)

        return data

    def step(self, visible_state,prev_action=None):
        """
        Updates the graph and updates the current graph state representation.
        :param visible_state: The observation description provided by TextWorld.
        :param prev_action: The previous action performed by the agent, if any.
        """
        self.update_state(visible_state, prev_action)
        self.graph_state_rep = self.get_state_representation()
       