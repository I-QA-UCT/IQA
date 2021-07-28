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
        pass

    def load_files(self):
        pass