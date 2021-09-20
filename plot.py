
import plotly.express as px
import plotly.graph_objects as go
import jsonlines
import pandas as pd
import numpy as np
import os

from plotly.subplots import make_subplots

# TLDEDA001

def smoothTriangle(data, degree, dropVals=False):
    triangle=np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals:
        return smoothed
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def read_file(filename):
    dataframe = []
    episode_counter = 0
    with jsonlines.open(filename) as reader:
        for obj in reader:
            if episode_counter>200:
                break
            obj["Episode"] = episode_counter
            episode_counter+=1
            dataframe.append(obj)

    return pd.DataFrame(dataframe)
        
def get_model_name(filename):
    model = filename[:filename.index("_")]
    
    if model == "a2c":
        model = "REINFORCE with Baseline"
    elif model == 'dqn':
        model = "DQN"

    if filename.find("semantics")>-1:
        model += " w/ Environment Dynamics"

    return model
   

def get_files(directory):
    files = []
    lines = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename[-4:] == "json":
            files.append(filepath)
            lines.append(get_model_name(filename))

    return files, lines

def SortNames(arr,partner):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                partner[j], partner[j + 1] = partner[j + 1], partner[j]
   
if __name__ == "__main__":
    
    SMOOTH = True
    SMOOTHING_DEGREE = 1
    QUESTIONS = ["location","existence","attribute"]
    GAMES= ["500","1"]
    MAPS = ["fixed","random"]
    QA = False
    

    LINE_COLOURS = {"REINFORCE with Baseline" : dict(
                        color='#fa4d56',
                        width=1.5
                    ),"REINFORCE with Baseline w/ Environment Dynamics" : dict(
                        color='#33b1ff',
                        width=1.5
                    ),"DQN w/ Environment Dynamics" : dict(
                        color='#8a3ffc',
                        width=1.5
                    ),"DQN" : dict(
                        color='#6fdc8c',
                        width=1.5
                    ),"DDQN" : dict(
                        color='#d2a106',
                        width=1.5
                    ),"Rainbow" : dict(
                        color='#ff7eb6',
                        width=1.5
                    )}

    for GAME in GAMES:
        fig = make_subplots(rows=2, cols=3)
        
        for i,QUESTION in enumerate(QUESTIONS):
            for j,MAP in enumerate(MAPS):

                try:
                    FILENAMES, LINES = get_files("../experiments/"+QUESTION+"/"+GAME+"/"+MAP)
                except:
                    continue
                
                

                SortNames(FILENAMES,LINES)  

                dataframes = []
                for filename in FILENAMES:
                    dataframes.append(read_file(filename))
                
                # fig = px.line(title="Training " + "Accuracy" if QA else 'Sufficient Information')
                # fig.update_xaxes(title_text='Episodes')
                # fig.update_yaxes(title_text='Accuracy' if QA else 'Sufficient Information')
                
                for k,dataframe in enumerate(dataframes):
                    fig.add_trace(go.Scatter(x=dataframe["Episode"],y=smoothTriangle(dataframe["qa" if QA else "sufficient info"],SMOOTHING_DEGREE) if SMOOTH else dataframe["qa" if QA else "sufficient info"],mode="lines",name=LINES[k],line=LINE_COLOURS[LINES[k]],showlegend=True if j == 0 and i ==0 else False),row=j+1,col=i+1)

        # fig.write_image("plots/"+QUESTION+"_"+GAME+"_"+MAP+".png")
        fig.show()

                