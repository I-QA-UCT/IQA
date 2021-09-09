import plotly.express as px
import jsonlines
import pandas as pd
import numpy as np

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
            if episode_counter>220:
                break
            obj["Episode"] = episode_counter
            episode_counter+=1
            dataframe.append(obj)

    return pd.DataFrame(dataframe)
        
if __name__ == "__main__":

    FILENAMES = ["../experiments/a2c_existence_500_random.json","../qaitlogs/DQN_Existence_rand1_seed123_size500.json","../qaitlogs/DDQN_Existence_rand1_seed123_size500.json","../qaitlogs/Rainbow_Existence_rand1_seed123_size500.json"]
    LINES = ["REINFORCE with baseline","DQN","DDQN","Rainbow"]
    
    SMOOTH = True
    SMOOTHING_DEGREE = 1
    
    dataframes = []
    for filename in FILENAMES:
        dataframes.append(read_file(filename))
    
    fig = px.line(title="Training Sufficient Information")
    fig.update_xaxes(title_text='Episodes')
    fig.update_yaxes(title_text='Sufficient Information')
    
    for i,dataframe in enumerate(dataframes):
        fig.add_scatter(x=dataframe["Episode"],y=smoothTriangle(dataframe["sufficient info"],SMOOTHING_DEGREE) if SMOOTH else dataframe["sufficient info"],mode="lines",name=LINES[i])
   
    fig.show()
    