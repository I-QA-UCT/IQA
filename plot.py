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
            if episode_counter>200:
                break
            obj["Episode"] = episode_counter
            episode_counter+=1
            dataframe.append(obj)

    return pd.DataFrame(dataframe)
        
if __name__ == "__main__":

    FILENAME_1 = "../experiments/a2c_existence_500_random.json"
    LINE_1 = "a2c without inverse semantics"
    FILENAME_2 = "../experiments/a2c_existence_500_random_inverse_semantics.json"
    LINE_2 = "a2c with inverse semantics"

    SMOOTH = True
    SMOOTHING_DEGREE = 1

    df1 = read_file(FILENAME_1)
    df2 = read_file(FILENAME_2)

    fig = px.line(title="Training Accuracy")
    fig.update_xaxes(title_text='Episodes')
    fig.update_yaxes(title_text='Accuracy')
    fig.add_scatter(x=df1["Episode"],y=smoothTriangle(df1["qa"],SMOOTHING_DEGREE) if SMOOTH else df1["qa"],mode="lines",name=LINE_1)
    fig.add_scatter(x=df2["Episode"],y=smoothTriangle(df2["qa"],SMOOTHING_DEGREE) if SMOOTH else df2["qa"],mode="lines",name=LINE_2)
    fig.show()
    