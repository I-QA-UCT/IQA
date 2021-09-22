import random
import json


def create_dataset(experiment,seed=12345):
    """
    Function for creating an online validation dataset of TextWorld games for the QAit task.
    """
    filename = "./decision_transformer/data/" + experiment
    print("="*50)
    print(f"Splitting {experiment}")
    print("="*50)
    rng = random.Random(seed)

    indicies = {"high" : set(), "moderate" : set(), "low" : set()}

    with open(filename+".json") as old_data, open(filename+"_split.json","w") as new_data:

        for episode_no,sample_entry in enumerate(old_data):
                        
            episode = json.loads(sample_entry)

            if episode["total_reward"] >= 1:
                indicies["high"].add(episode_no)
            elif  0.5 <= episode["total_reward"] < 1:
                indicies["moderate"].add(episode_no)
            elif 0 <= episode["total_reward"] < 0.5:
                indicies["low"].add(episode_no)
            else:
                raise NotImplementedError
        
            total_size = episode_no
        
        total_size+=1
        print("="*50)
        print("Original dataset proportions:")
        print("="*50)
        print(f'Size: {total_size}')
        print(f"Reward greater than 1.0: {len(indicies['high'])/total_size}")
        print(f"Reward greater than 0.5 and less than 1.0: {len(indicies['moderate'])/total_size}")
        print(f"Reward greater or equal to 0.0 and less than 0.5: {len(indicies['low'])/total_size}")

        minimum_size = len(min(indicies["high"], indicies["moderate"], indicies["low"], key=lambda x: len(x)))

        indicies["high"] = set(rng.sample(indicies["high"], minimum_size ))
        indicies["moderate"] = set(rng.sample(indicies["moderate"],minimum_size))
        indicies["low"] = set(rng.sample(indicies["low"], minimum_size))
        
        total_size = len(indicies["high"]) + len(indicies["high"]) + len(indicies["high"])
        print("="*50)
        print("New dataset proportions:")
        print("="*50)
        print(f'Size: {total_size}')
        print(f"Reward greater than 1.0: {len(indicies['high'])/total_size}")
        print(f"Reward greater than 0.5 and less than 1.0: {len(indicies['moderate'])/total_size}")
        print(f"Reward greater or equal to 0.0 and less than 0.5: {len(indicies['low'])/total_size}")
        
        old_data.seek(0)
        valid_inds = indicies["high"] | indicies["moderate"] | indicies["low"]
        for episode_no,sample_entry in enumerate(old_data):

            if episode_no in valid_inds:
                episode = json.loads(sample_entry)

                print(json.dumps(episode),file=new_data)

if __name__ == "__main__":
    map_types = [ "fixed_map"]
    question_types = ["location","attribute","existence"]

    for m in map_types:
        for q in question_types:
            create_dataset(m +"/"+q+"-500",seed=12345)

