
import json
import copy
import argparse



filename="./data/random_rollouts_attribute_500_random_map_AC"
with open(filename+".json") as old_data, open(filename+"_cleaned.json","w") as new_data:

    for episode_no,sample_entry in enumerate(old_data):
                    
        episode = json.loads(sample_entry)

        
        # trajectory["mask"] = episode["mask"]
        for i in range(len(episode["steps"])-1):
            
            game_step = episode["steps"][i]
            # game_step["state"].replace("<s>","").replace("</s>","").replace("<|>","")

            # Get the action, modifier, object triple
            episode["steps"][i]["command"] = copy.deepcopy(episode["steps"][i+1]["command"])
            # act, mod, obj = command["action"], command["modifier"], command["object"]
        del episode["steps"][-1]

        print(json.dumps(episode),file=new_data)
        new_data.flush()
