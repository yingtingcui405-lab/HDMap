import numpy as np 
import copy

def reduce_change_scores(lane_segments, reduced_set):
    for ls in lane_segments:
        ls["change_score"] = [score for score in ls["change_score"] if score in reduced_set]
        if len(ls["change_score"])==0: ls["change_score"] = [0]  
    return lane_segments    

def load_annotations(avm):
    all_pedestrian_crossings_gt = []
    all_pedestrian_crossings_st = []
    all_lane_segments_gt  = []
    all_lane_segments_st  = []
    for p in avm["pedCrossings"].values():
        pc_old = {"points": [np.array([e["x"], e["y"]]) for e in p["old"]["edges"]] + 
                            [np.array([p["old"]["edges"][0]["x"], p["old"]["edges"][0]["y"]])],
                    "id": p["old"]["id"],
                    "change_score": p["changes"]} if p["old"] != None else None

        pc_new = {"points": [np.array([e["x"], e["y"]]) for e in p["new"]["edges"]] + 
                            [np.array([p["new"]["edges"][0]["x"], p["new"]["edges"][0]["y"]])],
                    "id": p["new"]["id"], 
                    "change_score": p["changes"]} if p["new"] != None else None
        obj_gt = copy.deepcopy(pc_old)
        if p["changes"] == [0]:  # pc was here, is here
            all_pedestrian_crossings_gt.append(obj_gt) 
            obj_st = copy.deepcopy(pc_old)
            obj_st["change_score"] = []
            all_pedestrian_crossings_st.append(obj_st)      
      
        else: 
            obj_st = copy.deepcopy(pc_new)
            if p["changes"]==[1]: # pc was present in stale map, is no longer here                   
                obj_gt = copy.deepcopy(pc_new)
                all_pedestrian_crossings_gt.append(obj_gt) 
                obj_st["change_score"] = []
                all_pedestrian_crossings_st.append(obj_st)      
            elif p["changes"]==[2]:# pc wasn't here, but is present in up-to-date map 
                all_pedestrian_crossings_gt.append(obj_gt)     
            elif p["changes"]==[3]: # pc was here, is still here, but slight geometry adaptations have happened
                all_pedestrian_crossings_gt.append(obj_gt) 
                obj_st["change_score"] = []
                all_pedestrian_crossings_st.append(obj_st) 
    
    
    # add all lane segments to gt and add deletions because of explainability  
    all_lane_segments_gt = [
        copy.deepcopy({**l["old"], "change_score": l["changes"]})
        for l in avm["laneSegments"].values()
        if l["old"] is not None
    ] + [
        copy.deepcopy({**l["new"],  "change_score": l["changes"]})
        for l in avm["laneSegments"].values()
        if  l["changes"] == [1]
        ]

    # deletions in gt should only be connected to other deletions
    look_up_dir = {id["id"].split("-")[1]:idx for idx, id in enumerate(all_lane_segments_gt)}
    for segment in all_lane_segments_gt:
        if segment["change_score"]==[1]: 
            for pred in segment["predecessors"][:]:
                if not all_lane_segments_gt[look_up_dir[str(pred)]]["change_score"]==[1]:
                    segment["predecessors"].remove(pred)       
            for suc in segment["successors"][:]:
                if not all_lane_segments_gt[look_up_dir[str(suc)]]["change_score"]==[1]:
                    segment["successors"].remove(suc)
    
    all_lane_segments_st = [
            copy.deepcopy({**l["old"], "change_score": []})       # copy gt segments if no change occured 
            if l["changes"]==[0]
            else copy.deepcopy({**l["new"], "change_score": []})  # copy new segments otherwise
            for l in avm["laneSegments"].values()
            if l["changes"]!=[2]                                  # but only if the change is not a deletion
    ] 
    
    return all_pedestrian_crossings_gt, all_pedestrian_crossings_st, all_lane_segments_gt, all_lane_segments_st,
    
