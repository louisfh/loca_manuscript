## General imports ##
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

## OpenSoundscape imports ##
import opensoundscape
from opensoundscape.localization import SynchronizedRecorderArray
assert opensoundscape.__version__ == "0.10.2"

# the CONFIG.py file contains all the localization parameters for the speaker test
import CONFIG 

experiments = ["n2_t1", "n4_t1"] # the two trials where all speakers worked

for experiment in experiments:
    # pull out the detections for this experiment
    detections = pd.read_csv(f"./inputs/local_detections_{experiment}.csv", index_col = [0,1,2])
    detections_experiment = detections[detections.index.get_level_values(0).str.contains(experiment)]

    # pull out the aru_coords for this experimenti
    aru_coords = CONFIG.aru_coords
    aru_coords = aru_coords[aru_coords.index.str.contains(experiment)]

    print(f"Localizing detections for {experiment}")
    array = SynchronizedRecorderArray(aru_coords)

    # the unlocalized events are those that do not meet the localization criteria
    # we don't use these, but allow you to return them to investigate 
    # if the localization criteria are too strict
    localized_events, unlocalized_events = array.localize_detections(detections,
                                                                 min_n_receivers=CONFIG.min_n_receivers,
                                                                max_receiver_dist=CONFIG.max_receiver_dist,
                                                                cc_threshold=CONFIG.cc_threshold,
                                                                cc_filter=CONFIG.cc_filter,
                                                                bandpass_ranges=CONFIG.bandpass_ranges,
                                                                return_unlocalized=True,
                                                                num_workers = 16)

    print(f"{len(localized_events)} localized_events created")
    # we pickle and save these the output, as it is computationally costly to repeat the cross-correlation
    # if you want to e.g. try using a different cc_threshold, you can filter the localized
    # events by the new cc_threshold, without re-running the cross-correlation
    with open(f"output_data/{experiment}_localized_events.pkl", "wb") as f:
        pickle.dump(localized_events, f)
