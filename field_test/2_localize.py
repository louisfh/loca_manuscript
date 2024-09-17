import time
import pandas as pd
import numpy as np
import opensoundscape
from opensoundscape.localization import SynchronizedRecorderArray
from sklearn.cluster import DBSCAN
assert opensoundscape.__version__ == "0.10.2"

import CONFIG # contains all the localization parameters

##### Begin Script #####
# load preds and aru_coords
all_date_preds = pd.read_csv("./output_data/preds.csv", index_col = [0,1,2])
all_date_preds = all_date_preds[CONFIG.bandpass_ranges.keys()] # get just the species of interest
all_date_aru_coords = pd.read_csv("./inputs/aru_coords.csv", index_col = 0)

# drop the preds to only a specific date
dates = ["2022-05-13", "2022-05-14", "2022-05-15", "2022-05-16"]

all_dates_df = []
for date in dates:
    print("Localizing", date, "...")
    init_time = time.time()

    # filter preds and aru_coords to only those whose index contains the string date
    preds = all_date_preds[all_date_preds.index.get_level_values(0).str.contains(date)]
    aru_coords = all_date_aru_coords[all_date_aru_coords.index.str.contains(date)]
    
    # for each column in the preds dataframe, consider it a detection if it is above the cnn_threshold
    detections = preds > CONFIG.cnn_threshold

    ##### Do the localization #####
    print("Localizing ...")
    print(f"{sum(detections.sum(axis = 1) > 0)} detections for localization")

    array = SynchronizedRecorderArray(aru_coords)

    localized_events, unlocalized_events = array.localize_detections(detections,
                                                                 min_n_receivers=CONFIG.min_n_receivers,
                                                                max_receiver_dist=CONFIG.max_receiver_dist,
                                                                cc_threshold=CONFIG.cc_threshold,
                                                                cc_filter=CONFIG.cc_filter,
                                                                bandpass_ranges=CONFIG.bandpass_ranges,
                                                                return_unlocalized=True,
                                                                num_workers = 16)
    
    print(f"{len(localized_events)} localized_events created")

    #### cluster the events ####
    start_times = set(preds.index.get_level_values(1))
    # the clusters should be saved in a dataframe for ease of use
    cluster_df = pd.DataFrame()

    print(f"Clustering... parameters: rms_threshold = {CONFIG.rms_threshold}, eps = {CONFIG.eps}, min_samples = {CONFIG.min_samples}")
    for start in start_times:
        for species in preds.columns:
            events = [i for i in localized_events  if i.class_name == species and i.start_time == start]
            clusters = CONFIG.dbscan_cluster(events, rms_threshold=CONFIG.rms_threshold, eps=CONFIG.eps, min_samples=CONFIG.min_samples) 
            if clusters is None:
                cluster_dict = {"start_time":start, "species":species, "x":np.nan, "y":np.nan, "n_events":len(events)}
                cluster_df = pd.concat([cluster_df, pd.DataFrame([cluster_dict])], ignore_index = True)
                continue
            for cluster in clusters:
                cluster_dict = {"start_time":start, "species":species, "x":cluster[0], "y":cluster[1], "n_events":len(events)}
                cluster_df = pd.concat([cluster_df, pd.DataFrame([cluster_dict])], ignore_index = True)
    
    running_time = time.time() - init_time
    print(f"Writing out clustered data. Time elapsed: {running_time:.2f} seconds")
    cluster_df.to_csv(f"output_data/{date}_clustered_localizations.csv")
    cluster_df["date"] = date
    all_dates_df.append(cluster_df)

df = pd.concat(all_dates_df)

# save the dataframe
df.to_csv("output_data/All_localizations.csv", index=False)

# merge all of these for convenience
# pipeline_csvs = list(Path("./output_data").glob("*clustered_localizations.csv"))
# dfs = []
# for date in dates:
#     data = []
#     for csv in pipeline_csvs:
#         date_str  = csv.stem.split("_")[0]
#         if date_str != date:
#             continue
#         df = pd.read_csv(csv)
#         # drop nas
#         for _, row in df.iterrows():
#             data.append([date, row["species"], row["x"], row["y"]])
#     df = pd.DataFrame(data, columns=["date", "species", "x", "y"])
#     dfs.append(df)
# df = pd.concat(dfs)

# df.to_csv("./output_data/All_localizations.csv", index=False)
