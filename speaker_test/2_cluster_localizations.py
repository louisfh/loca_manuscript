## general imports ##
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import ConvexHull

## opensoundscape imports ##
import opensoundscape
from opensoundscape.localization import SynchronizedRecorderArray
assert opensoundscape.__version__ == "0.10.2"

# all the parameters used for clustering are in the CONFIG file
# the implementation of DBScan is also in the CONFIG
import CONFIG

experiments = ["n2_t1", "n4_t1"]

def check_if_point_in_hull(x,y, hull, margin=-10, eps=1e-6):
    """
    Checks if a point is inside a convex hull, with a margin of error.
    Returns True if the point is inside the hull, False otherwise.
    """
    coords = np.array([x, y])
    return all(np.dot(eq[:-1], coords) + eq[-1] <= margin + eps for eq in hull.equations)

def remove_points_outside_convex_hull(df_points : pd.DataFrame,
                                      aru_coords: pd.DataFrame, 
                                      margin=-5, eps=1e-6):
    """
    Removes points from a DataFrame that are outside a convex hull.
    """
    # make a hull drom the points of the aru_coords. This is a df with columns x and y
    hull = ConvexHull(aru_coords)

    # take the df_points (which has columns x and y), and convert it to an array of shape (n, 2)
    df_points = df_points[["x", "y"]]
    # apply check_if_point_in_hull to each row of df_points
    mask = df_points.apply(lambda row: check_if_point_in_hull(row.x, row.y, hull, margin, eps), axis=1)
    return df_points[mask]

for experiment in experiments:  
    # read in the 'localized events'
    # the output of the previous step
    with open(f"output_data/{experiment}_localized_events.pkl", "rb") as f:
        localized_events = pickle.load(f)
    
    #### cluster the localized events ####
    # we will cluster only events of the same species, and in the same time-window 
    # this resolves the redundancy of creating a new localized event for each recorder
    # and also lets you resolve multiple individuals of the same species at different positions
    species = list(set([e.class_name for e in localized_events]))
    start_times = set([e.start_time for e in localized_events])
    
    cluster_df = pd.DataFrame() # for storing the clustered events

    print(f"Clustering... parameters: rms_threshold = {CONFIG.rms_threshold}, eps = {CONFIG.eps}, min_samples = {CONFIG.min_samples}")
    for start in start_times:
        for sp in species:
            events = [i for i in localized_events if i.class_name == sp and i.start_time == start] # only the same species and same time-window               
            clusters = CONFIG.dbscan_cluster(events, rms_threshold=CONFIG.rms_threshold, eps=CONFIG.eps, min_samples=CONFIG.min_samples) 
            if clusters is None: # happens if there are no 'clusters' of repeated, close localizations
                continue
            for cluster in clusters: # otherwise, save each distinct cluster
                cluster_dict = {"start_time":start, "species":sp, "x":cluster[0], "y":cluster[1]}
                cluster_df = pd.concat([cluster_df, pd.DataFrame([cluster_dict])], ignore_index = True)
    cluster_df = remove_points_outside_convex_hull(cluster_df, CONFIG.aru_coords, margin = 10) 
    cluster_df.to_csv(f"output_data/{experiment}_clustered_localizations.csv")
