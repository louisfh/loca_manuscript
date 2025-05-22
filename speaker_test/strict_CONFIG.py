import pandas as pd

# the ARU coordinates
aru_coords = pd.read_csv("inputs/aru_coords.csv", index_col = 0)

# localization parameters
min_n_receivers = 6
max_receiver_dist = 150
cc_threshold = 0.02
rms_threshold = 2
cc_filter = "phat"
bandpass_ranges = {"AcadianFlycatcher": [3000,6000], 
                   "Black-and-whiteWarbler": [6500, 9500], 
                   "Black-throatedBlueWarbler": [3400, 6000], 
                   "Black-throatedGreenWarbler": [4000, 6000], 
                   "HoodedWarbler": [3500, 7000],
                   "ScarletTanager": [2000, 4000]}

# all the settings used for clustering
eps = 3
min_samples = 5

### Define a DBScan clustering function ###
def dbscan_cluster(events, rms_threshold, eps, min_samples):
    """
    Cluster a list of SpatialEvents using DBSCAN.
    Args:
        events: list of SpatialEvents
        rms_threshold: float
            The maximum TDOA residual RMS of the events to include in the clustering.
        eps: float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    Returns:
        The mean positions of the clusters.
    """
    from sklearn.cluster import DBSCAN
    import numpy as np
    # get the positions of the events
    positions = [e.location_estimate for e in events if e.residual_rms < rms_threshold]
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    except ValueError: #if there are no positions
            return None
    # get the cluster labels
    labels = clustering.labels_
    # return the mean positions of every cluster
    return [np.mean([p for i, p in enumerate(positions) if labels[i] == label], axis=0) for label in set(labels) if label != -1]  
