import pandas as pd
import numpy as np
speaker_coords = pd.read_csv("inputs/speaker_coords.csv", index_col=0)
truth = pd.read_csv("inputs/truth.csv")
n2_t1 = pd.read_csv("output_data/n2_t1_clustered_localizations.csv", index_col=0)
n2_t1["experiment"] = "n2_t1"
n4_t1 = pd.read_csv("output_data/n4_t1_clustered_localizations.csv", index_col=0)
n4_t1["experiment"] = "n4_t1"

all_localizations = pd.concat([n2_t1, n4_t1], ignore_index=True)

def find_nearest_speaker(x_pos, y_pos, speaker_coords):
    """
    Returns:
        the x and y coordinates of the nearest speaker
    """
    # if NaN return NaN. Ensures we don't try and find nearest speaker if we don't have a position.
    if np.isnan(x_pos) or np.isnan(y_pos):
        return np.array([np.nan, np.nan])
    else:
        distances = np.linalg.norm(speaker_coords - np.array([x_pos, y_pos]), axis=1)
        # return the coords of the nearest speaker
        nearest_speaker = speaker_coords.iloc[np.argmin(distances)].values
        return nearest_speaker

def calculate_error_and_recall(all_localizations, truth):
    """
    Takes in a results dataframe and returns:
        - mean error
        - recall
    """
    for i,row in all_localizations.iterrows():
        nearest_speaker = find_nearest_speaker(row["x"], row["y"], speaker_coords)
        all_localizations.loc[i, "nearest_speaker_x"] = nearest_speaker[0]
        all_localizations.loc[i, "nearest_speaker_y"] = nearest_speaker[1]
        all_localizations.loc[i, "error"] = np.linalg.norm(nearest_speaker - np.array([row["x"], row["y"]]))

    # now for each row in truth, see if it was localized
    for i, row in truth.iterrows():
        experiment = row["experiment"]
        # find the matching row in localizations. this should have the same species name and start_time
        start = row["start_time"]
        
        # species is the name of the column with a 1.0 in it
        species = row[row == 1.0].index[0]
        # find if there is a row in localizations that matches this
        matching_rows = all_localizations[
            (all_localizations["start_time"] == start) & 
            (all_localizations["species"] == species) &
            (all_localizations["experiment"] == experiment)]
        if len(matching_rows) == 0:
            truth.loc[i, "localized"] = False
            truth.loc[i, "error"] = np.nan
        else:
            truth.loc[i, "localized"] = True
            truth.loc[i, "error"] = matching_rows["error"].values[0]
        
    mean_error = truth["error"].mean()
    recall = truth["localized"].sum() / len(truth)
    return mean_error, recall

mean_error, recall = calculate_error_and_recall(all_localizations, truth)
print(f"Mean error: {mean_error}")
print(f"Recall: {recall}")

# count how many have errors below 5
num_localized = len(truth[truth["localized"] == True])
num_localized_below_5 = len(truth[(truth["localized"] == True) & (truth["error"] < 5)])
print(f"Number of localizations: {num_localized}")
print(f"Number of localizations with error below 5: {num_localized_below_5}")
print(f"Percentage of localizations with error below 5: {num_localized_below_5 / num_localized * 100:.2f}%")