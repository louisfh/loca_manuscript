This directory contains the code for localizing audio from the speaker test.
Please view this readme in a wide screen for correct formatting.

The code is structured as follows:

├── inputs/
│   ├── aru_coords.csv              CSV with the x,y (in UTM) positions of the recorders
│   ├── aru_coords.csv              CSV with the x,y (in UTM) positions of the speakers
│   ├── n2_t1_detections.csv        CSV with detections for each recorder (1/0 binary detections) in the n2 trial
│   └── n4_t1_detections.csv        "" Same as above for the n44 trial
├── output_data/
│   ├── n2_t1_localized_events.pkl  The pickled 'SpatialEvents' produced by the localization framework
│   │                               these have been saved to demonstrate how you can save intermediates
│   │                               to avoid repeating the computationally intensive cross-correlation
│   ├── n2_t1_clustered_localizations.csv    CSV of the estimated positions produced by the framework.
│   └── map_of_localizations.png             Figure of the positions
├── CONFIG.py                       Contains all localization parameters 
├── 1_localize_preds.py             Script for running the localization framework
├── 2_cluster_localizations.py      Script for clustering (using DBScan) the localized spatialevents
├── 3_plot_localized_positions.py   Script for plotting 
├── 4_calculate_recall.py           Script for calculating the pipeline's recall.
├── loose_CONFIG.py                 A less strict set of localization parameters (see SI)
├── strict_CONFIG.py                A more strict set of localization parameters (see SI)



