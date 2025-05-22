This directory contains the code for localizing audio from the speaker test.
Please view this readme in a wide screen for correct formatting.

The code is structured as follows:

├── inputs/
│   ├── aru_coords.csv                              CSV with the x,y (in UTM) positions of the recorders
│   ├── 2024-02-27_opso-0.10.1_nfwf-v3.0.model      CNN model object (Trained with Opensoundscape (v0.10.1))
│   └── All_sp_spotmaps.csv                         CSV of the digitized spotmapping observations. Coordinates 
│                                                   should be considered approximations with human error.
│                                                     
├── output_data/
│   ├── 2022-05-13_clustered_localizations.csv  CSV of birds localized by localization framework for specific date.
│   ├── 2022-05-14_clustered_localizations.csv  CSV of birds localized by localization framework for specific date.
│   ├── 2022-05-15_clustered_localizations.csv  CSV of birds localized by localization framework for specific date.
│   ├── 2022-05-16_clustered_localizations.csv  CSV of birds localized by localization framework for specific date.
│   ├── All_localizations.csv                   CSV of all birds localized on all 4 above dates.
│   ├── preds.csv                               Figure of the positions
├── CONFIG.py                       Contains all localization parameters 
├── 1_localize_preds.py             Script for running the localization framework
├── 2_cluster_localizations.py      Script for clustering (using DBScan) the localized spatialevents
├── 3_plot_localized_positions.py   Script for plotting 
└── 4_k_cross_r.ipynb               Notebook for calculating the Kcross between automatic pipeline and spotmapping

