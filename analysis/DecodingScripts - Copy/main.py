import numpy as np
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from process_logs import *
from decode import *

if '__main__' == __name__:
    print('loading files...')
    hstates_all, scalars_all = [], []
    idxes = [1, 3, 5, 9, 10]
    # loading three sets of logs corresponding to three separate model/agent runs
    for i in range(1, 4):
        hstates, scalars = process_hstates_and_scalars('./output_batch_dones{}'.format(i), idxes=idxes)
        hstates_all.append(hstates)
        scalars_all.append(scalars)
    n_models = len(scalars_all)
    n_episodes = len(idxes)
    print('loading complete!')

    print('labeling future...')
    intervals = np.logspace(0, 11, num = 12, base = 2)
    intervals = intervals.astype(int)
    for scalar_df in scalars_all:
        for key in scalar_df:
            for interval in intervals:
                label_future_and_past(scalar_df[key], interval=interval)
    print('labeling complete!')
    auc_vals = np.zeros((n_models, n_episodes, n_episodes))
    #scalar_vars = {'future_pos_x_t16': (0, 25), 'future_pos_y_t16': (0, 25)}
    scalar_vars = {'future_angle_t16': (0.01, round(np.pi,2)+0.01)}
    #scalar_vars = {'melee_on_screen':1.0}

    # Training decoder fully on model 1, validation run 1, then testing on model 1, validation run 3.
    # You can also test and train within the same episode via setting train_frac < 1
    auc, model = decode_variables(hstates_all[0][1], scalars_all[0][1], scalar_vars=scalar_vars, plot=False,
                     perm_test=False, train_frac=1.0, discard_values={},
                     keep_only={}, first_ep_only=True)
    auc, _ = decode_variables(hstates_all[0][3], scalars_all[0][3], scalar_vars=scalar_vars, plot=True,
                     perm_test=False, train_frac=0, discard_values={},
                     keep_only={}, model_pt=model, first_ep_only=True)
    plt.show()
