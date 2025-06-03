import numpy as np
import warnings
import os

#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def process_hstates_and_scalars(dir, idxes=None, offset=500000):
    '''

    :param dir: Name of the directory containing the log files
    :param idxes: Log idxes to load. By default, everything
    :param offset: Hacky offset for distinguishing validation logs.
    :return: dicts of hstates, scalars
    '''
    hstates_all = {}
    scalars_all = {}
    for fname in os.listdir(dir):
        if 'hstates' in fname:
            fname_new = fname.replace('_', '.')
            split = fname_new.split('.')
            idx = int(split[1])
            # HACK: the way we are naming validation logs right now is just adding an offset of 500k to the first index of the filename. For such logs, we use the second (batch) index as the key. This might not work in the future depending on how many logs we decide to store during training.
            if idx > offset and len(split) > 3:
                idx = int(split[2])
            if idxes is None or idx in idxes:
                hstates = np.loadtxt(open(os.path.join(dir, fname), 'rb'), delimiter=',', skiprows=0)
                hstates_all[idx] = hstates
        if 'scalars' in fname:
            fname_new = fname.replace('_', '.')
            split = fname_new.split('.')
            idx = int(split[1])
            if idx > offset and len(split) > 3:
                idx = int(split[2])
            if idxes is None or idx in idxes:
                scalars = pd.read_csv(os.path.join(dir, fname), header=0)
                scalars_all[idx] = scalars
                scalars_all[idx]['episode'] = scalars_all[idx]['done'].cumsum()
    return hstates_all, scalars_all


def label_future_and_past(scalars, interval=20, arena_size=50):
    '''

    :param scalars: Scalar df to add future/past labels in-place
    :param interval: The interval of temporal distance into future/past to label
    :param arena_size: size of grid. Leave at default
    :return:
    '''
    scalars['future_dist_t{}'.format(interval)] = np.nan
    scalars['future_angle_t{}'.format(interval)] = np.nan
    scalars['future_pos_x_t{}'.format(interval)] = np.nan
    scalars['future_pos_y_t{}'.format(interval)] = np.nan

    scalars['past_dist_t{}'.format(interval)] = np.nan
    scalars['past_angle_t{}'.format(interval)] = np.nan
    scalars['past_pos_x_t{}'.format(interval)] = np.nan
    scalars['past_pos_y_t{}'.format(interval)] = np.nan
    for ep, scalar_df in scalars.groupby('episode_id'):
        ep_len = len(scalar_df)
        idxes_now = scalar_df.index
        idxes_fut = scalar_df.index[interval:]
        idxes_past = scalar_df.index[:-interval]
        x_now = scalars.iloc[idxes_now]['player_position_x'].to_numpy()
        y_now = scalars.iloc[idxes_now]['player_position_y'].to_numpy()

        x_fut = scalars.iloc[idxes_fut]['player_position_x'].to_numpy()
        y_fut = scalars.iloc[idxes_fut]['player_position_y'].to_numpy()

        x_past = scalars.iloc[idxes_past]['player_position_x'].to_numpy()
        y_past = scalars.iloc[idxes_past]['player_position_y'].to_numpy()
        # process future
        if len(idxes_now) == len(idxes_fut) + interval:
            idx_now_for_fut = idxes_now[:-interval]
            displacement = np.array([x_fut - x_now[:-interval], y_fut - y_now[:-interval]]).T
            dist = np.linalg.norm(displacement, axis=-1, ord=1)
            angle = np.arctan2(displacement[:, 1], displacement[:, 0])
            scalars['future_dist_t{}'.format(interval)].iloc[idx_now_for_fut] = dist
            scalars['future_angle_t{}'.format(interval)].iloc[idx_now_for_fut] = angle
            scalars['future_pos_x_t{}'.format(interval)].iloc[idx_now_for_fut] = x_fut
            scalars['future_pos_y_t{}'.format(interval)].iloc[idx_now_for_fut] = y_fut

        # process past
        if len(idxes_now) == len(idxes_past) + interval:
            idx_now_for_past = idxes_now[interval:]
            displacement = np.array([x_past - x_now[interval:], y_past - y_now[interval:]]).T
            dist = np.linalg.norm(displacement, axis=-1, ord=1)
            angle = np.arctan2(displacement[:, 1], displacement[:, 0])
            scalars['past_dist_t{}'.format(interval)].iloc[idx_now_for_past] = dist
            scalars['past_angle_t{}'.format(interval)].iloc[idx_now_for_past] = angle
            scalars['past_pos_x_t{}'.format(interval)].iloc[idx_now_for_past] = x_fut
            scalars['past_pos_y_t{}'.format(interval)].iloc[idx_now_for_past] = y_fut
