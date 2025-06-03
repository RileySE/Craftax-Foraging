import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet


def decode_variables(hstates, scalars, scalar_vars, train_frac=0.8, plot=True, perm_test=False, first_ep_only=False, discard_values={},
                     keep_only={}, use_inf_as_na = True, model_pt = None):
    '''

    :param hstates: hstates df
    :param scalars: scalars df
    :param scalar_vars: dict of scalar variables to consider for generating what constitutes a positive label.
    Can include either inclusive bounds (lower_bound,upper_bound), or a single allowed value v, which is interpreted as (v,v).
    Variables conditions are combined in an AND fashion.
    Example of valid format: {'melee_on_screen': 1.0, 'dist_to_melee_l1':(0,6)}
    :param train_frac: Fraction of data to train on
    :param plot: Whether or not to plot the output. This is very slow to do  in a loop, so turn off if unnecessary
    :param perm_test: If true, permutes the order of the labels. Used as a null test against spurious decoding
    :param first_ep_only: If true, only use the first episode in the log file
    :param discard_values: Dict of scalar variables to define conditions on which to ignore timesteps for training/testing. For example, to discard sleep timesteps, use {'is_sleeping':1.0}
    :param keep_only: Dict of scalar variables to define conditions that must be satisfied to keep corresponding timesteps for training/testing.
    :param use_inf_as_na: Whether or not to set inf as na. Note that na values are discarded.
    :param model_pt: Pretrained logistic model to use. By default, none
    :return:
    '''
    pd.set_option('use_inf_as_na', use_inf_as_na)
    hstates_curr = hstates[scalars.index]

    idx_to_remove = pd.Series(False, index=scalars.index)

    #turn all conditions into upper and lower bounds
    for d in [scalar_vars, keep_only, discard_values]:
        for var in d:
            if not isinstance(d[var], tuple):
                d[var] = (d[var], d[var])
    #remove indices that correspond to na
    for var in scalar_vars:
        idx_to_remove = idx_to_remove | scalars[var].isna()

    for var in discard_values:
        idx_to_remove = idx_to_remove | (
                    (scalars[var] >= discard_values[var][0]) & (scalars[var] <= discard_values[var][1]))

    for var in keep_only:
        idx_to_remove = idx_to_remove | (
                    (scalars[var] < keep_only[var][0]) | (scalars[var] > keep_only[var][1]))

    if first_ep_only:
        first_id = scalars.iloc[0]['episode_id']
        idx_to_remove = idx_to_remove | (scalars['episode_id'] != first_id)

    idx_to_remove = idx_to_remove.to_numpy()
    hstates_curr = hstates_curr[~idx_to_remove]
    scalar_labels = np.ones(len(scalars) - np.count_nonzero(idx_to_remove)).astype(int)
    for var in scalar_vars:
        scalar_label_curr = scalars[var].to_numpy()[~idx_to_remove]
        lower, upper = scalar_vars[var]
        scalar_label_curr = ((scalar_label_curr <= upper) & (scalar_label_curr >= lower))
        scalar_labels = scalar_labels & scalar_label_curr
    print(len(scalar_labels), np.count_nonzero(scalar_labels))
    if perm_test:
        np.random.shuffle(scalar_labels)
    scalar_labels = scalar_labels.astype(float)
    n_tot = scalar_labels.shape[0]
    n_train = int(train_frac * n_tot)
    if model_pt is None:
        try:
            model = LogisticRegression(C=0.01, max_iter=100000, tol=1e-6).fit(hstates_curr[:n_train], scalar_labels[:n_train])
        except:
            return np.nan
    else:
        model = model_pt
    if n_train == n_tot:
        return np.nan, model
    pred = model.predict_proba(hstates_curr[n_train:])[:, 1]
    # plt.figure()
    # plt.scatter(pred, scalar_vals[n_train:], alpha = 0.2)
    results_df = pd.DataFrame({'decoded_prob': pred, 'ground_truth': scalar_labels[n_train:]})
    try:
        auc = roc_auc_score(scalar_labels[n_train:], pred)
    except:
        auc = np.nan
    if plot:
        plt.figure(dpi=100, figsize=(6, 4.5))
        sns.set(style="whitegrid")
        ax = sns.boxplot(x="ground_truth", y="decoded_prob", hue='ground_truth', data=results_df, showfliers=False)
        ax = sns.swarmplot(x="ground_truth", y="decoded_prob", data=results_df, color=".25", size=1.5, alpha=0.55)
        #ax = sns.violinplot(x="ground_truth", y="decoded_prob", data=results_df, inner='point', cut=0, density_norm="count", alpha =0.5)
        ax.set_xlabel('Ground Truth', fontsize=16)
        ax.set_ylabel('Decoded Prob.', fontsize=16)
        plt.ylim([-0.05, 1.05])
        title = ''
        for var in scalar_vars:
            lower, upper = scalar_vars[var]
            if lower == upper:
                title += '({}={})'.format(var, lower)
            else:
                title += r'$({} \leq$ {} $\leq {})$'.format(lower, var, upper)
        title += ': Test AUROC {}'.format(round(auc, 3))
        plt.title(title, fontsize=16)
        plt.tight_layout()
    print('AUROC:', auc)
    return auc, model


##Attempt at decoding absolute position via more finegrained regression (as opposed to more course-grained binary choice). did not work well so far...
def OLS_projection(neurons, behav_ts, train_frac=0.8, use_diff = False):
    '''

    :param neurons: (time, # neurons)
    :param behav_ts: (time, behav_dim)
    :param train_frac: int
    :param use_diff: bool
    :return:
    '''
    # mapping onto x,y speed (instead of position) gives better fits
    if use_diff:
        labels = np.diff(behav_ts, axis=0)
        inputs = neurons[:-1]
    else:
        labels = behav_ts
        inputs = neurons

    n_tot = inputs.shape[0]
    n_train = int(train_frac * n_tot)

    # reshape
    X = inputs
    #X = np.concatenate([X, np.ones(len(X))[:,np.newaxis]], axis=1)
    Y = labels
    X_train, Y_train = X[:n_train], Y[:n_train]
    # solve

    model = ElasticNet(max_iter=60000, tol=1e-6, alpha = 0.0001)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X)

    # weights = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ Y_train)
    # Y_pred = X @ weights

    # integrate along time axis
    if use_diff:
        Y_pred = np.cumsum(Y_pred, axis=0) + behav_ts[0]

    Y_pred_train, Y_pred_test = Y_pred[:n_train], Y_pred[n_train:]
    train_mse = np.sum((Y_pred_train - labels[:n_train])**2)
    test_mse = np.sum((Y_pred_test - labels[n_train:]) ** 2)
    print(train_mse, test_mse)

    return Y_pred