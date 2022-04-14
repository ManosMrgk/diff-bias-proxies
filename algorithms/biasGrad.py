"""
Bias gradient descent/ascent algorithm.
"""
import os.path

import numpy as np

import matplotlib.pyplot as plt

import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import balanced_accuracy_score

import utils.data_utils
from utils.evaluation import (get_objective, get_valid_objective, get_valid_objective_, get_best_thresh,
                              get_test_objective_)

import progressbar


def spd_diff(y_pred, y_true, p):
    # Differentiable surrogate for the SPD
    return torch.mean(y_pred[p == 0]) - torch.mean(y_pred[p == 1])


def eod_diff(y_pred, y_true, p):
    # Differentiable surrogate for the EOD
    if isinstance(p, torch.Tensor):
        return torch.mean(y_pred[torch.logical_and(p == 0, y_true == 1)]) - \
               torch.mean(y_pred[torch.logical_and(p == 1, y_true == 1)])
    else:
        return torch.mean(y_pred[np.logical_and(p == 0, y_true == 1)]) - \
               torch.mean(y_pred[np.logical_and(p == 1, y_true == 1)])


def choose_best_thresh_bal_acc(data: utils.data_utils.TabularData, valid_pred_scores: np.ndarray, n_thresh=101):
    # Find the best threshold
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(data.y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def choose_best_thresh_bal_acc_(y_valid: np.ndarray, valid_pred_scores: np.ndarray, n_thresh=101):
    # Find the best threshold
    threshs = np.linspace(0, 1, n_thresh)
    performances = []
    for thresh in threshs:
        perf = balanced_accuracy_score(y_valid, valid_pred_scores > thresh)
        performances.append(perf)
    best_thresh = threshs[np.argmax(performances)]

    return best_thresh


def plot_results(step_num: np.ndarray, objective: list, bias_metric: list, pred_performance: list,
                 j_best: int, seed: int, config: dict, suffix='', display=False):
    fig = plt.figure()
    plt.plot(step_num, objective, label='Constrained Objective')
    plt.plot(step_num, bias_metric, label='Bias: ' + str(config['metric']))
    plt.plot(step_num, pred_performance, label='Balanced Accuracy')
    plt.vlines(x=step_num[j_best], ymin=min(bias_metric), ymax=1.0, colors='red')
    plt.xlabel('Step №')
    plt.legend()
    plt.savefig(fname=os.path.join('results/figures/biasGrad_') + str(config['experiment_name'] + '_' + str(seed) + \
                                                             suffix + '.png'), dpi=300, bbox_inches="tight")
    if display:
        plt.show()


def save_finetuning_trajectory(results: dict, seed: int, config: dict):
    arr = np.stack((results['objective'], results['bias'], results['perf']), axis=1)
    np.savetxt(fname=os.path.join('results/logs/') + str(config['experiment_name'] + '_' + str(seed) +
                                                            '_trajectory' + '.csv'), X=arr)


def bias_gradient_decent(model: nn.Module, data, config: dict, seed: int, asc: bool = False, plot: bool = False,
                         display: bool = False, verbose: int = 1):
    # Suppress annoying warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if verbose:
        print('Performing bias gradient ascent/descent...')
        print()

    model_ = copy.deepcopy(model)

    objective_ = []
    bias_metric_ = []
    pred_performance_ = []

    best_model = None
    best_objective = -1
    j_best = -1
    best_bias = 1

    model_.train()

    if config['metric'] == 'spd':
        loss_fn = spd_diff
    elif config['metric'] == 'eod':
        loss_fn = eod_diff
    else:
        raise NotImplementedError('ERROR: bias metric not supported!')

    optimiser = optim.Adam(model_.parameters(), lr=config['biasGrad']['lr'])

    if verbose:
        bar = progressbar.ProgressBar(maxval=config['biasGrad']['n_epochs'])
        bar.start()
        bar_cnt = 0

    for i in range(config['biasGrad']['n_epochs']):
        if config['biasGrad']['val_only']:
            batch_idxs = torch.split(torch.randperm(data.X_valid.size(0)), config['biasGrad']['batch_size'])
        else:
            batch_idxs = torch.split(torch.randperm(data.X_train.size(0)), config['biasGrad']['batch_size'])
        train_loss = 0

        eval_factor = int(len(batch_idxs) / config['biasGrad']['n_evals'])
        batch_cnt = 0

        for batch in batch_idxs:
            if config['biasGrad']['val_only']:
                X = data.X_valid[batch, :]
                y = data.y_valid[batch]
                p = data.p_valid[batch]
            else:
                X = data.X_train[batch, :]
                y = data.y_train[batch]
                p = data.p_train[batch]

            optimiser.zero_grad()
            loss = loss_fn(y_pred=model_(X)[:, 0], y_true=y, p=p)
            if asc:
                loss = -loss
            loss.backward()
            train_loss += loss.item()
            optimiser.step()

            if batch_cnt % eval_factor == 0:
                with torch.no_grad():
                    valid_pred_scores = model_(data.X_valid)[:, 0].reshape(-1, 1).cpu().numpy()

                    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
                    best_thresh = choose_best_thresh_bal_acc(data=data, valid_pred_scores=valid_pred_scores)

                    obj_dict = get_objective((valid_pred_scores > best_thresh) * 1., data.y_valid.numpy(), data.p_valid,
                                             config['metric'], config['objective']['sharpness'],
                                             config['objective']['epsilon'])
                    objective_.append(obj_dict['objective'])
                    bias_metric_.append(obj_dict['bias'])
                    pred_performance_.append(obj_dict['performance'])

                    if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
                        best_bias = np.abs(obj_dict['bias'])
                        best_model = copy.deepcopy(model_)
                        j_best = len(objective_) - 1

            batch_cnt += 1

        if verbose:
            bar.update(bar_cnt)
            bar_cnt += 1

    if verbose:
        print('\n' * 2)

    save_finetuning_trajectory(
        results={'objective': pred_performance_ * (np.abs(bias_metric_) < config['objective']['epsilon']),
                 'bias': bias_metric_,
                 'perf': pred_performance_},
        seed=seed, config=config)

    # Plotting
    if plot:
        step_num = np.arange(1, len(objective_) + 1)
        plot_results(step_num=step_num, objective=objective_,
                     bias_metric=bias_metric_, pred_performance=pred_performance_, j_best=j_best,
                     seed=seed, config=config, display=display, suffix='')

    model_.eval()

    if best_model is None:
        print()
        print()
        print('No debiased model satisfies the constraints!')
        best_model = copy.deepcopy(model)

    best_model.eval()

    return best_model


def bias_gda_dataloaders(model: nn.Module, data_loader_train: DataLoader, data_loader_val: DataLoader, dataset_size_val,
                         opt_alg, device, config: dict, seed: int, plot: bool = False,
                         display: bool = False, verbose: int = 1):
    # Suppress annoying warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if verbose:
        print('Performing bias gradient ascent/descent...')
        print()

    model_ = copy.deepcopy(model)

    objective_ = []
    bias_metric_ = []
    pred_performance_ = []

    best_model = None
    j_best = -1
    best_bias = 1

    model_.train()

    if config['metric'] == 'spd':
        loss_fn = spd_diff
    elif config['metric'] == 'eod':
        loss_fn = eod_diff
    else:
        raise NotImplementedError('ERROR: bias metric not supported!')

    optimiser = opt_alg(params=model_.parameters(), lr=config['biasGrad']['lr'])

    # If training data are not provided, perform the procedure entirely on the validation data
    if data_loader_train is None:
        data_loader_train = data_loader_val

    if verbose:
        bar = progressbar.ProgressBar(maxval=config['biasGrad']['n_epochs'])
        bar.start()
        bar_cnt = 0

    # Evaluate the original model (in case its unbiased)
    with torch.no_grad():
        valid_pred_scores = np.zeros((dataset_size_val,))
        y_valid = np.zeros((dataset_size_val,))
        p_valid = np.zeros((dataset_size_val,))

        with torch.no_grad():
            cnt = 0
            for X_, y_, p_ in data_loader_val:
                X_ = X_.to(device)
                y_ = y_.to(device).to(torch.float)
                p_ = p_.to(device)

                outputs = model_(X_)

                valid_pred_scores[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                    'batch_size']] = outputs[:, 0].cpu().numpy()
                y_valid[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                    'batch_size']] = y_.cpu().numpy()
                p_valid[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                    'batch_size']] = p_.cpu().numpy()

                cnt += 1

        # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
        best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

        obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid, p_test=p_valid,
                                       config=config)
        objective_.append(obj_dict['objective'])
        bias_metric_.append(obj_dict['bias'])
        pred_performance_.append(obj_dict['performance'])

        if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
            best_bias = np.abs(obj_dict['bias'])
            best_model = copy.deepcopy(model_)
            j_best = len(objective_) - 1

        asc = obj_dict['bias'] < 0

    terminus_est = False
    for i in range(config['biasGrad']['n_epochs']):

        eval_factor = int(len(data_loader_train) / config['biasGrad']['n_evals'])
        batch_cnt = 0

        # Iterate over data
        for X, y, p in data_loader_train:

            X = X.to(device)
            y = y.to(device).to(torch.float)
            p = p.to(device)

            optimiser.zero_grad()

            loss = loss_fn(y_pred=model_(X)[:, 0], y_true=y, p=p)

            if asc:
                loss = -loss
            loss.backward()
            optimiser.step()

            # Evaluate on the validation data every `eval_factor' batches
            if batch_cnt % eval_factor == 0:
                with torch.no_grad():
                    valid_pred_scores = np.zeros((dataset_size_val,))
                    y_valid = np.zeros((dataset_size_val,))
                    p_valid = np.zeros((dataset_size_val,))

                    with torch.no_grad():
                        cnt = 0
                        for X_, y_, p_ in data_loader_val:
                            X_ = X_.to(device)
                            y_ = y_.to(device).to(torch.float)
                            p_ = p_.to(device)

                            outputs = model_(X_)

                            valid_pred_scores[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                                'batch_size']] = outputs[:, 0].cpu().numpy()
                            y_valid[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                                'batch_size']] = y_.cpu().numpy()
                            p_valid[cnt * config['biasGrad']['batch_size']:(cnt + 1) * config['biasGrad'][
                                'batch_size']] = p_.cpu().numpy()

                            cnt += 1

                    # Choose the best threshold w.r.t. the balanced accuracy on the held-out data
                    best_thresh = choose_best_thresh_bal_acc_(y_valid=y_valid, valid_pred_scores=valid_pred_scores)

                    obj_dict = get_test_objective_(y_pred=(valid_pred_scores > best_thresh) * 1., y_test=y_valid,
                                                   p_test=p_valid, config=config)
                    objective_.append(obj_dict['objective'])
                    bias_metric_.append(obj_dict['bias'])
                    pred_performance_.append(obj_dict['performance'])

                    if np.abs(obj_dict['bias']) < best_bias and obj_dict['performance'] >= config['biasGrad']['obj_lb']:
                        best_bias = np.abs(obj_dict['bias'])
                        best_model = copy.deepcopy(model_)
                        j_best = len(objective_) - 1

                    # Termination criterion
                    if obj_dict['performance'] <= 0.52:
                        terminus_est = True
                        break

            batch_cnt += 1

        if terminus_est:
            break

        if verbose:
            bar.update(bar_cnt)
            bar_cnt += 1

        if verbose:
            print('\n' * 2)

    save_finetuning_trajectory(
        results={'objective': pred_performance_ * (np.abs(bias_metric_) < config['objective']['epsilon']),
                 'bias': bias_metric_,
                 'perf': pred_performance_},
        seed=seed, config=config)

    # Plotting
    if plot:
        step_num = np.arange(1, len(objective_) + 1)
        plot_results(step_num=step_num, objective=objective_,
                     bias_metric=bias_metric_, pred_performance=pred_performance_, j_best=j_best,
                     seed=seed, config=config, display=display, suffix='')

    if best_model is None:
        print()
        print()
        print('No debiased model satisfies the constraints!')
        best_model = copy.deepcopy(model)

    model_.eval()
    best_model.eval()

    return best_model
