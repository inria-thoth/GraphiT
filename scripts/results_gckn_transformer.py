import pandas as pd

DATASET = 'MUTAG'
OUTPATH = '../logs_cv_gckn_trans/transformer/{}/gckn_{}_{}_{}_{}_True_True/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/fold-{}/logs{}.csv'
MODE = 'test'
LOGPATH = '../logs_cv_gckn_trans/transformer/{}/best_model.pkl'.format(DATASET)


def selection_model(fold_idx, pos_enc='diffusion', p=1, lappe=False, lap_dim=3):
    gckn_dim_grid = [32]
    gckn_path_grid = [5, 7]
    gckn_sigma_grid = [0.6]
    gckn_pooling_grid = ['sum']
    heads_grid = [1, 4, 8]
    layers_grid = range(1, 5)
    hidden_grid = [32, 64, 128]
    beta_grid = {'pstep': [0.5, 1.0], 'diffusion': [1.0,]}
    normalize_grid = ['sym']
    lr_grid = [0.001]
    wd_grid = [0.01, 0.001, 0.0001]
    dropouts = [0.0, 0.1]
    abs_PE = 'NoPE' if not lappe else 'Lap{}'.format(lap_dim)

    test_metric = ['test_acc', 'test_loss']

    param_names = ['gckn_path', 'gckn_dim', 'gckn_sigma', 'gckn_pooling',
        'wd', 'dropout', 'nb_heads', 'nb_layers', 'dim_hidden', 'lr', 'pos_enc', 'normalization', 'p', 'beta', 'fold']
    all_results = []
    num_results = 0

    for gckn_path in gckn_path_grid:
        for gckn_dim in gckn_dim_grid:
            for gckn_sigma in gckn_sigma_grid:
                for gckn_pooling in gckn_pooling_grid:
                    for head in heads_grid:
                        for layer in layers_grid:
                            for hidden in hidden_grid:
                                for lr in lr_grid:
                                    for wd in wd_grid:
                                        for normalize in normalize_grid:
                                                for beta in beta_grid[pos_enc]:
                                                    for dropout in dropouts:
                                                        
                                                        path = OUTPATH.format(
                                                            DATASET,
                                                            gckn_path, gckn_dim, gckn_sigma, gckn_pooling,
                                                            wd, dropout, lr, layer, head, hidden,
                                                            'LN', pos_enc, normalize, p, beta, fold_idx, '')
                                                        path_test = OUTPATH.format(
                                                            DATASET,
                                                            gckn_path, gckn_dim, gckn_sigma, gckn_pooling,
                                                            wd, dropout, lr, layer, head, hidden,
                                                            'LN', pos_enc, normalize, p, beta, fold_idx, '_test')
                                                        try:
                                                           metric_train = pd.read_csv(path, index_col=0)
                                                           if MODE == 'val':
                                                               metric_test = metric_train.copy()
                                                           else:
                                                               metric_test = pd.read_csv(path_test, index_col=0)
                                                           num_results += 1
                                                        except Exception:
                                                           continue
                                                        metric = pd.DataFrame(metric_train.iloc[-50:].mean()).T
                                                        metric_test_avg = metric_test.iloc[-50:].mean()
                                                        for tm in test_metric:
                                                            metric[tm] = metric_test_avg[tm]
                                                        metric['val_acc_std'] = metric_train.iloc[-50:]['val_acc'].std()
                                                        metric['test_acc_std'] = metric_test.iloc[-50:]['test_acc'].std()
                                                        params = [gckn_path, gckn_dim, gckn_sigma, gckn_pooling,
                                                            wd, dropout, head, layer, hidden, lr, pos_enc, normalize, p, beta, fold_idx]
                                                        for param, param_name in zip(params, param_names):
                                                            metric[param_name] = [param]
                                                        all_results.append(metric)
    all_results = pd.concat(all_results)
    best_model = all_results.loc[all_results['val_acc'] == all_results['val_acc'].max()]
    best_model = best_model.iloc[[best_model['val_acc_std'].idxmin()]]
    return best_model

def main():
    """
    This functions reads all the experiment results given a parameter grid
    and outputs the scores and best models per fold as well as a global
    score.
    """

    results = []
    pos_enc = 'pstep'
    p = 1
    lappe = False
    lap_dim = 2
    best_params = []
    for fold_idx in range(1, 11):
        model = selection_model(fold_idx, pos_enc, p, lappe, lap_dim)
        results.append(model)
        best_params.append(model.iloc[0].to_dict())

    table = pd.concat(results)
    main_info = ['val_acc', 'test_acc', 'gckn_path', 'wd', 'dropout', 'nb_heads', 'nb_layers', 'dim_hidden', 'lr', 'pos_enc', 'normalization', 'p', 'beta', 'fold']
    print(table[main_info])
    print("final acc: {}".format(table['test_acc'].mean()))
    print("final acc std: {}".format(table['test_acc'].std()))
    import pickle
    if MODE == 'val':
        with open(LOGPATH, 'wb') as handle:
            pickle.dump(best_params, handle)


if __name__ == "__main__":
    main()
