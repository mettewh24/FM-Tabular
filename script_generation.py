#!/usr/bin/env python
# coding: utf-8

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ot as pot

import time
import os
import copy

from utils import (
    compute_syndat_corr,
    compute_syndat_discr,
    compute_syndat_distr,
    minmax_scale_dummy,
    try_until_all_classes_found,
)
from data_loaders import dataset_loader
from sklearn.model_selection import train_test_split
import argparse
import gc
from ForestDiffusion import ForestDiffusionModel
from metrics import test_on_multiple_models, compute_coverage, test_imputation_regression, test_on_multiple_models_classifier
from STaSy.stasy import STaSy_model
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from CTABGANPlus.ctabgan import CTABGAN
from TabDDPM.scripts.pipeline import main_fn as tab_ddpm_fn
from TabDDPM.lib.dataset_prep import my_data_prep
from joblib.externals.loky import get_reusable_executor
from ot_flow import FMTabularGenerator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default=str(Path(__file__).resolve().parent / 'Results/generated/benchmark/metrics.csv'),
                    help='CSV file for the generation results')

parser.add_argument("--restore_from_name",type=str2bool, default=False, help="if True, restore session based on name")
parser.add_argument("--name",type=str, default='my_exp', help="used when restoring from crashed instances")

parser.add_argument("--methods",type=str, nargs='+', default=['TabDDPM', 'CTGAN', 'GaussianCopula', 'TVAE', 'CTABGAN', 'stasy', 'fm_tabular', 'forest_diffusion'], help="oracle, CTGAN, GaussianCopula, TVAE, CopulaGAN, CTABGAN, stasy, TabDDPM, forest_diffusion, fm_tabular")
parser.add_argument('--nexp', type=int, default=3,
                    help='number of experiences per parameter setting')
parser.add_argument('--ngen', type=int, default=5,
                    help='number of generations per method')
parser.add_argument('--n_tries', type=int, default=5,
                    help='number of models trained with different seeds in the metrics')
parser.add_argument('--datasets', nargs='+', type=str, default=['iris', 'wine', 'parkinsons','climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics','airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation',
                                                                'seeds', 'glass', 'ecoli', 'yeast', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump' ,
                                                                'tictactoe','congress', 'car', 'wine_quality_red', 'wine_quality_white', 'bean', 'california', 'libras'],
                    help='datasets on which to run the experiments')


# Forest hyperparameters
parser.add_argument('--forest_model', type=str, default='xgboost', help='xgboost, random_forest, lgbm, catboost')
parser.add_argument('--diffusion_type', type=str, default='flow', help='flow (flow-matching), vp (Variance-Preserving diffusion)')
parser.add_argument('--n_t', type=int, default=50, help='number of times t in [0,1]')
parser.add_argument('--n_t_sampling', type=int, default=0, help='number of times t in [0,1] for sampling  (0 will uses n_t steps; ignore this parameter honestly, its worth changing)')
parser.add_argument('--max_depth', type=int, default=7, help='max tree depth (xgboost, random_forest)')
parser.add_argument('--num_leaves', type=int, default=31, help='max number of leaves (lgbm)')
parser.add_argument('--n_estimators', type=int, default=100, help='number of trees (xgboost, random_forest, lgbm)')
parser.add_argument('--eta', type=float, default=0.3, help='lr (xgboost, random_forest, lgbm)')
parser.add_argument('--duplicate_K', type=int, default=100, help='number of times to duplicate the data for improved performanced of forests')
parser.add_argument('--gpu_hist', type=str2bool, default=False, help='If True, xgboost use the GPU')
parser.add_argument('--ycond', type=str2bool, default=True, help='If True, make a different forest model per label (when its not regression obviously)')
parser.add_argument('--eps', type=float, default=1e-3, help='')
parser.add_argument('--beta_min', type=float, default=0.1, help='')
parser.add_argument('--beta_max', type=float, default=8, help='')
parser.add_argument('--n_jobs', type=int, default=20, help='')
parser.add_argument('--n_batch', type=int, default=1, help='If >0 use the data iterator with the specified number of batches')

# stasy hyperparameters
parser.add_argument('--act', type=str, default='elu', help='')
parser.add_argument('--layer_type', type=str, default='concatsquash', help='')
parser.add_argument('--sde', type=str, default='vesde', help='')
parser.add_argument('--lr', type=float, default=2e-3, help='')
parser.add_argument('--num_scales', type=int, default=50, help='')

# fm_tabular hyperparameters
parser.add_argument('--fm_hidden_dim', type=int, default=512, help='Width of hidden layers in the fm_tabular velocity field')
parser.add_argument('--fm_time_dim', type=int, default=256, help='Dimension of the sinusoidal time embedding in fm_tabular')
parser.add_argument('--fm_depth', type=int, default=4, help='Number of residual blocks in the fm_tabular velocity field')
parser.add_argument('--fm_dropout', type=float, default=0.0, help='Dropout rate inside the fm_tabular velocity field')
parser.add_argument('--fm_num_epochs', type=int, default=2000, help='Training epochs for fm_tabular')
parser.add_argument('--fm_batch_size', type=int, default=512, help='Batch size for fm_tabular OT dataloader')
parser.add_argument('--fm_lr', type=float, default=3e-3, help='Learning rate for fm_tabular AdamW optimiser')
parser.add_argument('--fm_epsilon', type=float, default=0.5, help='Sinkhorn regularisation for fm_tabular OT coupling')
parser.add_argument('--fm_ot_solver', type=str, default='sinkhorn', help='OT solver for fm_tabular: sinkhorn or exact')
parser.add_argument('--fm_n_steps', type=int, default=100, help='Number of ODE steps for fm_tabular generation')
parser.add_argument('--fm_solver', type=str, default='euler', help='ODE solver for fm_tabular generation: euler, rk4, dopri5, ...')

args = parser.parse_args()


# Build the header for the results file, matching the CSV columns order
def _build_results_header() -> str:
    base_cols = [
        'dataset',
        'method',
        'W_train','W_test',
        'R2_real','R2_fake','R2_comb',
        'F1_real','F1_fake','F1_comb',
        'coverage_train','coverage_test',
        'percent_bias','coverage_rate','AW',
        'class_score','time'
    ]
    base_cols += [
        'W_train_std','W_test_std',
        'R2_real_std','R2_fake_std','R2_comb_std',
        'F1_real_std','F1_fake_std','F1_comb_std',
        'coverage_train_std','coverage_test_std',
        'percent_bias_std','coverage_rate_std','AW_std',
        'class_score_std','time_std'
    ]
    for key in ['lin','linboost','tree','treeboost']:
        base_cols += [
            f'R2_{key}_real',f'R2_{key}_fake',f'R2_{key}_comb',
            f'F1_{key}_real',f'F1_{key}_fake',f'F1_{key}_comb'
        ]
        base_cols += [
            f'R2_{key}_real_std',f'R2_{key}_fake_std',f'R2_{key}_comb_std',
            f'F1_{key}_real_std',f'F1_{key}_fake_std',f'F1_{key}_comb_std'
        ]

    base_cols += [
        'dist_score_train','dist_score_test',
        'corr_score_train','corr_score_test',
        'disc_score_train','disc_score_test'
    ]
    base_cols += [
        'dist_score_train_std','dist_score_test_std',
        'corr_score_train_std','corr_score_test_std',
        'disc_score_train_std','disc_score_test_std'
    ]
    return ','.join(base_cols) + '\n'


# Ensure the header is present as the first non-empty line of the output file
def _ensure_results_header(out_path: str, header: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        if not os.path.exists(out_path):
            with open(out_path, 'w') as f:
                f.write(header)
            return
        with open(out_path, 'r') as f:
            content = f.read()
        if len(content.strip()) == 0:
            with open(out_path, 'w') as f:
                f.write(header)
            return
        lines = content.splitlines()
        # Find first non-empty line
        first_idx = None
        for i, line in enumerate(lines):
            if line.strip() != '':
                first_idx = i
                break
        if first_idx is None:
            with open(out_path, 'w') as f:
                f.write(header)
            return
        first_line = lines[first_idx].strip().lower()
        if first_line.startswith('dataset'):
            return  # header already present
        # Prepend header before the first non-empty line
        new_content = '\n'.join(lines[:first_idx] + [header.rstrip('\n')] + lines[first_idx:])
        with open(out_path, 'w') as f:
            # Ensure trailing newline consistency
            f.write(new_content if new_content.endswith('\n') else new_content + '\n')
    except Exception:
        # Fail-safe: do not block the run if header handling goes wrong
        pass


def _safe_compute_syndat(metric_fn, metric_name, Xy_fake_scaled, Xy_ref_scaled, *, dataset, method, exp_id, gen_id, seed):
    try:
        return metric_fn(Xy_fake_scaled, Xy_ref_scaled, seed=seed)
    except Exception as e:
        print(f"Syndat {metric_name} failed for dataset={dataset}, method={method}, exp={exp_id}, gen={gen_id}: {e}")
        return np.nan

if __name__ == "__main__":

    if args.n_t_sampling == 0:
        args.n_t_sampling = args.n_t

    OTLIM = 5000

    # Ensure the results file starts with the header (only once)
    _ensure_results_header(args.out_path, _build_results_header())

    dataset_index_start = 0
    method_index_start = 0
    if args.restore_from_name:
        if os.path.isfile(args.name):
            with open(args.name, 'r') as f: # where we track where we are, to restore sessions after crashes
                dataset_index_start, method_index_start = f.read().split('&')
                dataset_index_start = int(dataset_index_start)
                method_index_start = int(method_index_start)

    for dataset_index in range(dataset_index_start, len(args.datasets)):

        dataset = args.datasets[dataset_index]
        print(dataset)
        X, bin_x, cat_x, int_x, y, bin_y, cat_y, int_y = dataset_loader(dataset)

        # Binary
        bin_indexes = []
        if bin_x is not None:
            bin_indexes = bin_indexes + bin_x
        bin_indexes_no_y = copy.deepcopy(bin_indexes)
        if bin_y:
            bin_indexes.append(X.shape[1])

        # Categorical (>=2 classes)
        cat_indexes = []
        if cat_x is not None:
            cat_indexes = cat_indexes + cat_x
        cat_indexes_no_y = copy.deepcopy(cat_indexes)
        if cat_y:
            cat_indexes.append(X.shape[1])

        # Integers
        int_indexes = []
        if int_x is not None:
            int_indexes = int_indexes + int_x
        int_indexes_no_y = copy.deepcopy(int_indexes)
        if int_y:
            int_indexes.append(X.shape[1])

        score_W1_train = {}
        score_W1_test = {}
        coverage = {}
        coverage_test = {}
        time_taken = {}
        percent_bias = {}
        coverage_rate = {}
        AW = {}
        f1_class = {}
        dist_score_train = {}
        dist_score_test = {}
        corr_score_train = {}
        corr_score_test = {}
        disc_score_train = {}
        disc_score_test = {}

        completed_runs = {}
        for method in args.methods:
            score_W1_test[method] = []
            score_W1_train[method] = []
            coverage[method] = []
            coverage_test[method] = []
            time_taken[method] = []
            percent_bias[method] = []
            coverage_rate[method] = []
            AW[method] = []
            f1_class[method] = []
            dist_score_train[method] = []
            dist_score_test[method] = []
            corr_score_train[method] = []
            corr_score_test[method] = []
            disc_score_train[method] = []
            disc_score_test[method] = []
            completed_runs[method] = 0

        R2 = {}
        f1 = {}
        for method in args.methods:
            R2[method] = {'real': {}, 'fake': {}, 'both': {}}
            f1[method] = {'real': {}, 'fake': {}, 'both': {}}
            for test_type in ['real','fake','both']:
                for test_type2 in ['mean','lin','linboost', 'tree', 'treeboost']:
                    R2[method][test_type][test_type2] = []
                    f1[method][test_type][test_type2] = []

        for method_index in range(method_index_start, len(args.methods)):
            
            method = args.methods[method_index]
            print(f'method={method}')

            with open(args.name, 'w') as f: # where we track where we are, to restore sessions after crashes
                f.write(f'{dataset_index}&{method_index}')

            for n in range(args.nexp):

                print(n)

                # Need to train/test split for evaluating the linear regression performance and for W1 based on test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n, stratify=y if bin_y or cat_y else None)
                Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
                Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
                Xy_test_df=pd.DataFrame(Xy_test, columns = [str(i) for i in range(Xy_test.shape[1])])
                Xy_train_df=pd.DataFrame(Xy_train, columns = [str(i) for i in range(Xy_train.shape[1])])
                os.makedirs('./train_test_data', exist_ok=True)
                Xy_test_df.to_csv(f'./train_test_data/test_data_{dataset}_exp{n}.csv', index=False)
                Xy_train_df.to_csv(f'./train_test_data/train_data_{dataset}_exp{n}.csv', index=False)

                Xy_train_used = Xy_train

                if method in ['TabDDPM','TVAE'] or not torch.cuda.is_available():
                    torch.set_default_tensor_type('torch.FloatTensor')
                else:
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')

                start = time.time()

                if method == 'oracle':
                    Xy_fake = np.tile(np.expand_dims(Xy_train, axis=0), reps=(args.ngen, 1, 1)) # [ngen, n, p]

                elif method == 'CTGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CTGANSynthesizer(metadata, epochs=2000,verbose=True)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'GaussianCopula':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = GaussianCopulaSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'TVAE':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = TVAESynthesizer(metadata, epochs=2000,verbose=True)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CopulaGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CopulaGANSynthesizer(metadata, epochs=2000,verbose=True)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CTABGAN': # CTABGAN+

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    synthesizer =  CTABGAN(pd_data = data_pd,
                                     categorical_columns = [str(i) for i in cat_indexes + bin_indexes],  
                                     general_columns= [str(i) for i in range(Xy_train_used.shape[1]) if i not in cat_indexes + bin_indexes + int_indexes],
                                     integer_columns = [str(i) for i in int_indexes]) 
                    synthesizer.fit()

                    def my_synthesizer():
                        synthetic_data = synthesizer.generate_samples()
                        return synthetic_data.to_numpy().astype('float')
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'stasy':
                    Xy_fake = STaSy_model(Xy_train_used,
                       categorical_columns=cat_indexes + bin_indexes, 
                       ordinal_columns=int_indexes, 
                       seed=n, 
                       epochs=10000,
                       ngen=args.ngen,
                       activation = args.act, layer_type = args.layer_type, sde = args.sde, lr = args.lr, num_scales = args.num_scales)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'TabDDPM': # TabDDPM

                    # Prep the data, will be save in the format that TabDDPM wants
                    columns = [str(i) for i in range(Xy_train_used.shape[1])]
                    data_pd = pd.DataFrame(Xy_train_used, columns = columns)
                    X_pd = data_pd[columns[0:-1]]
                    y_pd = data_pd[columns[-1]]
                    cat_ind = [str(i) for i in range(X_pd.shape[1]) if i in cat_indexes+bin_indexes]
                    noncat_ind = [str(i) for i in range(X_pd.shape[1]) if i not in cat_indexes+bin_indexes]
                    if cat_y:
                        task_type='multiclass'
                    elif bin_y:
                        task_type='binclass'
                    else:
                        task_type='regression'
                    my_data_prep(X_pd, y_pd, task=task_type, cat_ind=cat_ind, noncat_ind=noncat_ind)
                    synthetic_data = tab_ddpm_fn(config='TabDDPM/config/config.toml', 
                        cat_indexes=cat_indexes+bin_indexes, num_classes=len(np.unique(y_pd)) if cat_y or bin_y else 0, 
                        num_samples=Xy_train_used.shape[0], num_numerical_features=len(noncat_ind), seed=n, ngen=args.ngen)
                    Xy_fake = synthetic_data.astype('float')
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'forest_diffusion':

                    if args.ycond and (bin_y or cat_y):
                        forest_model = ForestDiffusionModel(X=Xy_train_used[:,:-1], 
                            label_y=Xy_train_used[:,-1],
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes_no_y,
                            bin_indexes=bin_indexes_no_y,
                            int_indexes=int_indexes_no_y,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n, nthread=1) 
                    else:
                        forest_model = ForestDiffusionModel(X=Xy_train_used, 
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes,
                            bin_indexes=bin_indexes,
                            int_indexes=int_indexes,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n, nthread=1) 
                    Xy_fake = forest_model.generate(batch_size=args.ngen*Xy_train_used.shape[0], n_t=args.n_t_sampling)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'fm_tabular':
                    try:
                        col_names = [str(i) for i in range(Xy_train_used.shape[1])]
                        data_pd = pd.DataFrame(Xy_train_used, columns=col_names)

                        cat_col_names  = [str(i) for i in cat_indexes + bin_indexes] or None
                        cont_col_names = [str(i) for i in range(Xy_train_used.shape[1])
                                        if i not in cat_indexes + bin_indexes] or None

                        y_col_name = str(Xy_train_used.shape[1] - 1)
                        use_y_cond = args.ycond and (bin_y or cat_y)

                        fm_device = "cuda" if torch.cuda.is_available() else "cpu"

                        fm_model = FMTabularGenerator(
                            device=fm_device,
                            hidden_dim=args.fm_hidden_dim,
                            time_dim=args.fm_time_dim,
                            depth=args.fm_depth,
                            dropout=args.fm_dropout,
                        )
                        fm_model.fit(
                            data_pd,
                            y_col=y_col_name if use_y_cond else None,
                            num_epochs=args.fm_num_epochs,
                            batch_size=args.fm_batch_size,
                            lr=args.fm_lr,
                            epsilon=args.fm_epsilon,
                            ot_solver=args.fm_ot_solver,
                            categorical_cols=cat_col_names,
                            continuous_cols=cont_col_names,
                            integer_idx=int_indexes,
                            seed=n,
                            verbose=True,
                        )

                        Xy_fake_list = []
                        for i_gen in range(args.ngen):
                            fake_df = fm_model.generate(
                                Xy_train_used.shape[0],
                                seed=n * args.ngen + i_gen,
                                n_steps=args.fm_n_steps,
                                method=args.fm_solver,
                            )
                            Xy_fake_list.append(fake_df[col_names].to_numpy(dtype=np.float64))
                        Xy_fake = np.stack(Xy_fake_list, axis=0)

                    except Exception as e:
                        print(f"Error in fm_tabular for dataset {dataset}: {e}")
                        continue
                end = time.time()
                get_reusable_executor().shutdown(wait=True)
                gc.collect()
                time_taken[method] += [(end - start)]
                print(f"time taken for traning + generation: {end - start} seconds")
                assert Xy_fake.shape[0] == args.ngen and Xy_fake.shape[1] == Xy_train_used.shape[0] and Xy_fake.shape[2] == Xy_train_used.shape[1]
                completed_runs[method] += 1

                for gen_i in range(args.ngen):
                    
                    if method == 'forest_diffusion':
                        os.makedirs(f'synthetic_data_files/{method}_{args.diffusion_type}', exist_ok=True)
                    else:
                        os.makedirs(f'synthetic_data_files/{method}', exist_ok=True)

                    Xy_fake_i = Xy_fake[gen_i]
                    data_pd = pd.DataFrame(Xy_fake_i, columns = [str(p) for p in range(Xy_fake_i.shape[1])])
                    if method == 'forest_diffusion':
                        data_pd.to_csv(f'synthetic_data_files/{method}_{args.diffusion_type}/synthetic_data_{method}_{args.diffusion_type}_{dataset}_exp{n}_gen{gen_i}.csv', index=False)
                    else:
                        data_pd.to_csv(f'synthetic_data_files/{method}/synthetic_data_{method}_{dataset}_exp{n}_gen{gen_i}.csv', index=False)

                    # Mixed data is tricky, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
                    # To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
                    # Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
                    # Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
                    # After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
                    Xy_train_scaled, Xy_fake_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_fake_i, cat_indexes, divide_by=2)
                    _, Xy_test_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_test, cat_indexes, divide_by=2)

                    # Wasserstein-1 based on L1 cost (after scaling)
                    if Xy_train.shape[0] < OTLIM:
                        score_W1_train[method] += [pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock'))]
                        score_W1_test[method] += [pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock'))] 

                    X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                    # Trained on real data
                    f1_r2_start = time.time()
                    f1_real, R2_real = test_on_multiple_models(X_train, y_train, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on fake data
                    f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on real data and fake data
                    X_both = np.concatenate((X_train,X_fake), axis=0)
                    y_both = np.concatenate((y_train,y_fake))
                    f1_both, R2_both = test_on_multiple_models(X_both, y_both, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)
                    f1_r2_end = time.time()
                    print(f"F1/R2 metric evaluation time for dataset={dataset}, method={method}, exp={n}, gen={gen_i}: {f1_r2_end - f1_r2_start} seconds")
                    
                    for key in['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                        f1[method]['real'][key] += [f1_real[key]]
                        f1[method]['fake'][key] += [f1_fake[key]]
                        f1[method]['both'][key] += [f1_both[key]]
                        R2[method]['real'][key] += [R2_real[key]]
                        R2[method]['fake'][key] += [R2_fake[key]]
                        R2[method]['both'][key] += [R2_both[key]]

                    # Get another different fake data for use as test fake-data
                    class_score_start = time.time()
                    Xy_fake_j = Xy_fake[(gen_i + 1) % args.ngen] # 0 -> 1, 1-> 2, n -> 0
                    # Classifier comparing real to fake data, the less it classify fake data as fake = the better
                    f1_class[method] += [test_on_multiple_models_classifier(X_train_real=Xy_train, X_train_fake=Xy_fake_i, X_test_fake=Xy_fake_j, cat_indexes=cat_indexes, nexp=args.n_tries)]
                    class_score_end = time.time()
                    print(f"Class score metric evaluation time for dataset={dataset}, method={method}, exp={n}, gen={gen_i}: {class_score_end - class_score_start} seconds")

                    # coverage based on L1 cost (after scaling)
                    coverage_start = time.time()
                    coverage[method] += [compute_coverage(Xy_train_scaled, Xy_fake_scaled, None)]
                    coverage_test[method] += [compute_coverage(Xy_test_scaled, Xy_fake_scaled, None)]
                    coverage_end = time.time()
                    print(f"Coverage metric evaluation time for dataset={dataset}, method={method}, exp={n}, gen={gen_i}: {coverage_end - coverage_start} seconds")
                    
                    #Syndat scores computation for comparison with previous work
                    syndat_start = time.time()
                    syndat_jobs = [
                        ('dist_score_train', compute_syndat_distr, Xy_train_scaled, dist_score_train[method]),
                        ('dist_score_test', compute_syndat_distr, Xy_test_scaled, dist_score_test[method]),
                        ('corr_score_train', compute_syndat_corr, Xy_train_scaled, corr_score_train[method]),
                        ('corr_score_test', compute_syndat_corr, Xy_test_scaled, corr_score_test[method]),
                        ('disc_score_train', compute_syndat_discr, Xy_train_scaled, disc_score_train[method]),
                        ('disc_score_test', compute_syndat_discr, Xy_test_scaled, disc_score_test[method]),
                    ]
                    for metric_name, metric_fn, Xy_ref_scaled, metric_store in syndat_jobs:
                        metric_value = _safe_compute_syndat(
                            metric_fn,
                            metric_name,
                            Xy_fake_scaled,
                            Xy_ref_scaled,
                            dataset=dataset,
                            method=method,
                            exp_id=n,
                            gen_id=gen_i,
                            seed=n,
                        )
                        metric_store += [metric_value]
                    syndat_end = time.time()
                    print(f"Syndat metric evaluation time for dataset={dataset}, method={method}, exp={n}, gen={gen_i}: {syndat_end - syndat_start} seconds")
                # Statistical measures
                imputation_start = time.time()
                X_fake = []
                y_fake = []
                for gen_i in range(args.ngen):
                    X_fake.append(np.expand_dims(Xy_fake[gen_i][:,:-1], axis=0))
                    y_fake.append(np.expand_dims(Xy_fake[gen_i][:,-1], axis=0))
                X_fake = np.concatenate(X_fake, axis=0) # [nimp, n, p-1]
                y_fake = np.concatenate(y_fake, axis=0) # [nimp, n, 1]
                # Compute regression statistics only for non-classification targets.
                if not cat_y and not bin_y:
                    percent_bias_, coverage_rate_, AW_ = test_imputation_regression(X_train, y_train, X_fake, y_fake, 
                        cat_indexes=cat_indexes_no_y, type_model='regression')
                else: 
                    percent_bias_, coverage_rate_, AW_ = 0.0, 0.0, 0.0
                percent_bias[method] += [percent_bias_]
                coverage_rate[method] += [coverage_rate_]
                AW[method] += [AW_]
                imputation_end = time.time()
                print(f"Imputation metric evaluation time for dataset={dataset}, method={method}, exp={n}: {imputation_end - imputation_start} seconds")

            # Keep individual class scores for both mean and standard deviation.
            # Write results in csv file
            # Columns: dataset,method,score_W1_train,score_W1_test,R2_real,R2_fake,f1_real,f1_fake, coverage
            if method == 'forest_diffusion':
                method_str = f"{method} n_t={args.n_t} n_t_sampling={args.n_t_sampling} model={args.forest_model} diffusion={args.diffusion_type} duplicate_K={args.duplicate_K} ycond={args.ycond} "
                if args.forest_model == 'xgboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} eta={args.eta} "
                elif args.forest_model == 'random_forest':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'catboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'lgbm':
                    method_str += f"num_leaves={args.num_leaves} n_trees={args.n_estimators} lr={args.eta} "
            elif method == 'fm_tabular':
                method_str = (
                    f"{method} "
                    f"hidden_dim={args.fm_hidden_dim} time_dim={args.fm_time_dim} "
                    f"depth={args.fm_depth} dropout={args.fm_dropout} "
                    f"epochs={args.fm_num_epochs} batch_size={args.fm_batch_size} "
                    f"lr={args.fm_lr} epsilon={args.fm_epsilon} "
                    f"ot_solver={args.fm_ot_solver} n_steps={args.fm_n_steps} "
                    f"solver={args.fm_solver} "
                    f"completed_runs={completed_runs[method]}/{args.nexp}"
                )
            else:
                method_str = f"{method}"
            print("Writing results without missing data")
            row = [dataset, method_str]
            for aggregate in (np.mean, np.std):
                row.extend([
                    aggregate(score_W1_train[method]), aggregate(score_W1_test[method]),
                    aggregate(R2[method]['real']['mean']), aggregate(R2[method]['fake']['mean']),
                    aggregate(R2[method]['both']['mean']),
                    aggregate(f1[method]['real']['mean']), aggregate(f1[method]['fake']['mean']),
                    aggregate(f1[method]['both']['mean']),
                    aggregate(coverage[method]), aggregate(coverage_test[method]),
                    aggregate(percent_bias[method]), aggregate(coverage_rate[method]),
                    aggregate(AW[method]), aggregate(f1_class[method]), aggregate(time_taken[method]),
                ])
            for key in ['lin', 'linboost', 'tree', 'treeboost']:
                for aggregate in (np.mean, np.std):
                    row.extend(aggregate(scores[method][kind][key])
                               for scores in (R2, f1) for kind in ('real', 'fake', 'both'))
            for aggregate in (np.mean, np.std):
                row.extend(aggregate(scores[method]) for scores in (
                    dist_score_train, dist_score_test, corr_score_train,
                    corr_score_test, disc_score_train, disc_score_test,
                ))
            print(row)
            with open(args.out_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
        method_index_start = 0 #  so we loop back again
