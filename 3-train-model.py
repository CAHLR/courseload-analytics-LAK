#!/usr/bin/env python
# coding: utf-8

# Only use one thread
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# Do not use GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np

import inspect
import random
import pickle
import math
import textwrap
import time
import warnings

from scipy.stats import pearsonr, mode
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score, mean_absolute_error, log_loss
from sklearn.model_selection import cross_val_score, RepeatedKFold, KFold
from sklearn.svm import SVR
from tqdm import tqdm

import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf

from utils import *

def main():

    USE_INDIV_SURVEY_VARS = True
    USE_IMPORTANCE_ITEMS = False

    IMPUTING_STRATEGY = 'control variables' 
    #IMPUTING_STRATEGY = 'knn' # k=2 

    df = pd.read_csv('../research-data/processed/lak22-courseload-final-studydata.csv')

    ADDITIONAL_INDIV_VARS = [
        'course_name_number', 'is_stem_course', 'is_stem_student', 'course_student_stem_match',
         'n_satisfied_prereqs_2021_Spring', 'n_satisfied_prereqs_all_past_semesters',
        'percent_satisfied_prereqs_2021_Spring', 'percent_satisfied_prereqs_all_past_semesters',
        'is_non_letter_grade_course', 'student_gpa', 'student_gpa_major', 
        'tl_importance', 'me_importance', 'ps_importance', 'combined_importance', 
        'tl_manage', 'me_manage', 'ps_manage', 'cl_combined_manage'
    ]
    if not USE_IMPORTANCE_ITEMS:
        for var in ['tl_importance', 'me_importance', 'ps_importance', 'combined_importance']:
            del df[var]

    if not USE_INDIV_SURVEY_VARS:
        for var in ADDITIONAL_INDIV_VARS:
            del df[var]
            
    # Remove string section information
    for col in ['section_num','secondary_section_number','all_section_numbers']:
        if col in df.columns:
            del df[col]
            
    # Remove Labels that are not needed
    for col in ['tl2', 'tl_sensitivity', 'me_sensitivity', 'ps_sensitivity', 'cl_sensitivity',
                'tl1_smoothed_lmm', 'me_smoothed_lmm', 'ps_smoothed_lmm', 'cl_smoothed_lmm', 
                'tl1_smoothed_student_average', 'me_smoothed_student_average', 'ps_smoothed_student_average',
                'cl_smoothed_student_average']:
        if col in df.columns:
            del df[col]

    # Drop string columns and get dummies for string var
    df = df.set_index('course_name_number')
    df = pd.get_dummies(df, columns=['class_type']) # upper, lower division, grad

    # Train (CV) and holdout
    train, test = train_test_split(df, test_size=0.15, random_state=12345, shuffle=True)

    prelim = dict()
    for l in tqdm(LABELS):
        prelim[l] = run_model_training(train, test, target=l, n_searches=25, 
                        ignore_warnings=True, imputing_strategy=IMPUTING_STRATEGY)

    with open(f'../workload-ml/models/model-results-25-{IMPUTING_STRATEGY}.p', 'wb') as f:
        pickle.dump(prelim, f)

if __name__ == '__main__':
    main()
