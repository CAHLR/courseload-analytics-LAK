"""
Collection of objects and functions used across multiple scripts.
"""
import ast
import inspect
import itertools
import json
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import sys
import tensorflow as tf
import textwrap
import time
import warnings
import xgboost as xgb

from datetime import timedelta
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import pearsonr, mode
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score, mean_absolute_error, log_loss
from sklearn.model_selection import cross_val_score, RepeatedKFold, KFold
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR
from tqdm import tqdm

# C2V
COURSE2VEC = np.load('../LAK_paper_data/course2vec.npy')
COURSE2VEC_idx = json.load(open('../LAK_paper_data/course2idx.json'))
AVG_C2V = COURSE2VEC.mean(axis=0)

# get course vector for a given course ID (cid)
def get_c2v(cid: str) -> list:
    try:
        idx = COURSE2VEC_idx[cid]-1 # minus 1 because the index starts from 1 rather than 0
    except KeyError:
        try:
            idx = COURSE2VEC_idx[cid[::-1].replace(' ', '_', 1)[::-1]]-1
        except KeyError:
            try:
                idx = COURSE2VEC_idx[ABBR_CID2CID[cid]]-1
            except:
                return np.nan
    vec = COURSE2VEC[idx]
    return vec

# Survey answers to numeric
def answers_to_numeric(row, col: str) -> float:
    """
    This function converts verbatim survey scale responses to numeric values (e.g., 'sometimes' to 3).
    Each course load type (time load, mental effort, psychological stress) has a control variable 
    asking the student about how manageable the load was. These control variables will be subtracted
    from the main survey ratings separately (e.g., high stress and high managability is low actual stress, high 
    stress and low managability is high actual stress) or included in the predictive model as control
    variable.
    """
    if row[col] in ['0-5 hours per week', 'Nearly never', 'Nearly always unmanageable', 
                    'A very low amount', 'Not important at all']:
        return 1.0
    elif row[col] in ['6-10 hours per week', 'Seldom', 'Mostly unmanageable', 
                      'A low amount', 'Slightly important']:
        return 2.0
    elif row[col] in ['11-15 hours per week', 'Sometimes', 'Sometimes manageable', 
                      'A moderate amount', 'Moderately important']:
        return 3.0
    elif row[col] in ['16-20 hours per week', 'Frequently', 'Mostly manageable', 
                      'A high amount', 'Important']:
        return 4.0
    elif row[col] in ['21-25 hours per week', 'Nearly always', 'Nearly always manageable', 
                      'A very high amount', 'Very important']:
        return 5.0
    elif row[col] == '26+ hours per week':
        return 6.0
    else:
        raise ColumnError('No relevant column values detected, please check which column you passed as argument.')  
        return
      
# STEM Mapping of Majors and Courses on department level

d_dept_stem = {
'-': np.nan,
'African American Studies': False,
'Ag & Env Chem Grad Grp': True,
'Ag & Resource Econ & Pol': True,
'Anc Hist Med Arc Grad Grp': False,
'Ancient Greek & Roman Studies': False,
'Anthropology': False,
'Applied Sci & Tech Grad Grp': True,
'Architecture': True,
'Art Practice': False,
'Asian Studies Grad Grp': False,
'Astronomy': True,
'Bioengineering': True,
'Bioengineering-UCSF Grad Grp': True,
'Biophysics Grad Grp': True,
'Biostatistics Grad Grp': True,
'Buddhist Studies Grad Grp': False,
'Business': False,
'Chem & Biomolecular Eng': True,
'Chemistry': True,
'City & Regional Planning': True,
'Civil & Environmental Eng': True,
'Classics': False,
'College Writing Programs': False,
'Comparative Biochem Grad Grp': True,
'Comparative Literature': False,
'Computational Biology Grad Grp': True,
'Critical Theory Grad Grp': False,
'Data Science': True,
'Demography': False,
'Design Innovation': False,
'Development Eng Grad Grp': True,
'Development Practice Grad Grp': False,
'Earth & Planetary Science': True,
'East Asian Lang & Culture': False,
'Economics': True,
'Education': False,
'Electrical Eng & Computer Sci': True,
'Endocrinology Grad Grp': True,
'Energy & Resources Grad Grp': True,
'Engineering Joint Programs': True,
'Engineering Science': True,
'English': False,
'Env Sci, Policy, & Mgmt': True,
'Environmental Health Sci GG': True,
'Epidemiology Grad Grp': True,
'Ethnic Studies': False,
'European Studies Grad Grp': False,
'FPF-African American Studies': False,
'FPF-Anc Greek & Roman Studies': False,
'FPF-Anthropology': False,
'FPF-Art Practice': False,
'FPF-Astronomy': True,
'FPF-Chemistry': True,
'FPF-Classics': False,
'FPF-College Writing Program': False,
'FPF-Comparative Literature': False,
'FPF-Earth & Planetary Science': True,
'FPF-English': False,
'FPF-Env Sci, Policy, & Mgmt': True,
'FPF-Ethnic Studies': False,
'FPF-Film & Media': False,
'FPF-Gender & Womens Studies': False,
'FPF-Geography': False,
'FPF-History': False,
'FPF-History of Art': False,
'FPF-IAS Teaching Program': False,
'FPF-Integrative Biology': True,
'FPF-Interdisc Social Sci Pgms': False,
'FPF-Legal Studies': False,
'FPF-Letters & Science': np.nan, # comprises of multiple departments
'FPF-Linguistics': False,
'FPF-Mathematics': True,
'FPF-Molecular & Cell Biology': True,
'FPF-Music': False,
'FPF-Philosophy': False,
'FPF-Political Science': False,
'FPF-Psychology': True,
'FPF-Rhetoric': False,
'FPF-Sociology': False,
'FPF-South & SE Asian Studies': False,
'FPF-Statistics': True,
'FPF-UG Interdisciplinary Stds': False,
'Film and Media': False,
'Folklore Grad Grp': False,
'French': False,
'Gender & Womens Studies': False,
'Geography': False,
'German': False,
'Global Metro Std Grad Grp': False,
'Global Studies Grad Grp': False,
'Grad Division Other Programs': np.nan,
'Health & Medical Sci Grad Grp': True,
'Health Policy GG': False,
'History': False,
'History of Art': False,
'Industrial Eng & Ops Research': True,
'Infectious Diseases & Immun GG': True,
'Information': True,
'Integrative Biology': True,
'Interdisc Social Science Pgms': False,
'Interdisciplinary Doctoral Pgm': False,
'Italian Studies': False,
'JSP Graduate Program': False,
'Jewish Studies Program': False,
'Journalism': False,
'L&S Chemistry': True,
'L&S Computer Science': True,
'L&S Data Science': True,
'L&S Envir Econ & Policy': True,
'L&S Legal Studies': False,
'L&S Ops Rsch & Mgmt Sci': True,
'L&S Public Health': True,
'L&S Social Welfare': False,
'L&S Undeclared': np.nan,
'Landscape Arch & Env Plan': True,
'Latin American Studies GG': False,
'Law': False,
'Linguistics': False,
'Logic and Method of Science GG': False,
'Materials Science & Eng': True,
'Mathematics': True,
'Mechanical Engineering': True,
'Medieval Studies Program': False,
'Metabolic Biology Grad Grp': True,
'Microbiology Grad Grp': True,
'Middle Eastern Lang & Cultures': False,
'Military Affairs Program': False,
'Molecular & Cell Biology': True,
'Molecular Toxicology Grad Grp': True,
'Music': False,
'Nano Sci & Eng Grad Grp': True,
'Near Eastern Religions GG': False,
'Near Eastern Studies': False,
'Neuroscience Graduate Program': True,
'New Media Grad Grp': False,
'Nuclear Engineering': True,
'Nutritional Sciences & Tox': True,
'Optometry': True,
'Other Arts & Humanities Pgms': False,
'Other Bio Sciences Pgms': True,
'Other Clg of Natural Res Pgms': True,
'Other EVCP Programs': False,
'Other Env Design Programs': True,
'Other Math & Physical Sci Pgms': True,
'Other Social Sciences Programs': False,
'Performance Studies Grad Grp': False,
'Philosophy': False,
'Physical Education': False,
'Physics': True,
'Plant & Microbial Biology': True,
'Political Science': False,
'Psychology': True,
'Public Health': True,
'Public Policy': False,
'Rangeland & Wildlife Mgmt GG': False,
'Rhetoric': False,
'Romance Lang & Lit Grad Pgm': False,
'Scandinavian': False,
'Sci & Tech Stds Grad Grp': True,
'Science & Math Educ Grad Grp': True,
'Slavic Languages & Literatures': False,
'Social Welfare': False,
'Sociology': False,
'Sociology and Demography GG': False,
'South & SE Asian Studies': False,
'Spanish & Portuguese': False,
'Statistics': True,
'Study of Religion Grad Grp': False,
'Theater Dance & Perf Stds': False,
'UC Education Abroad Program': False,
'UCBX-Concurrent Enrollment Dpt': False,
'UG Interdisciplinary Studies': False,
'Urban Design Grad Grp': False,
'Vision Science Grad Grp': True
}

# Hand-coded mapping of courses and majors from survey data
# Based on https://www.ice.gov/sites/default/files/documents/stem-list.pdf
d_stem_courses = {
"American Studies 101": False,
"American Studies C172": False,
"Anthropology 115": False,
"Anthropology 121AC": False,
"Anthropology 141": False,
"Anthropology 160AC": False,
"Anthropology 3AC": False,
"Architecture 11B": False,  # only lists Naval Architecture and Marine Engineering.
"Architecture 170B": False,  # only lists Naval Architecture and Marine Engineering.
"Architecture 198BC": False,  # only lists Naval Architecture and Marine Engineering.
"Asian Am & Asn Diaspora Stds 121": False,
"Asian Am & Asn Diaspora Stds 132AC": False,
"Asian Am & Asn Diaspora Stds 171": False,
"Asian Am & Asn Diaspora Stds 20A": False,
"Astronomy 84": True,
"Astronomy C12": True,
"Bioengineering 100": True,
"Bioengineering 104": True,
"Bioengineering 11": True,
"Bioengineering 110": True,
"Bioengineering 153": True,
"Bioengineering 25": True,
"Bioengineering 98": True,
"Biology 1A": True,
"Biology 1AL": True,
"Biology 1B": True,
"Business Admin-Undergrad 10": False,   # list includes business, but only business statistics and not administration
"Business Admin-Undergrad 102B": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 103": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 105": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 106": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 131": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 135": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 141": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 147": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 169": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 192T": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 194": False,   # list includes business, but only business statistics and not administration,
"Business Admin-Undergrad 198": False,   # list includes business, but only business statistics and not administration,
"Celtic Studies R1B": False,
"Chemical Engineering 141": True,
"Chemical Engineering 150A": True,
"Chemical Engineering 98": True,
"Chemistry 12B": True,
"Chemistry 1A": True,
"Chemistry 1AL": True,
"Chemistry 1B": True,
"Chemistry 3A": True,
"Chemistry 3AL": True,
"Chemistry 3B": True,
"Chemistry 3BL": True,
"Chemistry 98": True,
"Chinese 10Y": False,
"Chinese 1A": False,
"Civil & Environmental Eng 105": True,
"Civil & Environmental Eng 107": True,
"Civil & Environmental Eng 11": True,
"Civil & Environmental Eng 113": True,
"Civil & Environmental Eng 123": True,
"Civil & Environmental Eng 155": True,
"Civil & Environmental Eng 166": True,
"Civil & Environmental Eng 175": True,
"Civil & Environmental Eng 198": True,
"Civil & Environmental Eng 199": True,
"Civil & Environmental Eng C88": True,
"Classics 10B": False,
"Classics 130E": False,
"Classics 28": False,
"Cognitive Science 1": True,
"Cognitive Science 131": True,
"Cognitive Science 190": True,
"College Writing Programs R1A": False,
"College Writing Programs R4B": False,
"Computer Science 10": True,
"Computer Science 161": True,
"Computer Science 162": True,
"Computer Science 170": True,
"Computer Science 188": True,
"Computer Science 189": True,
"Computer Science 194": True,
"Computer Science 195": True,
"Computer Science 197": True,
"Computer Science 198": True,
"Computer Science 370": True,
"Computer Science 47B": True,
"Computer Science 61A": True,
"Computer Science 61B": True,
"Computer Science 61C": True,
"Computer Science 70": True,
"Computer Science 88": True,
"Computer Science W182": True,
"Computer Science W186": True,
"Data Science, Undergraduate 198": True,   # data science is usually listed as 11.0401 Information Science/Studies.
"Data Science, Undergraduate C100": True,
"Data Science, Undergraduate C104": True,
"Data Science, Undergraduate C8": True,
"Demography C175": False,
"Design Innovation 10": False,
"Design Innovation 15": False,
"Design Innovation 198": False,
"Design Innovation 98": False,
"Dutch 171AC": False,
"Earth & Planetary Science C12": True,
"Economics 1": False,     # list only lists econometrics/quantitative economics
"Economics 100A": False,
"Economics 100B": False,
"Economics 101A": False,
"Economics 115": False,
"Economics 157": False,
"Economics 172": False,
"Education 130": False,
"Education 197": False,
"Electrical Eng & Computer Sci 126": True,
"Electrical Eng & Computer Sci 127": True,
"Electrical Eng & Computer Sci 16A": True,
"Electrical Eng & Computer Sci 16B": True,
"Energy and Resources 98": True,
"Engineering 125": True,
"Engineering 26": True,
"Engineering 29": True,
"English 110": False,
"English 170": False,
"English 24": False,
"English 43B": False,
"English 45C": False,
"English R1B": False,
"Env Sci, Policy, & Mgmt 114": True,  # Environmental Science is Listed
"Env Sci, Policy, & Mgmt 131": True,
"Env Sci, Policy, & Mgmt 152": True,
"Env Sci, Policy, & Mgmt 40": True,
"Env Sci, Policy, & Mgmt 50AC": True,
"Env Sci, Policy, & Mgmt 98": True,
"Env Sci, Policy, & Mgmt 98BC": True,
"Env Sci, Policy, & Mgmt C167": True,
"Environ Econ & Policy C101": False,
"Environmental Design 100": False,
"Ethnic Studies 101A": False,
"Ethnic Studies 190": False,
"Ethnic Studies 197": False,
"Film 171": False,
"Film R1B": False,
"French 1": False,
"French 2": False,
"Gender & Womens Studies 100AC": False,
"Gender & Womens Studies 139": False,
"Geography 130": False, # only lists Geographic Information Science and Cartography.
"Geography 70AC": False,
"Global Poverty & Practice 105": False,
"Global Studies 110Q": False,
"Global Studies 173": False,
"Global Studies C10A": False,
"History 100M": False,
"History 109C": False,
"History 160": False,
"History 190": False,
"History 6B": False,
"History C139C": False,
"History R1B": False,
"History of Art 190F": False,
"Industrial Eng & Ops Rsch 135": True,
"Industrial Eng & Ops Rsch 162": True,
"Industrial Eng & Ops Rsch 165": True,
"Industrial Eng & Ops Rsch 166": True,
"Industrial Eng & Ops Rsch 170": True,
"Industrial Eng & Ops Rsch 173": True,
"Industrial Eng & Ops Rsch 185": True,
"Industrial Eng & Ops Rsch 186": True,
"Industrial Eng & Ops Rsch 190E": True,
"Industrial Eng & Ops Rsch 195": True,
"Industrial Eng & Ops Rsch 221": True,
"Industrial Eng & Ops Rsch 95": True,
"Information C265": True, # iSchool course on interface design
"Integrative Biology 169": True,
"Integrative Biology 192": True,
"Integrative Biology 198": True,
"Integrative Biology 77B": True,
"Integrative Biology 84": True,
"Integrative Biology 98": True,
"Integrative Biology 98BC": True,
"Integrative Biology C32": True,
"Interdisciplinary Studies 100J": False, # "The Social Life of Computing", historical and ethnographic methods
"Italian Studies R5B": False,
"Korean 10B": False,
"Korean 112": False,
"LGBT Studies 145": False,
"Landscape Arch & Env Plan 1": False, # only lists Naval Architecture and Marine Engineering.
"Latin 100": False,
"Letters & Science 22": False,  # interdisciplinary studies
"Letters & Science 25": False,  # interdisciplinary studies
"Linguistics 100": False, # only lists Cognitive Psychology and Psycholinguistics.
"Linguistics 115": False,
"Linguistics 47": False,
"Linguistics C105": False,
"Materials Science & Eng 45": True,
"Mathematics 104": True,
"Mathematics 10B": True,
"Mathematics 110": True,
"Mathematics 124": True,
"Mathematics 128A": True,
"Mathematics 152": True,
"Mathematics 160": True,
"Mathematics 16B": True,
"Mathematics 1A": True,
"Mathematics 1B": True,
"Mathematics 53": True,
"Mathematics 1B": True,
"Mathematics 53": True,
"Mathematics 54": True,
"Mathematics 55": True,
"Mathematics 98": True,
"Mathematics 98BC": True,
"Mechanical Engineering 104": True,
"Mechanical Engineering 40": True,
"Mechanical Engineering C85": True,
"Media Studies 111": False,
"Media Studies 113": False,
"Military Affairs 180": False,  # must be applied military technology to be STE
"Molecular & Cell Biology 100B": True,
"Molecular & Cell Biology 102": True,
"Molecular & Cell Biology 140": True,
"Molecular & Cell Biology 140L": True,
"Molecular & Cell Biology 198": True,
"Molecular & Cell Biology 199": True,
"Molecular & Cell Biology 38": True,
"Molecular & Cell Biology 50": True,
"Molecular & Cell Biology 90E": True,
"Molecular & Cell Biology C61": True,
"Molecular & Cell Biology C95B": True,
"Music 128": False,
"Music 159": False,
"Music 168B": False,
"Music 168C": False,
"Music 168CS": False,
"Music 170": False,
"Music 20A": False,
"Music 25": False,
"Music 27": False,
"Music 45M": False,
"Music 52A": False,
"Music 52B": False,
"Music 53A": False,
"Music 53B": False,
"Music 80": False,
"Music R1B": False,
"Near Eastern Studies 10": False,
"Near Eastern Studies 18": False,
"Nuclear Engineering 155": True,
"Nuclear Engineering 162": True,
"Nutritional Science & Tox 10S": True,
"Nutritional Science & Tox 11": True,
"Nutritional Science & Tox 160": True,
"Nutritional Science & Tox 170": True,
"Nutritional Science & Tox 190": True,
"Nutritional Science & Tox 198": True,
"Nutritional Science & Tox 20": True,
"Philosophy 104": False,
"Philosophy 121": False,
"Philosophy 12A": False,
"Philosophy 135": False,
"Philosophy 161": False,
"Philosophy 25B": False,
"Philosophy 3": False,
"Physical Education 1": False,
"Physics 112": True,
"Physics 137A": True,
"Physics 137B": True,
"Physics 7A": True,
"Physics 7B": True,
"Physics 8A": True,
"Physics 8B": True,
"Physics C21": True,
"Plant & Microbial Biology 122": True,
"Plant & Microbial Biology 40": True,
"Plant & Microbial Biology C112L": True,
"Political Science 103": False,
"Political Science 111AC": False,
"Political Science 112C": False,
"Political Science 146A": False,
"Political Science 148A": False,
"Political Science 149E": False,
"Political Science 149P": False,
"Political Science 179": False,
"Political Science 197": False,
"Political Science 2": False,
"Psychology 1": True,
"Psychology 110": True,
"Psychology 114": True,
"Psychology 130": True,
"Psychology 135": True,
"Psychology 160": True,
"Psychology 167AC": True,
"Psychology 198": True,
"Psychology 290B": True,
"Psychology C116": True,
"Psychology W1": True,
"Public Health 126": False, # only Veterinary Preventive Medicine, Epidemiology, and Public Health and Health Engineering
"Public Health 142": False,
"Public Health 150E": False,
"Public Health 198": False,
"Public Health W250B": False,
"Public Policy 101": False,
"Public Policy 157": False,
"Public Policy 192AC": False,
"Public Policy 198": False,
"Public Policy C103": False,
"Rhetoric R1B": False,
"Scandinavian 106": False,
"Slavic Languages & Lit R5B": False,
"Social Welfare 112": False,
"Social Welfare 114": False,
"Sociology 1": False,
"Sociology 127": False,
"Sociology 140": False,
"Sociology 167": False,
"Sociology 198": False,
"Sociology 3AC": False,
"Southeast Asian 148": False,
"Southeast Asian R5B": False,
"Spanish 131": False,
"Spanish 135": False,
"Statistics 133": True,
"Statistics 134": True,
"Statistics 135": True,
"Statistics 150": True,
"Statistics 20": True,
"Statistics 33B": True,
"Statistics 88": True,
"Statistics 89A": True,
"Statistics C131A": True,
"Statistics C140": True,
"Theater Dance & Perf Stds 111": False,
"Theater Dance & Perf Stds 172": False,
"Theater Dance & Perf Stds 52AC": False,
"Theater Dance & Perf Stds R1B": False,
"UGIS-UG Interdisc Studies 192A": False,
"UGIS-UG Interdisc Studies 192B": False,
"UGIS-UG Interdisc Studies 192D": False,
"UGIS-UG Interdisc Studies 192E": False,
"UGIS-UG Interdisc Studies C122": False
}

d_stem_majors = {
"Anthropology": False,
"Applied Mathematics": True,
"Architecture": False, 
"Bioengineering": True,
"Business Administration": False,
"Chemical Engineering": True,
"Chemistry": True,
"Civil & Environmental Eng": True,
"Civil Engineering": True,
"Cognitive Science": True,
"Computer Science": True,
"Economics": False,
"Electrical Eng & Comp Sci": True,
"Engineering Physics": True,
"English": False,
"Environmental Sciences": True,
"Gender & Womens Studies": False,
"Global Studies": False,
"Industrial Eng & Ops Rsch": True,
"Integrative Biology": True,
"L&S Computer Science": True,
"L&S Data Science": True,
"L&S Public Health": False,
"L&S Social Welfare": False,
"Letters & Sci Undeclared": np.NaN,
"Linguistics": False,
"MCB-Biochem & Mol Biol": True,
"MCB-Cell & Dev Biology": True,
"MCB-Genetics": True,
"MCB-Neurobiology": True,
"Mathematics": True,
"Mechanical Engineering": True,
"Media Studies": False,
"Microbial Biology": True,
"Molecular & Cell Biology": True,
"Molecular Environ Biology": True,
"Music": False,
"Nut Sci-Physio & Metabol": True,
"Nutritional Sci-Dietetics": True,
"Nutritional Sci-Toxicology": True,
"Nutritional Science": True,
"Physics": True,
"Political Economy": False,
"Political Science": False,
"Psychology": True,
"Public Health": False,
"Sociology": False,
"Statistics": True
}

# LMS Functions

# Semester start and end dates
# Reference:
# https://registrar.ANON-UNIVERSITY.DOMAIN/calendar/
semester_frames = {
    '2017 Spring': ('2017-01-10 00:00:00.000', '2017-05-12 23:59:59.999'),
    '2017 Fall': ('2017-08-15 00:00:00.000', '2017-12-15 23:59:59.999'),
    '2018 Spring': ('2018-01-09 00:00:00.000', '2018-05-11 23:59:59.999'),
    '2018 Fall': ('2018-08-15 00:00:00.000', '2018-12-14 23:59:59.999'),
    '2019 Spring': ('2019-01-15 00:00:00.000', '2019-05-17 23:59:59.999'),
    '2019 Fall': ('2019-08-21 00:00:00.000', '2019-12-20 23:59:59.999'),
    '2020 Spring': ('2020-01-14 00:00:00.000', '2020-05-15 23:59:59.999'),
    '2020 Fall': ('2020-08-19 00:00:00.000', '2020-12-18 23:59:59.999'),
    '2021 Spring': ('2021-01-12 00:00:00.000', '2021-05-14 23:59:59.999')
}

# Preprocessing

def lms_preproc(d: dict, semester_start='2021-01-18 00:00:00.000', semester_end='2021-05-13 23:59:59.999'):
    temp = d['enrollments'][['course_id', 'user_id', 'enrollment_role_type']]

    # Filter Students, Teachers, TAs
    temp = temp[temp['enrollment_role_type'].isin(['StudentEnrollment', 'TeacherEnrollment', 'TaEnrollment'])]
    temp['enrollment_role_type'] = temp['enrollment_role_type'].str.replace('Enrollment', '') # make string cleaner

    # Get count of roles of users across all courses
    check = temp.groupby(['user_id','course_id', 'enrollment_role_type']).size().reset_index().rename(columns={0:'count'})

    # Check if any users have more than one role in any course
    check = check.groupby(['user_id','course_id']).size().reset_index().rename(columns={0:'count'})

    # Drop duplicates of these 924 instances as follows: If user has been a teacher or TA at some point,
    # assign teacher or TA. Reason: Teachers might have been enrolling in their course as students for testing
    # purposes

    # Sort data frame such that Teacher and TA enrollment appear first
    temp = temp.sort_values(by='enrollment_role_type', ascending=False)

    # Drop duplicates such that first unique combination with Teacher or TA is kept
    temp = temp.drop_duplicates(subset=['course_id', 'user_id'], keep='first', inplace=False)

    user_role_reference_table = temp
    # Simplify variable name
    user_role_reference_table = user_role_reference_table.rename({'enrollment_role_type': 'user_role'}, axis=1)
    
    d['discussion_entry'] = d['discussion_entry'].merge(user_role_reference_table, on=['course_id', 'user_id'], how='left')
    d['submissions'] = d['submissions'].merge(user_role_reference_table, on=['course_id', 'user_id'], how='left')
    d['submission_comments'] = d['submission_comments'].rename({'author_id': 'user_id'}, axis=1) # fix user id var name
    d['submission_comments']['user_id'] = d['submission_comments'].user_id.fillna(0).astype(int) # fix encoding for joining
    d['submission_comments'] = d['submission_comments'].merge(user_role_reference_table, on=['course_id', 'user_id'], how='left')
    
    # Take most recent enrollment state of each user in each course
    temp = d['enrollments'][['course_id', 'user_id', 'enrollment_updated_at', 'enrollment_state']]
    temp = temp.sort_values(by=['enrollment_updated_at', 'course_id', 'user_id'], ascending=False)
    temp = temp.drop_duplicates(subset=['course_id', 'user_id'], keep='first', inplace=False)
    enrollment_status = temp
    
    # Join last updated status including timestamp to tables
    d['discussion_entry'] = d['discussion_entry'].merge(enrollment_status, on=['course_id', 'user_id'], how='left')
    d['submissions'] = d['submissions'].merge(enrollment_status, on=['course_id', 'user_id'], how='left')
    d['submission_comments'] = d['submission_comments'].merge(enrollment_status, on=['course_id', 'user_id'], how='left')
    
    # Create relevant variables
    # A: If user is student and last updated enrollment status is deleted, then assign dropout status
    #    If user is not student, assign -1, if user is student and not a dropout, assign 0 (active or completed)
    # B: If user is a dropout student, return last updated enrollment status as time of dropout, else assign NA

    def student_dropout_conditions(row):
        if row['user_role'] != 'Student':
            return -1
        else:
            if row['enrollment_state'] in ['active', 'completed']:
                return 0
            elif row['enrollment_state'] == 'deleted':
                return 1
            else:
                return np.nan

    def dropout_at_conditions(row):
        if row['is_student_dropout'] != 1:
            return np.nan
        else:
            return row['enrollment_updated_at']

    d['discussion_entry']['is_student_dropout'] = d['discussion_entry'].apply(student_dropout_conditions, axis=1)
    # Submission data misses for some semesters
    try:
        d['submissions']['is_student_dropout'] = d['submissions'].apply(student_dropout_conditions, axis=1)
    except:
        pass
    d['submission_comments']['is_student_dropout'] = d['submission_comments'].apply(student_dropout_conditions, axis=1)

    d['discussion_entry']['dropout_at'] = d['discussion_entry'].apply(dropout_at_conditions, axis=1)
    # Submission data misses for some semesters
    try:
        d['submissions']['dropout_at'] = d['submissions'].apply(dropout_at_conditions, axis=1)
    except:
        pass
    d['submission_comments']['dropout_at'] = d['submission_comments'].apply(dropout_at_conditions, axis=1)
    
    # Assignments, due and unlock dates
    updated_due_dates = d['assignments_overrides'].sort_values(by=['updated_at'], ascending=False)  # sort by most recent
    updated_due_dates = updated_due_dates.loc[updated_due_dates['due_at'].notnull(), ['assignment_id', 'due_at']]
    updated_due_dates.assignment_id = updated_due_dates.assignment_id.fillna(0).astype(int)
    updated_due_dates = updated_due_dates.drop_duplicates(subset = 'assignment_id', keep = 'first') # keep most recent

    # Join most recent entries to main table
    d['assignments'] = pd.merge(d['assignments'], updated_due_dates, on='assignment_id', how='left')

    # Take most recent due_at if available, else take asn_due_at from original table
    d['assignments']['due_at_correct'] = d['assignments'][['asn_due_at', 'due_at']].apply(lambda x: x['asn_due_at'] if pd.isnull(x['due_at']) else x['due_at'], axis=1)

    d['assignments']['due_at_correct'] = pd.to_datetime(d['assignments']['due_at_correct'], errors = 'coerce')
    
    updated_unlock_dates = d['assignments_overrides'].sort_values(by=['updated_at'], ascending=False)
    updated_unlock_dates = updated_unlock_dates.loc[updated_unlock_dates['unlock_at'].notnull(), ['assignment_id', 'unlock_at']]
    updated_unlock_dates.assignment_id = updated_unlock_dates.assignment_id.fillna(0).astype(int)
    updated_unlock_dates = updated_unlock_dates.drop_duplicates(subset = 'assignment_id', keep = 'first')

    d['assignments'] = pd.merge(d['assignments'], updated_unlock_dates, on='assignment_id', how='left')
    d['assignments']['unlock_at_updated'] = d['assignments'][['asn_unlock_at', 'unlock_at']].apply(lambda x: x['asn_unlock_at'] if pd.isnull(x['unlock_at']) else x['unlock_at'], axis=1)

    d['assignments']['unlock_at_updated'] = pd.to_datetime(d['assignments']['unlock_at_updated'], errors = 'coerce')
    
    d['assignments'] = d['assignments'][d['assignments'].workflow_state == 'published']
    
    # Submission data misses for some semesters
    if d['submissions'].shape[0] > 0:
        assignment_ids_with_submissions = set(d['submissions'][
                                    (d['submissions'].user_role == 'Student') & 
                                    (d['submissions'].assignment_id.isin(d['assignments'].assignment_id))]\
                                          .assignment_id)

        n_assignments_with_submissions = len(assignment_ids_with_submissions)
        n_assignments = len(pd.unique(d['assignments'].assignment_id))

        d['assignments'] = d['assignments'][d['assignments'].assignment_id.isin(assignment_ids_with_submissions)]
    
    semester_start = pd.to_datetime(semester_start, errors = 'coerce')
    d['assignments'] = d['assignments'].loc[~(
        (d['assignments'].due_at_correct.notna()) &
        (d['assignments'].due_at_correct < semester_start)
    ),]
    
    d['assignments_with_due'] = d['assignments'][d['assignments'].due_at_correct.notna()]
    d['assignments_with_due-unlock'] = d['assignments_with_due'][d['assignments_with_due'].unlock_at_updated.notna()]
    
    # Join course_name_number and section_num to submissions table
    canvas_courses = d['course_section'][
        ['canvas_course_global_id', 'course_subject_name_number', 'section_num']
    ]
    canvas_courses.columns = ['course_id', 'course_name_number', 'section_num']

    d['submissions'] = d['submissions'].merge(canvas_courses, on='course_id', how='left')
    d['submission_comments'] = d['submission_comments'].merge(canvas_courses, on='course_id', how='left')
    d['assignments'] = d['assignments'].merge(canvas_courses, on='course_id', how='left')
    d['assignments_with_due'] = d['assignments_with_due'].merge(canvas_courses, on='course_id', how='left')
    d['assignments_with_due-unlock'] = d['assignments_with_due-unlock'].merge(canvas_courses, on='course_id', how='left')
    d['discussion_entry'] = d['discussion_entry'].merge(canvas_courses, on='course_id', how='left')
    d['enrollments'] = d['enrollments'].merge(canvas_courses, on='course_id', how='left')
    
    
    # The dates for the Spring 2021 semester are January 18, 2021 to May 13, 2021.
    semester_start = pd.to_datetime(semester_start, errors = 'coerce')
    semester_end = pd.to_datetime(semester_end, errors = 'coerce')
    semester_quarter_limits = pd.date_range(semester_start, semester_end, periods=5)

    # Take most recent enrollment state of each user in each course
    temp = d['enrollments'][['course_name_number', 'section_num', 'user_id', 'enrollment_updated_at', 'enrollment_state', 'enrollment_role_type']]
    temp = temp.sort_values(by=['enrollment_updated_at', 'course_name_number', 'section_num', 'user_id'], ascending=False)
    temp = temp.drop_duplicates(subset=['course_name_number', 'section_num', 'user_id'], keep='first', inplace=False)

    # Filter students
    temp = temp[temp['enrollment_role_type'] == 'StudentEnrollment']

    # If students dropped out before start of Spring Semester, do not impute Spring Semester start date
    # If students dropped out after end of Spring Semester, do not impute Spring Semester end date

    # Rather, remove students from the calculation which dropped out outside of the semester
    temp['enrollment_updated_at'] = pd.to_datetime(temp['enrollment_updated_at'], errors = 'coerce')
    temp = temp[(temp.enrollment_updated_at >= semester_start) & (temp.enrollment_updated_at <= semester_end)]

    # Create "dropped out in quarter n" binary variables for each quarter
    temp['dropped_out_q1'] = (temp['enrollment_state'] == 'deleted') & (
        semester_quarter_limits[0] <= temp['enrollment_updated_at']) & (
        temp['enrollment_updated_at'] <= semester_quarter_limits[1])
    temp['dropped_out_q2'] = (temp['enrollment_state'] == 'deleted') & (
        semester_quarter_limits[1] <= temp['enrollment_updated_at']) & (
        temp['enrollment_updated_at'] <= semester_quarter_limits[2])
    temp['dropped_out_q3'] = (temp['enrollment_state'] == 'deleted') & (
        semester_quarter_limits[2] <= temp['enrollment_updated_at']) & (
        temp['enrollment_updated_at'] <= semester_quarter_limits[3])
    temp['dropped_out_q4'] = (temp['enrollment_state'] == 'deleted') & (
        semester_quarter_limits[3] <= temp['enrollment_updated_at']) & (
        temp['enrollment_updated_at'] <= semester_quarter_limits[4])

    d['temp_dropout_reference'] = temp

    # Get teaching staff reference

    # Take most recent enrollment state of each user in each course
    temp = d['enrollments'][['course_name_number', 'section_num', 'user_id', 'enrollment_updated_at', 'enrollment_state', 'enrollment_role_type']]
    temp = temp.sort_values(by=['enrollment_updated_at', 'course_name_number', 'section_num', 'user_id'], ascending=False)
    temp = temp.drop_duplicates(subset=['course_name_number', 'section_num', 'user_id'], keep='first', inplace=False)

    # Filter teaching staff
    temp = temp[temp['enrollment_role_type'].isin(['TeacherEnrollment', 'TaEnrollment'])]

    d['temp_teaching_staff_reference'] = temp
    
    return d

    
# Features

def get_assignment_spread(df, name1: str, section1: str, section2: list) -> float:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if temp.shape[0] in [0, 1]: # standard deviation requires 2+ data points
        return 0
    else:
        return temp.due_at_correct.astype(int).std()

# parallel assignments
## Current approach which adds a time period in front of assignment deadlines and counts pair-wise overlap ##

# For all courses
# 1. Create a timeframe for each assignment from deadline-1day to deadline
# 2. Count number of timeframes that overlap in each course

def check_overlap(tuple1, tuple2, i, j):
    if i==j: # a timeframe will always overlap itself
        return False
    # this condition for overlap holds independent of which timeframe starts earlier
    elif tuple1[0] < tuple2[1] and tuple2[0] < tuple1[1]:
        return tuple(sorted([i, j])) # sort in order to filter out inverse later
    else:
        return False

def get_parallel_assingments(df, name1: str, section1: str, section2: list, grace_period_days: int) -> int:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if temp.shape[0] == 0:
        return np.nan
    else:
        temp['asn_frame_lag'] = temp['due_at_correct'] - timedelta(days=grace_period_days)
        temp['asn_frame'] = list(zip(temp.asn_frame_lag, temp.due_at_correct))
        out = []
        for i, timeframe1 in enumerate(temp.asn_frame):
            for j, timeframe2 in enumerate(temp.asn_frame):
                out.append(check_overlap(timeframe1, timeframe2, i, j))
        out = set([element for element in out if element != False and element is not None]) # casting to set drops inverse 
        return len(out)

# Flexible approach based on graded (3 days) or not graded (1 day)

def timeframe_conditions(row):
    res = 1 # 1 day, extend by factors and return value
    if row['grading_type'] in ['points', 'percent', 'letter_grade', 'gpa_scale']:
        res *= 3 
    return row['due_at_correct'] - timedelta(days=res)

def get_parallel_assingments_flexible(df, name1: str, section1: str, section2: list) -> int:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if temp.shape[0] == 0:
        return np.nan
    else:
        temp['asn_frame_start'] = temp.apply(timeframe_conditions, axis=1)
        temp['asn_frame'] = list(zip(temp.asn_frame_start, temp.due_at_correct))
        out = []
        for i, timeframe1 in enumerate(temp.asn_frame):
            for j, timeframe2 in enumerate(temp.asn_frame):
                out.append(check_overlap(timeframe1, timeframe2, i, j))
        out = set([element for element in out if element != False and element is not None]) # casting to set drops inverse 
        return len(out)

def get_n_course_assignments(df, name1: str, section1: str, section2: list, graded_only=False) -> int:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if graded_only:
        temp = temp[temp['grading_type'].isin(['points', 'percent', 'letter_grade', 'gpa_scale'])]
    return 0 if temp.shape[0] == 0 else temp.shape[0]

def get_graded_assignments_week(df, week_start_dates, week_end_dates, name1: str, section1: str, section2: list,
                                metric='average'):
    """
    Parse 'average' or 'max' as metric to get either the average number of graded assignments per weke
    or the maximum number of assignments during the whole semester which was due in a single calendar week.
    """
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    temp = temp[temp['grading_type'].isin(['points', 'percent', 'letter_grade', 'gpa_scale'])]
    
    assignments_due_per_week_list = []
    
    for week_start, week_end in zip(week_start_dates, week_end_dates):
        subset = temp[(temp['due_at_correct'] < week_end) & (week_start < temp['due_at_correct'])] # due this week only
        assignments_due_per_week_list.append(subset.shape[0])

    if metric == 'average':
        return sum(assignments_due_per_week_list)/len(assignments_due_per_week_list)
    elif metric == 'max':
        return max(assignments_due_per_week_list)
    else: raise ArgumentError('Please parse either "average" or "max" to argument "metric"')
    

def get_avg_submission_time_to_deadline_minutes(df, name1: str, section1: str, section2: list, dropout_status=0) -> float:
    temp_assignments = df['assignments_with_due'][(df['assignments_with_due'].course_name_number==name1) & (df['assignments_with_due']['section_num'].isin([section1] + section2))]
    temp_submission = df['submissions'][(df['submissions'].course_name_number==name1) & (df['submissions']['section_num'].isin([section1] + section2))]
    if temp_assignments.shape[0] == 0 or temp_submission.shape[0] == 0: 
        return 0
    else:
        join_this = temp_submission[
            (temp_submission.user_role == 'Student') &
            (temp_submission.is_student_dropout == dropout_status) 
        ][['assignment_id', 'submitted_at']]
        temp = pd.merge(temp_assignments[['course_id', 'assignment_id', 'due_at_correct']], 
                        join_this, on='assignment_id', how='left')

        # Where possible, create average timeframe difference in minutes
        temp['submitted_at'] = pd.to_datetime(temp['submitted_at'])
        #temp = temp.dropna() 
        
        temp['submission_diff'] = temp['due_at_correct'] - temp['submitted_at']
        
        return temp.submission_diff.dt.total_seconds().mean()/60 


def get_early_assignment_availability_ratio(df, semester_start_plus_two_weeks,
                                            name1: str, section1: str, section2: list) -> float:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if temp.shape[0] == 0:
        return np.nan
    else:
        return sum(temp['unlock_at_updated'] <= semester_start_plus_two_weeks) / temp.shape[0]

def get_avg_diff_available_due_assignments(df, name1: str, section1: str, section2: list) -> float:
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    if temp.shape[0] == 0:
        return 0
    else:
        temp['diff_available_due'] = temp['due_at_correct'] - temp['unlock_at_updated']
        return temp.diff_available_due.dt.total_seconds().mean()/60 
    
# submission_comments_avg_size_bytes
def get_submission_comments_avg_size_bytes(df, name1: str, section1: str, section2: list) -> float:
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1] + section2))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        all_submission_comments = temp#[ user_roles could not be assigned to author in 90% of cases -> all comments will be used since variable will be NA in most instances otherwise
            #temp.user_role.isin(['Ta', 'Teacher'])
        #] 
        if all_submission_comments.shape[0] == 0: # if there are submission comments, we can not divide by 0
            return 0 
        else:
            return sum(all_submission_comments.message_size_bytes) / all_submission_comments.shape[0]

def get_submission_comments_per_student(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function divides the number of submission comments made by TAs and teachers by the
    number of students in a course, regardless of the dropout status of students.
    """
    temp_comments = df['submission_comments'][(df['submission_comments']['course_name_number']==name1) & (df['submission_comments']['section_num'].isin([section1] + section2))]
    temp_submission = df['submissions'][(df['submissions']['course_name_number']==name1) & (df['submissions']['section_num'].isin([section1] + section2))]
    if temp_submission.shape[0] == 0: 
        return 0
    else:
        n_students_course = len(set(temp_submission[
            (temp_submission.user_role == 'Student') #&
            #(temp_submission.is_student_dropout == 0)
        ].user_id))
        if n_students_course == 0: # if no students are enrolled, we can not divide by 0
            return 0
        else:
            all_submission_comments = temp_comments#[ # user roles could not be reliably added to submission comments table
                #(temp_comments.user_role.isin(['Ta', 'Teacher']))
            #]
            return all_submission_comments.shape[0]/n_students_course

def get_percent_submissions_submission_comments(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function creates the intersection of submission IDs by students and submission comment
    parents IDs and returns the ratio of submission IDs by students which received submission comments.
    """
    # user roles could not be reliably added to submission comments table
    temp_comments = df['submission_comments'][(df['submission_comments']['course_name_number']==name1) & (df['submission_comments']['section_num'].isin([section1] + section2))]
    temp_submission = df['submissions'][(df['submissions']['course_name_number']==name1) & (df['submissions']['section_num'].isin([section1] + section2))]
    if temp_submission.shape[0] == 0: 
        return np.nan
    else:
        # Get all submission IDs of submissions made by students that did not drop out
        all_orig_student_submission_ids = set(pd.unique(temp_submission.submission_id).tolist())

        # Get parent references from submission comments
        all_parent_submission_comment_ids = set(pd.unique(temp_comments.submission_id).tolist())
        
        if len(all_orig_student_submission_ids) == 0: # can not divide by 0
            return np.nan
        else:
            return len(all_orig_student_submission_ids & all_parent_submission_comment_ids) / len(all_orig_student_submission_ids)
        
def get_n_enrollments(df, name1: str, section1: str) -> float:
    temp = df[(df['course_name_number']==name1) & (df['section_num'] == section1)]
    if temp.shape[0] == 0: # if no enrollment records are available, return NA
        return np.nan
    else:
        return temp.shape[0]

# dropped_out_ratio_q{1,2,3,4}
def get_dropped_out_ratio(df, name1: str, section1: str, reference_var_quarter: str) -> float:
    """
    Calculates ratio of student dropout as a fraction of the total number of students that originally
    enrolled in the course. 
    Please parse dropped_out_q{1,2,3,4} to variables 'reference_var_quarter'.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'] == section1)]
    if temp.shape[0] == 0: # if no enrollment records are available, return NA
        return np.nan
    else:
        return temp[temp[reference_var_quarter] == True].shape[0] / temp.shape[0]
    
def get_n_original_forum_posts(df, name1: str, section1: str, section2: str) -> int:
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        all_original_posts = temp[
            (temp.depth == 1) & 
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 0)]
        return all_original_posts.shape[0]

def get_n_original_forum_posts_dropout(df, name1: str, section1: str, section2: str) -> int:
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        all_original_posts = temp[
            (temp.depth == 1) & 
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 1)]
        return all_original_posts.shape[0]

def get_original_student_post_avg_size_bytes(df, name1: str, section1: str, section2: str) -> float:
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        all_original_posts = temp[
            (temp.depth == 1) & 
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 0)] 
        if all_original_posts.shape[0] == 0: # if there are no original posts by students, we can not divide by 0
            return 0
        else:
            return sum(all_original_posts.message_length) / all_original_posts.shape[0]

def get_original_student_post_avg_size_bytes_dropout(df, name1: str, section1: str, section2: str) -> float:
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        all_original_posts = temp[
            (temp.depth == 1) & 
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 1)] 
        if all_original_posts.shape[0] == 0: # if there are no original posts by students, we can not divide by 0
            return 0
        else:
            return sum(all_original_posts.message_length) / all_original_posts.shape[0]

def get_original_forum_posts_per_student(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function divides the number of original posts made by students that did not drop out
    by the number of students that did not drop out and made at least one forum post (on all levels) in either 
    course section.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        n_students_course = len(set(temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 0)
        ].user_id))
        if n_students_course == 0: # if no students participated in forum, we can not divide by 0
            return 0
        else :
            all_original_posts = temp[
                (temp.depth == 1) & 
                (temp.user_role == 'Student') &
                (temp.is_student_dropout == 0)]
            return all_original_posts.shape[0]/n_students_course

def get_original_forum_posts_per_student_dropout(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function divides the number of original posts made by students that did drop out
    by the number of students that did drop out and made at least one forum post (on all levels) in either 
    course section.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        n_students_course = len(set(temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 1)
        ].user_id))
        if n_students_course == 0: # if no students participated in forum, we can not divide by 0
            return 0
        else :
            all_original_posts = temp[
                (temp.depth == 1) & 
                (temp.user_role == 'Student') &
                (temp.is_student_dropout == 1)]
            return all_original_posts.shape[0]/n_students_course

def get_ta_teacher_posts_per_student(df, name1: str, section1: str, section2: str) -> float:
    """
    Note: This function divides the number of posts made by TAs or teachers
    by the number of students (regardless of dropout status)  that made at least one forum post 
    (on all levels) in either course section.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return 0
    else:
        n_students_course = len(set(temp[
            (temp.user_role == 'Student')
        ].user_id))
        if n_students_course == 0: # if no students participated in forum, we can not divide by 0
            return 0
        else :
            all_original_teacher_posts = temp[temp.user_role.isin(['Ta', 'Teacher'])]
            return all_original_teacher_posts.shape[0]/n_students_course

def get_ta_teacher_reply_time(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function returns the average reply time of TAs and teachers to posts by students 
    IF posts by students received a reply by a TA or teacher. Students are defined as students
    that did not drop out. A separate variable for students who dropped out will be created.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return np.nan
    else:
        reference_ids = temp[
            temp.depth != 1 & 
            temp.user_role.isin(['Ta', 'Teacher'])
        ][['course_id', 'parent_discussion_entry_id', 'discussion_entry_id', 'created_at']]
        
        # Get posting dates of parent IDs, if they are posts by students and by non-dropouts
        join_this = temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 0)
        ][['discussion_entry_id', 'created_at']]
        
        # If parent to teacher reply is a post by  student non-dropout, join creation date of parent post
        diff = pd.merge(reference_ids, join_this, 
                 how='left', left_on=['parent_discussion_entry_id'], right_on=['discussion_entry_id'])
        
        # If parent is not such a post, an NA is joined, which is then omitted:
        diff = diff.dropna()
        
        if diff.shape[0] == 0:  # if no instance of reply time is available for neither section, then return NA
            return np.nan

        else:
            # Rename vars
            del diff['discussion_entry_id_y'] # redundant, this is the parent ID from the ta/teacher reply
            diff = diff.rename({'discussion_entry_id_x': 'discussion_entry_id', 'created_at_x': 'created_at_reply', 
                       'created_at_y': 'created_at_parent'}, axis=1)

            diff['created_at_reply'] = pd.to_datetime(diff['created_at_reply'], errors = 'coerce')
            diff['created_at_parent'] = pd.to_datetime(diff['created_at_parent'], errors = 'coerce')
            diff['reply_time_minutes'] = diff.apply(lambda x: (x['created_at_reply']-x['created_at_parent']), axis=1)
            
            return diff['reply_time_minutes'].dt.total_seconds().mean()/60 # gets actually converted to mins here

def get_ta_teacher_reply_time_dropout(df, name1: str, section1: str, section2: list) -> float:
    """
    Note: This function returns the average reply time of TAs and teachers to posts by students 
    IF posts by students received a reply by a TA or teacher. Students are defined as students
    that dropped out.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return np.nan
    else:
        reference_ids = temp[
            temp.depth != 1 & 
            temp.user_role.isin(['Ta', 'Teacher'])
        ][['course_id', 'parent_discussion_entry_id', 'discussion_entry_id', 'created_at']]
        
        # Get posting dates of parent IDs, if they are posts by students and by non-dropouts
        join_this = temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 1)
        ][['discussion_entry_id', 'created_at']]
        
        # If parent to teacher reply is a post by  student non-dropout, join creation date of parent post
        diff = pd.merge(reference_ids, join_this, 
                 how='left', left_on=['parent_discussion_entry_id'], right_on=['discussion_entry_id'])
        
        # If parent is not such a post, an NA is joined, which is then omitted:
        diff = diff.dropna()
        
        if diff.shape[0] == 0:  # if no instance of reply time is available for neither section, then return NA
            return np.nan

        else:
            # Rename vars
            del diff['discussion_entry_id_y'] # redundant, this is the parent ID from the ta/teacher reply
            diff = diff.rename({'discussion_entry_id_x': 'discussion_entry_id', 'created_at_x': 'created_at_reply', 
                       'created_at_y': 'created_at_parent'}, axis=1)

            diff['created_at_reply'] = pd.to_datetime(diff['created_at_reply'], errors = 'coerce')
            diff['created_at_parent'] = pd.to_datetime(diff['created_at_parent'], errors = 'coerce')
            diff['reply_time_minutes'] = diff.apply(lambda x: (x['created_at_reply']-x['created_at_parent']), axis=1)
            
            return diff['reply_time_minutes'].dt.total_seconds().mean()/60 # gets actually converted to mins here

def get_reply_ratio(df, name1: str, section1: str, section2: str) -> float:
    """
    Note: This function returns the ratio of posts made by students that did not drop out
    that received a reply by any user (TA/teacher/students*) which also includes replies
    by students that dropped out because posts that already received a reply by these students
    are less likely to receive another reply.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return np.nan
    else:
        all_orig_student_post_ids = set(temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 0)
        ].discussion_entry_id)
        
        if len(all_orig_student_post_ids) == 0: # if there are not posts made by students we can not divide by 0
            return np.nan
        else: 
            all_parent_post_ids = set(temp[
                (temp.user_role.isin(['Student', 'Ta', 'Teacher']))
            ].parent_discussion_entry_id)

            # The intersection of post IDs and parent post IDs represent those student posts that received replies
            return len(all_orig_student_post_ids & all_parent_post_ids)/len(all_orig_student_post_ids)

def get_reply_ratio_dropout(df, name1: str, section1: str, section2: str) -> float:
    """
    Note: This function returns the ratio of posts made by students that did drop out
    that received a reply by any user (TA/teacher/students*) which also includes replies
    by students that did not drop out because posts that already received a reply by these students
    are less likely to receive another reply.
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no forum data available on neither section, return NA
        return np.nan
    else:
        all_orig_student_post_ids = set(temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == 1)
        ].discussion_entry_id)
        
        if len(all_orig_student_post_ids) == 0: # if there are not posts made by students we can not divide by 0
            return np.nan
        else: 
            all_parent_post_ids = set(temp[
                (temp.depth != 1) & 
                (temp.user_role.isin(['Student', 'Ta', 'Teacher']))
            ].parent_discussion_entry_id)

            # The intersection of post IDs and parent post IDs represent those student posts that received replies
            return len(all_orig_student_post_ids & all_parent_post_ids)/len(all_orig_student_post_ids)
        
def get_n_course_assignments_deadline_unlock(df, name1: str, section1: str, section2: list, graded_only=False) -> int:
    """
    Returns the number of course assignments with deadlines or deadlines and unlock dates.
    """
    temp = df[(df.course_name_number==name1) & (df.section_num.isin([section1] + section2))]
    return temp.shape[0]

def get_n_submissions_students(df, name1: str, section1: str, section2: str) -> float:
    """
    This function returns the total number of submissions by students regardless of dropout
    status to ascertain whether the submission feature was used in individual canvas courses
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no submission data available on neither section, return 0
        return 0
    else:
        all_submissions = temp[
            (temp.user_role == 'Student')
        ]
        return all_submissions.shape[0]

def get_n_submissions_students_by_dropout(df, name1: str, section1: str, section2: str, dropout_status=0) -> float:
    """
    This function returns the total number of submissions by students by dropout
    status to ascertain whether the submission feature was used in individual canvas courses
    """
    temp = df[(df['course_name_number']==name1) & (df['section_num'].isin([section1, section2]))]
    if temp.shape[0] == 0: # if no submission data available on neither section, return 0
        return 0
    else:
        all_submissions = temp[
            (temp.user_role == 'Student') &
            (temp.is_student_dropout == dropout_status)
        ]
        return all_submissions.shape[0]
    
def read_LMS_data(reference='2021 Spring'):
    if reference == '2021 Spring':
        path = '../research-data/LMS_Spring2021'
    elif reference in ['2020 Spring', '2019 Spring', '2018 Spring', '2017 Spring',
                       '2020 Fall', '2019 Fall', '2018 Fall', '2017 Fall']:
        path = '../remote_data/unzipped/phase2_revised/final/'
    else:
        return
    
    filelist = []
    for root, _, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    d = {}
    for f in filelist: 
        if "000" not in f:
            continue
        entry_name = re.sub(r'_000$|000$', '', f.rsplit('/', 1)[-1])
        entry = pd.read_csv(f, delimiter='\t', low_memory=False)
        if 'term_name' not in entry.columns:
            continue
        else:
            d[entry_name] = entry[entry['term_name']==reference].copy()
            
    # Get course reference for df.iterrows() methods
    courses = d['course_section'][['course_subject_name_number', 'section_num']]
    courses.columns = ['course_name_number', 'section_num']

    courses_primary_sections = courses[courses['section_num'] == '001'].copy()
    courses_other_sections = courses[courses['section_num'] != '001'].copy()
    courses_other_sections = courses_other_sections\
        .groupby('course_name_number')\
        .section_num.apply(lambda s: str(list(s)))\
        .reset_index()\
        .rename(columns={'section_num': 'secondary_section_number'})
    courses_all_sections = courses\
        .groupby('course_name_number')\
        .section_num.apply(lambda s: str(list(s)))\
        .reset_index()\
        .rename(columns={'section_num': 'all_section_numbers'})
    processed = courses_primary_sections\
        .merge(courses_other_sections, on='course_name_number', how='left')\
        .fillna("[]")\
        .merge(courses_all_sections, on='course_name_number', how='left')\
        .fillna("[]")
    
    d['COURSE_REFERENCE'] = processed.copy()
        
    return d
    
def run_course_feature_engineering(reference='2021 Spring', outf='../research-data/processed/lms-testing.csv',
            semester_start='2021-01-18 00:00:00.000', semester_end='2021-05-13 23:59:59.999',
            aggregate_indiv_variables=False, verbose=True, debug=False):
    
    # Read in data
    d = read_LMS_data(reference=reference)
    if verbose: print(f'Read in data for {reference}!')
    
    # Preproc
    d = lms_preproc(d, semester_start=semester_start, semester_end=semester_end)
    if verbose: print(f'Preprocessed data for {reference}!')
    
    # Extract COURSE_REFERENCE
    if debug:
        COURSE_REFERENCE = d['COURSE_REFERENCE'].iloc[:5].copy()
    else:
        COURSE_REFERENCE = d['COURSE_REFERENCE'].copy()
    
    # Semester weeks helper vars
    semester_start = pd.to_datetime(semester_start, errors = 'coerce')
    semester_end = pd.to_datetime(semester_end, errors = 'coerce')

    weeks = []
    while semester_start <= semester_end:
        weeks.append(semester_start)
        semester_start += timedelta(days=7)

    week_start_dates = weeks[:-1]
    week_end_dates = weeks[1:]

    semester_start_plus_two_weeks = pd.to_datetime(semester_start, errors = 'coerce') + timedelta(days=14)
    
    if verbose: print(f'Creating variables for {reference}...')
        
    if aggregate_indiv_variables:  
        df_majors = pd.read_csv('../edw_askoski_student_majors_hashed.txt', sep='|', low_memory=False)
        df_grades = pd.read_csv('../edw_askoski_student_grades_hashed.txt', sep='|', low_memory=False)
        df_prereqs = pd.read_csv('../research-data/course_prereqs.tsv')
        
        # Course to department mapping
        d_cid_dept = df_grades\
            .groupby('COURSE_SUBJECT_NAME_NUMBER')\
            ['CRS_ACADEMIC_DEPT_SHORT_NM']\
            .apply(set)\
            .to_dict()

        d_cid_stem = {k: d_dept_stem[list(v)[0]] for k, v in d_cid_dept.items()}

        # Major to department mapping
        d_major_dept = df_majors\
            .groupby('MAJOR_NAME')\
            ['ACADEMIC_DEPARTMENT_NAME']\
            .apply(set)\
            .to_dict()

        d_major_stem = {k: d_dept_stem[list(v)[0]] for k, v in d_major_dept.items()}

        # Mapping from course_id to prereq list
        d_cid_prereqs = df_prereqs[['cid', 'course_prereqs']].set_index('cid').to_dict()['course_prereqs']
        d_cid_prereqs = {k: v for k, v in d_cid_prereqs.items() if v!='[]'} # simplify

        # Mapping from course_id to student_id enrollement in semester
        d_cid_anonid = df_grades[df_grades['#SEMESTER_YEAR_NAME_CONCAT'] == reference]\
            .groupby('COURSE_SUBJECT_NAME_NUMBER')\
            ['ANON_ID']\
            .apply(list)\
            .to_dict()

        # For all students in values of above dict, taken courses in semester
        d_anonid_cid_sem = df_grades[df_grades['#SEMESTER_YEAR_NAME_CONCAT'] == reference]\
            .groupby('ANON_ID')\
            ['COURSE_SUBJECT_NAME_NUMBER']\
            .apply(list)\
            .to_dict()

        # For all students in values of above dict, taken courses in all past semesters
        d_anonid_cid_all = df_grades\
            .groupby('ANON_ID')\
            ['COURSE_SUBJECT_NAME_NUMBER']\
            .apply(list)\
            .to_dict()

        # For all students in values of above dict, majors and departments
        d_anonid_majors_sem = df_majors[df_majors['#SEMESTER_YEAR_NAME_CONCAT'] == reference]\
            .groupby('ANON_ID')\
            ['MAJOR_NAME']\
            .apply(set)\
            .to_dict()

        d_anonid_depts_sem = df_majors[df_majors['#SEMESTER_YEAR_NAME_CONCAT'] == reference]\
            .groupby('ANON_ID')\
            ['ACADEMIC_DEPARTMENT_NAME']\
            .apply(set)\
            .to_dict()

        # For all students in values of above dict, GPA
        # Adjust for >>> '2021 Spring' > '2021 Fall' -> True
        if 'Spring' in reference:
            d_anon_gpa = df_grades[(df_grades['#SEMESTER_YEAR_NAME_CONCAT'] <= reference) & 
                      (df_grades['#SEMESTER_YEAR_NAME_CONCAT'] != reference[:4]+' Fall') & 
                      (df_grades['GRADE_TYPE_DESC'] == 'Letter Grade')]\
                .groupby('ANON_ID')\
                [['GRADE_POINTS_NBR', 'STUDENT_CREDIT_HRS_NBR']]\
                .apply(lambda grades: np.sum(grades['GRADE_POINTS_NBR'] * grades['STUDENT_CREDIT_HRS_NBR']) / np.sum(grades['STUDENT_CREDIT_HRS_NBR']))\
                .to_dict()
        else:
            d_anon_gpa = df_grades[(df_grades['#SEMESTER_YEAR_NAME_CONCAT'] <= reference[:4]+' Spring') & 
                      (df_grades['GRADE_TYPE_DESC'] == 'Letter Grade')]\
                .groupby('ANON_ID')\
                [['GRADE_POINTS_NBR', 'STUDENT_CREDIT_HRS_NBR']]\
                .apply(lambda grades: np.sum(grades['GRADE_POINTS_NBR'] * grades['STUDENT_CREDIT_HRS_NBR']) / np.sum(grades['STUDENT_CREDIT_HRS_NBR']))\
                .to_dict()

        # Helper variable for filtering: is_major
        df_grades['ANON_ID_MAJOR'] = df_grades['ANON_ID'].map(d_anonid_majors_sem)
        df_grades['ANON_ID_DEPT'] = df_grades['ANON_ID'].map(d_anonid_depts_sem)

        def get_is_major_helper(df):
            try:
                res = len(set(df['ANON_ID_MAJOR']) | set(df['ANON_ID_DEPT']) &\
                      set(df['COURSE_SUBJECT_SHORT_NM']) | set(df['CRS_ACADEMIC_DEPT_SHORT_NM'])) > 0
            except:
                res = False
            return res

        df_grades['IS_MAJOR'] = df_grades[['ANON_ID_MAJOR', 'ANON_ID_DEPT', 'COURSE_SUBJECT_SHORT_NM', 'CRS_ACADEMIC_DEPT_SHORT_NM']]\
            .apply(lambda df: get_is_major_helper(df), axis=1)

        # For all students in values of above dict, major GPA
        # Adjust for >>> '2021 Spring' > '2021 Fall' -> True
        if 'Spring' in reference:
            d_anon_gpa_major = df_grades[(df_grades['#SEMESTER_YEAR_NAME_CONCAT'] <= reference) & 
                      (df_grades['#SEMESTER_YEAR_NAME_CONCAT'] != reference[:4]+' Fall') & 
                      (df_grades['GRADE_TYPE_DESC'] == 'Letter Grade') &
                      (df_grades['IS_MAJOR'])]\
                .groupby('ANON_ID')\
                [['GRADE_POINTS_NBR', 'STUDENT_CREDIT_HRS_NBR']]\
                .apply(lambda grades: np.sum(grades['GRADE_POINTS_NBR'] * grades['STUDENT_CREDIT_HRS_NBR']) / np.sum(grades['STUDENT_CREDIT_HRS_NBR']))\
                .to_dict()
        else:
            d_anon_gpa_major = df_grades[(df_grades['#SEMESTER_YEAR_NAME_CONCAT'] <= reference[:4]+' Spring') & 
                      (df_grades['GRADE_TYPE_DESC'] == 'Letter Grade') &
                      (df_grades['IS_MAJOR'])]\
                .groupby('ANON_ID')\
                [['GRADE_POINTS_NBR', 'STUDENT_CREDIT_HRS_NBR']]\
                .apply(lambda grades: np.sum(grades['GRADE_POINTS_NBR'] * grades['STUDENT_CREDIT_HRS_NBR']) / np.sum(grades['STUDENT_CREDIT_HRS_NBR']))\
                .to_dict()

        # Mapping from course ID to number of enrollments for which a letter grade is available
        d_cid_n_letter_grades = df_grades[(df_grades['#SEMESTER_YEAR_NAME_CONCAT'] == reference) & 
                  (df_grades['GRADE_TYPE_DESC'] == 'Letter Grade')]\
            .groupby('COURSE_SUBJECT_NAME_NUMBER')\
            .size()\
            .to_dict()

        def cid_cleaning(abbr_cid):
            """
            Standardize CID for lookup in enrollment data.
            """
            return ' '.join(abbr_cid.split(' ')[:-1]) + '_' + abbr_cid.split(' ')[-1]

        n_satisfied_prereqs_current_semester = []
        n_satisfied_prereqs_all_past_semesters = []
        percent_satisfied_prereqs_current_semester = []
        percent_satisfied_prereqs_all_past_semesters = []
        student_gpa = []
        student_gpa_major = []
        is_stem_student = [] 
        is_stem_course = []
        course_student_stem_match = [] 
        is_non_letter_grade_course = []

        # Average across students enrolled in course
        for course in tqdm(COURSE_REFERENCE['course_name_number']):
            prereqs = d_cid_prereqs.get(cid_cleaning(course))
            students = d_cid_anonid.get(course)
            if students is None:
                n_satisfied_prereqs_current_semester.append(np.nan)
                n_satisfied_prereqs_all_past_semesters.append(np.nan)
                percent_satisfied_prereqs_current_semester.append(np.nan)
                percent_satisfied_prereqs_all_past_semesters.append(np.nan)
                student_gpa.append(np.nan)
                student_gpa_major.append(np.nan)
                is_stem_course.append(np.nan)
                is_stem_student.append(np.nan)
                course_student_stem_match.append(np.nan) 
                is_non_letter_grade_course.append(np.nan)
                continue
            # 1 Average satisfied prereqs across students
            if prereqs is None:
                n_satisfied_prereqs_current_semester.append(np.nan)
                n_satisfied_prereqs_all_past_semesters.append(np.nan)
                percent_satisfied_prereqs_current_semester.append(np.nan)
                percent_satisfied_prereqs_all_past_semesters.append(np.nan)
            else:
                prereqs = ast.literal_eval(prereqs)
                satis_prereqs_all = []
                satis_prereqs_sem = []
                for student in students:
                    courses_all = d_anonid_cid_all.get(student)
                    courses_sem = d_anonid_cid_sem.get(student)
                    satis_prereqs_all.append(len(set(courses_all) & set(prereqs)))
                    satis_prereqs_sem.append(len(set(courses_sem) & set(prereqs)))
                n_satisfied_prereqs_all_past_semesters.append(np.mean(satis_prereqs_all))
                n_satisfied_prereqs_current_semester.append(np.mean(satis_prereqs_sem))
                percent_satisfied_prereqs_all_past_semesters.append(np.mean(satis_prereqs_all)/len(prereqs))
                percent_satisfied_prereqs_current_semester.append(np.mean(satis_prereqs_sem)/len(prereqs))
            # 2 Average GPA across students
            student_gpas = [] 
            student_major_gpas = []
            for student in students:
                gpa = d_anon_gpa.get(student)
                if gpa is not None:
                    student_gpas.append(gpa)
                major_gpa = d_anon_gpa_major.get(student)
                if major_gpa is not None:
                    student_major_gpas.append(major_gpa)
            student_gpa.append(np.mean(student_gpas))
            student_gpa_major.append(np.mean(student_major_gpas))
            # 3 Stem Status Course, Student Major, Match
            course_stem_status = d_cid_stem.get(course)
            is_stem_course.append(course_stem_status)
            student_stem_statuses = []
            student_stem_matches = []
            for student in students:
                anon_dept = d_anonid_depts_sem.get(student)
                if anon_dept is not None:
                    stem_student_status = d_dept_stem.get(list(anon_dept)[0])
                    student_stem_statuses.append(stem_student_status)
                else:
                    student_stem_statuses.append(np.nan)
                if isinstance(stem_student_status, bool) and isinstance(course_stem_status, bool):
                    student_stem_matches.append(not (stem_student_status^course_stem_status))
                else:
                    student_stem_matches.append(np.nan)
            is_stem_student.append(np.mean(student_stem_statuses))
            course_student_stem_match.append(np.mean(student_stem_matches))
            # 4 Is non letter grade course (i.e., ['GRADE_TYPE_DESC'] != 'Letter Grade' for no enrollment in course)
            letter_grade_counts = d_cid_n_letter_grades.get(course)
            if letter_grade_counts is not None:
                is_non_letter_grade_course.append(letter_grade_counts == 0)
            else:
                is_non_letter_grade_course.append(np.nan)
                
        if verbose: print(f'Created aggregated individualized features for {reference}...')
    
    # Course-level enrollment-based variables
    COURSE_DATA = pd.read_csv('../LAK_paper_data/course_description_final.tsv', sep='\t', index_col=0)
    PREREQ_LOOKUP = COURSE_DATA[['cid', 'course_prereqs']].set_index('cid').to_dict()['course_prereqs']
    ABBR_CID2CID = COURSE_DATA[['abbr_cid', 'cid']].set_index('abbr_cid').to_dict()['cid']
    DESC_LOOKUP = COURSE_DATA[['cid', 'course_description']].set_index('cid').to_dict()['course_description']
    EDW_GRADES = pd.read_csv('../research-data/edw_student_grades_000', sep='\t')
    CREDIT_HOURS_LOOKUP = EDW_GRADES[['course_subject_name_number', 
                                      'student_credit_hrs_nbr']]\
                                    .set_index('course_subject_name_number').to_dict()['student_credit_hrs_nbr']
    EDW_MEAN_GRADES = EDW_GRADES\
        .groupby('course_subject_name_number')\
        [['grade_points_nbr']]\
        .mean()\
        .to_dict()['grade_points_nbr']

    EDW_SD_GRADES = EDW_GRADES\
        .groupby('course_subject_name_number')\
        [['grade_points_nbr']]\
        .std()\
        .to_dict()['grade_points_nbr']

    EDW_GRADES['is_non_letter_grade'] = EDW_GRADES['grade_type_desc'].map(lambda s: s == 'Non-Letter Grade')
    EDW_PERC_NON_LETTER_GRADES = EDW_GRADES\
        .groupby('course_subject_name_number')\
        [['is_non_letter_grade']]\
        .mean()\
        .to_dict()['is_non_letter_grade']

    EDW_GRADES['is_pass_or_satisfactory'] = EDW_GRADES['grade_nm'].map(lambda s: s in ['Pass', 'Satisfactory'])
    EDW_PASS_SATIS_AMONG_NON_LETTER = EDW_GRADES[EDW_GRADES['grade_type_desc'] == 'Non-Letter Grade']\
        .groupby('course_subject_name_number')\
        [['is_pass_or_satisfactory']]\
        .mean()\
        .to_dict()['is_pass_or_satisfactory']
    
    # get course vector for a given course ID (cid)
    def get_c2v(cid: str) -> list:
        try:
            idx = COURSE2VEC_idx[cid]-1 # minus 1 because the index starts from 1 rather than 0
        except KeyError:
            try:
                idx = COURSE2VEC_idx[cid[::-1].replace(' ', '_', 1)[::-1]]-1
            except KeyError:
                try:
                    idx = COURSE2VEC_idx[ABBR_CID2CID[cid]]-1
                except:
                    return np.nan
        vec = COURSE2VEC[idx]
        return vec
    
    # average prereq cid
    def get_avg_prereq_c2v(cid: str) -> list:
        try:
            prereqs = PREREQ_LOOKUP[cid]
        except KeyError:
            try:
                prereqs = PREREQ_LOOKUP[cid[::-1].replace(' ', '_', 1)[::-1]]
            except KeyError:
                return np.nan
        vs = [get_c2v(cid) for cid in ast.literal_eval(prereqs)]
        if len(vs) > 0:
            vs = np.mean(np.array(vs, dtype=object), axis=0)
        return vs
    
    def get_description_length(cid: str, use_words_not_chars=True) -> list:
        try:
            desc = DESC_LOOKUP[cid]
        except KeyError:
            try:
                desc = DESC_LOOKUP[cid[::-1].replace(' ', '_', 1)[::-1]]
            except KeyError:
                return np.nan
        if use_words_not_chars:
            try:
                desc = desc.split(' ')
            except AttributeError:
                return np.nan
        try:
            ans = len(desc)
        except TypeError:
            return np.nan
        return ans
    
    def get_division_category(cid: str) -> str:
        # Extract first one, two, or three consecutive digits
        pattern = re.compile(r'[0-9]{1,3}')
        try:
            ret = int(pattern.findall(cid)[0])
        except:
            return np.nan
        if ret < 100:
            return 'lower-division'
        elif ret < 200:
            return 'upper-division'
        else:
            return 'graduate'
    
    def get_n_credit_hours(cid: str) -> str:
        try:
            ch = CREDIT_HOURS_LOOKUP[cid]
        except KeyError:
            try:
                ch = CREDIT_HOURS_LOOKUP[cid[::-1].replace(' ', '_', 1)[::-1]]
            except KeyError:
                return np.nan
        return ch
    
    def get_n_prereqs(cid: str) -> list:
        try:
            prereqs = PREREQ_LOOKUP[cid]
        except KeyError:
            try:
                prereqs = PREREQ_LOOKUP[cid[::-1].replace(' ', '_', 1)[::-1]]
            except KeyError:
                return np.nan
        return len(prereqs)
    
    def get_grade_data(cid: str, ref_dict: dict) -> float:
        try:
            ans = ref_dict[cid]
        except KeyError:
            try:
                ans = ref_dict[cid[::-1].replace(' ', '_', 1)[::-1]]
            except KeyError:
                return np.nan
        return ans

    # Add
    COURSE_REFERENCE['c2v'] = COURSE_REFERENCE.course_name_number.map(get_c2v)

    c2v_cols = pd.DataFrame(COURSE_REFERENCE["c2v"].fillna("").apply(list).to_list(), 
                              columns=['c2v_'+str(i) for i in range(1, 301)])

    COURSE_REFERENCE = pd.concat([COURSE_REFERENCE, c2v_cols], axis=1).drop(columns=['c2v'])
    
    COURSE_REFERENCE['c2v'] = COURSE_REFERENCE.course_name_number.map(get_avg_prereq_c2v)

    c2v_cols = pd.DataFrame(COURSE_REFERENCE["c2v"].fillna("").apply(list).to_list(), 
                              columns=['c2v_prereq_avg_'+str(i) for i in range(1, 301)])

    COURSE_REFERENCE = pd.concat([COURSE_REFERENCE, c2v_cols], axis=1).drop(columns=['c2v'])
    
    COURSE_REFERENCE['catalog_n_words'] = COURSE_REFERENCE\
                                            .course_name_number\
                                            .map(lambda s: get_description_length(s, use_words_not_chars=True))
    COURSE_REFERENCE['catalog_n_chars'] = COURSE_REFERENCE\
                                            .course_name_number\
                                            .map(lambda s: get_description_length(s, use_words_not_chars=False))
    
    COURSE_REFERENCE['class_type'] = COURSE_REFERENCE.course_name_number.map(get_division_category)

    COURSE_REFERENCE['n_credit_hours'] = COURSE_REFERENCE.course_name_number.map(get_n_credit_hours)
    
    COURSE_REFERENCE['n_prereqs'] = COURSE_REFERENCE.course_name_number.map(get_n_prereqs)
    
    COURSE_REFERENCE['course_gpa_spring21_mean'] =\
        COURSE_REFERENCE.course_name_number\
            .map(lambda s: get_grade_data(s, ref_dict=EDW_MEAN_GRADES))

    COURSE_REFERENCE['course_gpa_spring21_sd'] =\
        COURSE_REFERENCE.course_name_number\
            .map(lambda s: get_grade_data(s, ref_dict=EDW_SD_GRADES))

    COURSE_REFERENCE['percentage_of_non_letter_grades'] =\
        COURSE_REFERENCE.course_name_number\
            .map(lambda s: get_grade_data(s, ref_dict=EDW_PERC_NON_LETTER_GRADES))

    COURSE_REFERENCE['percentage_of_pass_or_satisfactory_among_non_letter_grades'] =\
        COURSE_REFERENCE.course_name_number\
            .map(lambda s: get_grade_data(s, ref_dict=EDW_PASS_SATIS_AMONG_NON_LETTER))

    if verbose: print(f'Created enrollment-level course features for {reference}...')
        
    # All variables
    n_enrolled_students = []
    n_enrolled_teaching_staff = []
    dropout_ratio_q1 = []
    dropout_ratio_q2 = []
    dropout_ratio_q3 = []
    dropout_ratio_q4 = []
    submission_comments_avg_size_bytes = []
    submission_comments_per_student = []
    percent_submissions_submission_comments = []
    assignment_spread = []
    parallel_assingments_1day = []
    parallel_assingments_3day = []
    parallel_assingments_flexible = []
    n_course_assignments = []
    n_course_assignments_graded = []
    graded_assignments_week_average = []
    graded_assignments_week_max = []
    avg_submission_time_to_deadline_minutes = []
    early_assignment_availability_ratio = []
    avg_diff_available_due_assignments = []
    n_original_posts = []
    n_original_posts_dropout = []
    original_student_post_avg_size_bytes = []
    original_student_post_avg_size_bytes_dropout = []
    original_forum_posts_per_student = []
    original_forum_posts_per_student_dropout = []
    ta_teacher_posts_per_student = []
    ta_teacher_reply_time = []
    ta_teacher_reply_time_dropout = []
    forum_reply_ratio = []
    forum_reply_ratio_dropout = []
    n_course_assignments_deadline = []
    n_course_assignments_deadline_unlock = []
    n_submissions_students = [] 
    n_submissions_students_non_dropout = [] 
    n_submissions_students_dropout = [] 
    # Fill lists
    for index, row in tqdm(COURSE_REFERENCE.iterrows()):
        # Catch errors at secondary section evaluation
        try:
            secondary_sections = ast.literal_eval(row['secondary_section_number'])
        except:
            secondary_sections = []
        
        n_enrolled_students.append(
        get_n_enrollments(d['temp_dropout_reference'], 
                               row['course_name_number'], 
                               row['section_num'])
        )
        n_enrolled_teaching_staff.append(
        get_n_enrollments(d['temp_teaching_staff_reference'], 
                               row['course_name_number'], 
                               row['section_num'])
        )
        dropout_ratio_q1.append(
        get_dropped_out_ratio(d['temp_dropout_reference'], 
                               row['course_name_number'], 
                               row['section_num'], 
                               'dropped_out_q1')
        )
        dropout_ratio_q2.append(
        get_dropped_out_ratio(d['temp_dropout_reference'], 
                               row['course_name_number'], 
                               row['section_num'], 
                               'dropped_out_q2')
        )
        dropout_ratio_q3.append(
        get_dropped_out_ratio(d['temp_dropout_reference'], 
                               row['course_name_number'], 
                               row['section_num'], 
                               'dropped_out_q3')
        )
        dropout_ratio_q4.append(
        get_dropped_out_ratio(d['temp_dropout_reference'], 
                               row['course_name_number'], 
                               row['section_num'], 
                               'dropped_out_q4')
        )
        submission_comments_avg_size_bytes.append(
        get_submission_comments_avg_size_bytes(d['submission_comments'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   secondary_sections)
        )
        submission_comments_per_student.append(
        get_submission_comments_per_student(d, 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   secondary_sections)
        )
        percent_submissions_submission_comments.append(
        get_percent_submissions_submission_comments(d,
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   secondary_sections)
        )
        assignment_spread.append(
        get_assignment_spread(d['assignments_with_due'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        parallel_assingments_1day.append(
        get_parallel_assingments(d['assignments_with_due'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               grace_period_days=1)
        )
        parallel_assingments_3day.append(
        get_parallel_assingments(d['assignments_with_due'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               grace_period_days=3)
        )
        parallel_assingments_flexible.append(
        get_parallel_assingments_flexible(d['assignments_with_due'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        n_course_assignments.append(
        get_n_course_assignments(d['assignments'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               graded_only=False)
        )
        n_course_assignments_graded.append(
        get_n_course_assignments(d['assignments'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               graded_only=True)
        )
        graded_assignments_week_average.append(
        get_graded_assignments_week(d['assignments_with_due'],
                               week_start_dates, week_end_dates,
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               metric='average')
        )
        graded_assignments_week_max.append(
        get_graded_assignments_week(d['assignments_with_due'],
                               week_start_dates, week_end_dates,
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections,
                               metric='max')
        )
        avg_submission_time_to_deadline_minutes.append(
        get_avg_submission_time_to_deadline_minutes(d,
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        early_assignment_availability_ratio.append(
        get_early_assignment_availability_ratio(d['assignments_with_due-unlock'],
                               semester_start_plus_two_weeks,
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        avg_diff_available_due_assignments.append(
        get_avg_diff_available_due_assignments(d['assignments_with_due-unlock'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        n_original_posts.append(
        get_n_original_forum_posts(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        n_original_posts_dropout.append(
        get_n_original_forum_posts_dropout(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        original_student_post_avg_size_bytes.append(
        get_original_student_post_avg_size_bytes(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        original_student_post_avg_size_bytes_dropout.append(
        get_original_student_post_avg_size_bytes_dropout(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        original_forum_posts_per_student.append(
        get_original_forum_posts_per_student(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        original_forum_posts_per_student_dropout.append(
        get_original_forum_posts_per_student_dropout(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        ta_teacher_posts_per_student.append(
        get_ta_teacher_posts_per_student(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        ta_teacher_reply_time.append(
        get_ta_teacher_reply_time(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        ta_teacher_reply_time_dropout.append(
        get_ta_teacher_reply_time_dropout(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        forum_reply_ratio.append(
        get_reply_ratio(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        forum_reply_ratio_dropout.append(
        get_reply_ratio_dropout(d['discussion_entry'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )    
        n_course_assignments_deadline.append(
        get_n_course_assignments_deadline_unlock(d['assignments_with_due'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        n_course_assignments_deadline_unlock.append(
        get_n_course_assignments_deadline_unlock(d['assignments_with_due-unlock'],
                               row['course_name_number'], 
                               row['section_num'], 
                               secondary_sections)
        )
        n_submissions_students.append(
        get_n_submissions_students(d['submissions'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'])
        )
        n_submissions_students_non_dropout.append(
        get_n_submissions_students_by_dropout(d['submissions'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'],
                                   dropout_status=0)
        )
        n_submissions_students_dropout.append(
        get_n_submissions_students_by_dropout(d['submissions'], 
                                   row['course_name_number'], 
                                   row['section_num'], 
                                   row['secondary_section_number'],
                                   dropout_status=1)
    )
    # Add to course data set and create additional aggregates
    dat = COURSE_REFERENCE # Includes enrollment-level course features, e.g., C2V
    
    # Dropout Ratios
    dat['dropout_ratio_q1'] = dropout_ratio_q1
    dat['dropout_ratio_q2'] = dropout_ratio_q2
    dat['dropout_ratio_q3'] = dropout_ratio_q3
    dat['dropout_ratio_q4'] = dropout_ratio_q4

    # Enrollment stats
    dat['n_enrolled_students'] = n_enrolled_students
    dat['n_enrolled_teaching_staff'] = n_enrolled_teaching_staff
    dat['student_to_instructional_staff_ratio'] = np.array(n_enrolled_students) / np.array(n_enrolled_teaching_staff)

    # Assignments, fix typos in new var names
    dat['assignment_spread'] = assignment_spread
    dat['parallel_assignments_1day'] = parallel_assingments_1day
    dat['parallel_assignments_3day'] = parallel_assingments_3day
    dat['parallel_assignments_flexible'] = parallel_assingments_flexible
    dat['n_course_assignments'] = n_course_assignments
    dat['n_course_assignments_graded'] = n_course_assignments_graded
    dat['graded_assignments_week_average'] = graded_assignments_week_average
    dat['graded_assignments_week_max'] = graded_assignments_week_max
    dat['avg_submission_time_to_deadline_minutes'] = avg_submission_time_to_deadline_minutes
    dat['early_assignment_availability_ratio'] = early_assignment_availability_ratio
    dat['avg_diff_available_due_assignments'] = avg_diff_available_due_assignments

    # Submission Comments
    dat['submission_comments_avg_size_bytes'] = submission_comments_avg_size_bytes
    dat['submission_comments_per_student'] = submission_comments_per_student
    dat['percent_submissions_submission_comments'] = percent_submissions_submission_comments

    # Forum post quantity
    dat['n_original_posts'] = n_original_posts
    dat['n_original_posts_dropout'] = n_original_posts_dropout
    dat['original_student_post_avg_size_bytes'] = original_student_post_avg_size_bytes
    dat['original_student_post_avg_size_bytes_dropout'] = original_student_post_avg_size_bytes_dropout
    dat['original_forum_posts_per_student'] = original_forum_posts_per_student
    dat['original_forum_posts_per_student_dropout'] = original_forum_posts_per_student_dropout

    # Forum post responsivity
    dat['ta_teacher_posts_per_student'] = ta_teacher_posts_per_student
    dat['ta_teacher_reply_time'] = ta_teacher_reply_time
    dat['ta_teacher_reply_time_dropout'] = ta_teacher_reply_time_dropout
    dat['forum_reply_ratio'] = forum_reply_ratio
    dat['forum_reply_ratio_dropout'] = forum_reply_ratio_dropout

    dat['holds_secondary_sections'] = dat['secondary_section_number'] == '[]'
    dat.section_num = dat.section_num.astype(str)

    # Binary control variables
    dat['students_did_not_use_forum'] = dat['n_original_posts'] == 0
    dat['teachers_ta_did_not_use_forum'] = dat['ta_teacher_posts_per_student'] == 0
    dat['course_did_not_use_submission_comments'] = dat['submission_comments_per_student'] == 0
    dat['course_did_not_use_assignments'] = dat['n_course_assignments'] == 0

    dat['students_dropout_did_not_use_forum'] = dat['n_original_posts_dropout'] == 0
    dat['course_did_not_use_forum'] = dat['students_did_not_use_forum'] & dat['teachers_ta_did_not_use_forum'] & dat['students_dropout_did_not_use_forum']
    dat['course_did_not_use_assignments_with_deadlines'] = pd.Series(n_course_assignments_deadline) == 0
    dat['course_did_not_use_assignments_with_deadlines_unlock'] = pd.Series(n_course_assignments_deadline_unlock) == 0
    dat['course_did_not_use_submissions'] = pd.Series(n_submissions_students) == 0
    dat['course_did_not_use_submissions_non_dropout'] = pd.Series(n_submissions_students_non_dropout) == 0
    dat['course_did_not_use_submissions_dropout'] = pd.Series(n_submissions_students_dropout) == 0
    
    # Add, if needed, aggregate indiv variables
    if aggregate_indiv_variables: 
        dat['n_satisfied_prereqs_current_semester'] = n_satisfied_prereqs_current_semester
        dat['n_satisfied_prereqs_all_past_semesters'] = n_satisfied_prereqs_all_past_semesters
        dat['percent_satisfied_prereqs_current_semester'] = percent_satisfied_prereqs_current_semester
        dat['percent_satisfied_prereqs_all_past_semesters'] = percent_satisfied_prereqs_all_past_semesters
        dat['student_gpa'] = student_gpa
        dat['student_gpa_major'] = student_gpa_major
        dat['is_stem_student'] = is_stem_student 
        dat['is_stem_course'] = is_stem_course
        dat['course_student_stem_match'] = course_student_stem_match 
        dat['is_non_letter_grade_course'] = is_non_letter_grade_course
    
    if verbose: print(f'Done! Saving data for {reference} to {outf}...')
    dat.to_csv(outf, index=False)
    if verbose: print(f'Done!')
    
    return
    
### MACHINE LEARNING FUNCTIONS ###

LABELS = ['tl1', 'me', 'ps', 'cl_combined', 
          'tl_manage', 'me_manage', 'ps_manage', 'cl_combined_manage',
          'tl_agg', 'me_agg', 'ps_agg', 'cl_agg']

def control_variables_imputing(df):
    """
    The default 'imputation' strategy based on previous paper.
    Creates control variables and sets corresponding variables
    to 0 (e.g., course did not use canvas forum -> n_assignments=0).
    """
    
    # First, deal with missing C2V variables and fill in mean C2V vector
    df['has_no_prereq_c2v'] = df['c2v_prereq_avg_1'].map(lambda x: pd.isna(x))    
    fillna_dict = {'c2v_prereq_avg_'+str(i): AVG_C2V[i-1] for i in range(1, 300+1)}
    df.fillna(fillna_dict, inplace=True)
    df['has_no_course_c2v'] = df['c2v_1'].map(lambda x: pd.isna(x))    
    fillna_dict = {'c2v_'+str(i): AVG_C2V[i-1] for i in range(1, 300+1)}
    df.fillna(fillna_dict, inplace=True)
    
    # Catalog
    df['no_catalog_description'] = df['catalog_n_chars'].map(lambda x: pd.isna(x))
    df.fillna({'catalog_n_chars': 0, 'catalog_n_words': 0}, inplace=True)
    
    # Dropout
    df['no_dropout_data'] = df['dropout_ratio_q1'].map(lambda x: pd.isna(x))
    fillna_dict = {'dropout_ratio_q'+str(i): 0 for i in range(1, 4+1)}
    df.fillna(fillna_dict, inplace=True)
    
    # Prereq data
    df['no_prereq_data'] = df['n_prereqs'].map(lambda x: pd.isna(x))
    fillna_dict = {'n_prereqs': 0}
    df.fillna(fillna_dict, inplace=True)
    
    # Canvas data of teaching staff
    df['no_teaching_staff_lms_data'] = df['n_enrolled_teaching_staff'].map(lambda x: pd.isna(x))
    fillna_dict = {'n_enrolled_teaching_staff': 0}
    df.fillna(fillna_dict, inplace=True)
                                                
    # No credit hours data
    df['no_credit_hours_data'] = df['n_credit_hours'].map(lambda x: pd.isna(x))
    fillna_dict = {'n_credit_hours': 0}
    df.fillna(fillna_dict, inplace=True)
    
    # Grade data (course)
    df['no_course_grade_data'] = df['course_gpa_spring21_mean'].map(lambda x: pd.isna(x))
    for variable in ['course_gpa_spring21_mean', 'course_gpa_spring21_sd', 
                     'percentage_of_pass_or_satisfactory_among_non_letter_grades',
                     'percentage_of_non_letter_grades', 'is_non_letter_grade_course']:
        df.fillna({variable: 0}, inplace=True)
    
    # Grade data (student)
    df['no_student_grade_data'] = df['student_gpa_major'].map(lambda x: pd.isna(x))
    for variable in ['student_gpa', 'student_gpa_major']:
        df.fillna({variable: 0}, inplace=True)
        
    # Stem status (student)
    df['no_stem_status_student'] = df['is_stem_student'].map(lambda x: pd.isna(x))
    for variable in ['course_student_stem_match', 'is_stem_student']:
        df.fillna({variable: False}, inplace=True)
    
    # Stem status (course)
    df['no_stem_status_course'] = df['is_stem_course'].map(lambda x: pd.isna(x))
    df.fillna({'is_stem_course': 0}, inplace=True)
    
    # Prerequistites satisfied
    df['no_prereq_data'] = df['n_satisfied_prereqs_all_past_semesters'].map(lambda x: pd.isna(x))
    for variable in ['n_satisfied_prereqs_all_past_semesters', 'percent_satisfied_prereqs_2021_Spring',
                     'percent_satisfied_prereqs_all_past_semesters', 'percent_satisfied_prereqs_2021_Spring']:
        df.fillna({variable: 0}, inplace=True)

    # Pre-existing control variables
    pre_existing_controls = {
     'course_did_not_use_assignments',
     'course_did_not_use_assignments_with_deadlines',
     'course_did_not_use_assignments_with_deadlines_unlock',
     'course_did_not_use_forum',
     'course_did_not_use_submission_comments',
     'course_did_not_use_submissions',
     'teachers_ta_did_not_use_forum',
     'course_did_not_use_submissions_dropout',
     'course_did_not_use_submissions_non_dropout',
     'students_dropout_did_not_use_forum',
     'students_did_not_use_forum',
     'holds_secondary_sections'
    }
    if len(pre_existing_controls - set(df.columns)) > 0:
        warnings.warn("There are missing control variables in your data.")
        print(pre_existing_controls - set(df.columns))
        
    for variable in pre_existing_controls:
        df.fillna({variable: True}, inplace=True)
    
    # Fill LMS variables
    for variable in ['assignment_spread', 'avg_diff_available_due_assignments', 
                     'avg_submission_time_to_deadline_minutes', 
                     'ta_teacher_posts_per_student',
                     'ta_teacher_reply_time',
                     'ta_teacher_reply_time_dropout',
                     'submission_comments_avg_size_bytes',
                     'submission_comments_per_student',
                     'early_assignment_availability_ratio',
                     'forum_reply_ratio',
                     'forum_reply_ratio_dropout',
                     'graded_assignments_week_average',
                     'graded_assignments_week_max',
                     'student_to_instructional_staff_ratio',
                     'parallel_assignments_1day', 'parallel_assignments_3day',
                     'parallel_assignments_flexible', 
                     'percent_submissions_submission_comments',
                     'n_enrolled_students', 
                     'n_satisfied_prereqs_current_semester',
                     'percent_satisfied_prereqs_current_semester',
                     'original_forum_posts_per_student',
                     'original_student_post_avg_size_bytes_dropout',
                     'original_student_post_avg_size_bytes',
                     'n_original_posts_dropout',
                     'original_forum_posts_per_student_dropout',
                     'n_original_posts',
                     'n_course_assignments',
                     'n_course_assignments_graded']:
        df.fillna({variable: 0}, inplace=True)
    
    # Vars that have been renamed
    if 'n_satisfied_prereqs_2021_Spring' in df.columns:
        df.fillna({'n_satisfied_prereqs_2021_Spring': 0}, inplace=True)
    if 'percent_satisfied_prereqs_2021_Spring' in df.columns:
        df.fillna({'percent_satisfied_prereqs_2021_Spring': 0}, inplace=True)

    return df.dropna()

def knn_impute(df, k=2):
    imputer = KNNImputer(n_neighbors=k)
    res = pd.DataFrame(imputer.fit_transform(df.copy()), columns=df.columns)
    return res

def random_baseline(train, test, target='cl_combined'):
    
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target)
    
    preds = np.array([np.mean(y_train)]*y_train.shape[0])
    gold = y_train
    
    return preds, gold

def preproc(train, test, target='cl_combined', scale=True, imputing_strategy='knn', keras=False, is_xgb=False):
    # Drop labels that are not relevant
    other_labels = set(LABELS) - set([target])
    train.drop(columns=other_labels, inplace=True, errors='ignore')
    test.drop(columns=other_labels, inplace=True, errors='ignore')
    
    if imputing_strategy == 'knn':
        train = knn_impute(train.copy())
        test = knn_impute(test.copy())
    elif imputing_strategy == 'control variables':
        train = control_variables_imputing(train.copy())
        test = control_variables_imputing(test.copy())
    else:
        train.dropna(inplace=True)
        test.dropna(inplace=True)
    
    # Select variables
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    # Scaling predictors (z-scoring), apply separately to avoid leakage
    if scale:
        tmp = []
        for dataset in [X_train, X_test]:
            scaler = StandardScaler()
            scaler.fit(dataset)
            ans = scaler.transform(dataset)
            ans = pd.DataFrame(ans, columns=dataset.columns)
            tmp.append(ans)
        X_train, X_test = tmp[0].copy(), tmp[1].copy()
            
    if keras:
        tmp = []
        for dataset in [X_train, y_train, X_test, y_test]:
            ans = np.array(dataset).astype('float32')
            tmp.append(ans)
        X_train, y_train, X_test, y_test = tmp[0], tmp[1], tmp[2], tmp[3]
        
    if is_xgb:
        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        return data_dmatrix, X_train, y_train, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

def run_xgb(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):
    
    # Preproc
    data_dmatrix, X_train, y_train, X_test, y_test = preproc(train, test, target=target, 
                                                             is_xgb=True, imputing_strategy=imputing_strategy)

    # Define random search search space
    reg_alphas = [random.uniform(0.1, 10) for _ in range(n_random_searches)]
    gammas = [random.uniform(0.1, 10) for _ in range(n_random_searches)]
    learning_rates = [random.uniform(0, 1) for _ in range(n_random_searches)]
    
    best_mse = 10e5
    #for reg_alpha, gamma, learning_rate in tqdm(zip(reg_alphas, gammas, learning_rates)):
    for reg_alpha, gamma, learning_rate in zip(reg_alphas, gammas, learning_rates):  # no TQDM

        params = {
            "objective":"reg:squarederror",
            'reg_alpha': reg_alpha,
            'learning_rate': learning_rate, 
            'gamma': gamma
        }
        
        xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=50, early_stopping_rounds=None,  # Checked that train and test error do not change
                    as_pandas=True, seed=123)                        # substantially after around 50 boosts
        
        mse = xgb_cv.tail(1)['test-rmse-mean'].values[0]

        if mse < best_mse:
            # CV error
            best_mse = mse
            clf = xgb.XGBRegressor(objective="reg:squarederror",
                    reg_alpha=reg_alpha,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    eval_metric="rmse")
            best_clf = clone(clf)
            # CV Preds
            cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123)
            preds = cross_val_predict(best_clf, X_train, y_train, cv=cv, n_jobs = 1)
            gold = y_train
            # Train on full training data
            best_clf = clone(clf)
            best_clf.fit(X_train, y_train)
            
    return best_clf, best_mse, preds, gold

def run_linreg(train, test, target='cl_combined', top_k=2, imputing_strategy='knn'):
    # Preproc
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target, 
                                              imputing_strategy=imputing_strategy)
    
    # fit multiple polynomial features, did not converge for k > 2 during testing
    degrees = list(range(1, top_k+1)) # [1, 2, 3, 6, 10, 20]
    
    # full search
    best_mse = 10e100
    best_model = None
    preds = None
    gold = None
    for degree in degrees:
    
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123)
        scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))
        
        print(mse)

        if mse < best_mse:
            preds = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
            gold = y_train
            best_mse = mse
            best_degree = degree
            best_model = clone(model)
            # Train on full training data
            best_model.fit(X_train, y_train)
    
    return best_model, best_mse, preds, gold

def run_randomforest(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):
    # Preproc, keep all NAs because tree-based models can handle them
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target, 
                                               imputing_strategy=imputing_strategy)
    
    # Init, regularization parameters and training procedure alternations
    # n_estimators = 100
    # max_features{“sqrt”, “log2”, None}
    # ccp_alpha default=0.0 [0, 1] # pruning parameter for new optmization criterion
    # min_impurity_decrease [0, 0.25] # split if impurity decrase larger than value; impurity is the variance of y in regression
    n_trees = [int(random.uniform(50, 250)) for _ in range(n_random_searches)]
    max_features = [random.sample(['sqrt', 'log2', None], 1)[0] for _ in range(n_random_searches)]
    ccp_alphas = [random.uniform(0, 1) for _ in range(n_random_searches)]
    min_impurity_decreases = [random.uniform(0, 0.25) for _ in range(n_random_searches)]
    
    best_mse = 10e100
    best_model = None
    for n_tree, max_feat, ccp_alpha, min_purity_decrease in zip(n_trees, max_features, ccp_alphas, min_impurity_decreases):
        model = RandomForestRegressor(n_estimators=n_tree,
                                       max_features=max_feat,
                                       ccp_alpha=ccp_alpha,
                                       min_impurity_decrease=min_purity_decrease)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123)
        scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))

        if mse < best_mse:
            preds = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
            gold = y_train
            best_mse = mse
            best_model = clone(model)
            # Train on full training data
            best_model.fit(X_train, y_train)
    
    return best_model, best_mse, preds, gold

def run_neuralnet(train, test, n_random_searches=10, target='cl_combined', 
                  minmax=False, imputing_strategy='knn'):
    # Preproc, keep all NAs because tree-based models can handle them
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), 
                                                        target=target, 
                                                        keras=True, imputing_strategy=imputing_strategy)
    
    # MinMax for classification problem with sigmoid activation
    if minmax:
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(y_train.reshape(-1, 1))
        y_train = mm_scaler.transform(y_train.reshape(-1, 1))[:, 0]
        
    # Output activation function
    output_activation_function = 'relu' if not minmax else 'sigmoid'
    
    # Loss function
    loss_function = 'mean_squared_error' if not minmax else 'binary_crossentropy'
    
    # Hyperparameters
    # Paper: Delving Deeper into MOOC Student Dropout Prediction
    # 1-16 hidden layers
    # 1-10 hidden units per layer
    # Added: Dropout
    # Sigmoid instead of ReLU -> blows up activation in training -> NA prediction -> error
    n_hiddens = [int(random.uniform(1, 5)) for _ in range(n_random_searches)]
    n_units = [int(random.uniform(10, 100)) for _ in range(n_random_searches)]
    dropouts = [random.uniform(0, 0.95) for _ in range(n_random_searches)]
    
    best_mse = 10e100
    best_model = None
    best_n_hidden, best_n_unit, best_dropout = None, None, None
    # define neural network
    for n_hidden, n_unit, dropout in zip(n_hiddens, n_units, dropouts):        
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='sigmoid'))
            model.add(Dropout(dropout))
            # First hidden layer (minimum is 1)
            model.add(Dense(n_unit, input_dim=X_train.shape[1], activation='sigmoid'))
            model.add(Dropout(dropout))
            # Additional layers
            for _ in range(n_hidden-1):
                model.add(Dense(n_unit, input_dim=n_unit, activation='sigmoid'))
                model.add(Dropout(dropout))
            model.add(Dense(1, activation=output_activation_function))

            # Compile model
            model.compile(loss=loss_function, optimizer='adam')
            return model

        callback = EarlyStopping(monitor='loss', patience=4, min_delta=0)
        model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=1, 
                               verbose=0, callbacks=[callback])
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123)
        scores = cross_val_score(model, np.array(X_train.copy()).astype('float32'), 
                                 np.array(y_train.copy()).astype('float32'), 
                                 scoring='neg_mean_squared_error', cv=cv, n_jobs=1)        
        mse = np.absolute(np.mean(scores))
                
        if mse < best_mse:
            preds = cross_val_predict(model, np.array(X_train.copy()).astype('float32'), np.array(y_train.copy()).astype('float32'), cv=cv, method='predict')
            gold = y_train
            best_mse = mse
            best_model = clone(model)
            best_n_hidden, best_n_unit, best_dropout = n_hidden, n_unit, dropout
            # Train on full training data
            best_model.fit(X_train, y_train)
        
    # prepare best model saving because many Keras object can not be pickled as they are
    try:
        d_best_model = best_model.get_params()
        d_best_model['build_fn'] = inspect.getsource(d_best_model['build_fn'])
        d_best_model['n_hidden'] = best_n_hidden
        d_best_model['n_unit'] = best_n_unit
        d_best_model['dropout'] = best_dropout
        
    except:
        d_best_model, preds, gold = None, None, None
    
    return d_best_model, best_mse, preds, gold

def run_elasticnet(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):
    # Preproc
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target,
                                              imputing_strategy=imputing_strategy)

    # Define random search search space
    alphas = [10**(random.uniform(-6, 4)) for _ in range(n_random_searches)]
    l1_ratios = [random.uniform(0, 1) for _ in range(n_random_searches)] 

    best_mse = 10e100
    
    for alpha, l1_ratio in zip(alphas, l1_ratios): # no TQDM
    #for alpha, l1_ratio in tqdm(zip(alphas, l1_ratios)):

        # MSE error is hard-coded into ElasticNetCV
        eNet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.001)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123) # TODO Discuss

        scores = cross_val_score(eNet, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))
        
        if mse < best_mse:
            preds = cross_val_predict(eNet, X_train, y_train, cv=cv, method='predict')
            gold = y_train
            best_params = {
                'alpha': alpha,
                'l1_ratio': l1_ratio
            }
            best_mse = mse
            # Fit on full training data
            best_eNet = clone(eNet)
            best_eNet.fit(X_train, y_train)
            
    return best_eNet, best_mse, preds, gold

def run_lasso(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):
    # Preproc
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target,
                                              imputing_strategy=imputing_strategy)

    # Define random search search space
    alphas = [10**(random.uniform(-6, 4)) for _ in range(n_random_searches)]

    best_mse = 10e100    
    for alpha in alphas: 
        # MSE error is hard-coded into ElasticNetCV
        eNet = Lasso(alpha=alpha, max_iter=10000, tol=0.001)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123) # TODO Discuss

        scores = cross_val_score(eNet, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))
        
        if mse < best_mse:
            preds = cross_val_predict(eNet, X_train, y_train, cv=cv, method='predict')
            gold = y_train
            best_params = {
                'alpha': alpha
            }
            best_mse = mse
            # Fit on full training data
            best_eNet = clone(eNet)
            best_eNet.fit(X_train, y_train)
            
    return best_eNet, best_mse, preds, gold

def run_svm(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):    
    # Preproc
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target, 
                                              imputing_strategy=imputing_strategy)
    
    # Init random search
    
    Cs = [10**random.uniform(-1, 4) for _ in range(n_random_searches)]
    gammas = [10**random.uniform(-9, -1) for _ in range(n_random_searches)]
    epsilons = [10**random.uniform(-4, 0) for _ in range(n_random_searches)]
        
    best_mse = 10e100
    for C, gamma, epsilon in zip(Cs, gammas, epsilons):
        
        svm = SVR(kernel='rbf', gamma=1e-8, C=C, epsilon=epsilon)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123) 
        scores = cross_val_score(svm, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))
        
        if mse < best_mse:
            preds = cross_val_predict(svm, X_train, y_train, cv=cv, method='predict')
            gold = y_train

            best_params = {
                'C': C,
                'gamma': gamma,
                'epsilon': epsilon            
            }
            
            best_mse = mse
            svm = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
            # Train on full training data
            best_svm = clone(svm)
            best_svm.fit(X_train, y_train)
            
    return best_svm, best_mse, preds, gold

def run_svm(train, test, n_random_searches=10, target='cl_combined', imputing_strategy='knn'):    
    # Preproc
    X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target, 
                                              imputing_strategy=imputing_strategy)
    
    # Init random search
    
    Cs = [10**random.uniform(-1, 4) for _ in range(n_random_searches)]
    gammas = [10**random.uniform(-9, -1) for _ in range(n_random_searches)]
    epsilons = [10**random.uniform(-4, 0) for _ in range(n_random_searches)]
        
    best_mse = 10e100
    for C, gamma, epsilon in zip(Cs, gammas, epsilons):
        
        svm = SVR(kernel='rbf', gamma=1e-8, C=C, epsilon=epsilon)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=123) 
        scores = cross_val_score(svm, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
        mse = np.absolute(np.mean(scores))
        
        if mse < best_mse:
            preds = cross_val_predict(svm, X_train, y_train, cv=cv, method='predict')
            gold = y_train

            best_params = {
                'C': C,
                'gamma': gamma,
                'epsilon': epsilon            
            }
            
            best_mse = mse
            svm = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
            # Train on full training data
            best_svm = clone(svm)
            best_svm.fit(X_train, y_train)
            
    return best_svm, best_mse, preds, gold

def cap_score(x, upper_bound=5, lower_bound=1):
    if x < lower_bound:
        return lower_bound
    elif x > upper_bound:
        return upper_bound
    else:
        return x
    
def metric_score_capping(preds, upper_bound=5, lower_bound=1): 
    return np.array([cap_score(x, upper_bound=upper_bound, lower_bound=lower_bound) for x in preds])

def collapse_to_three(x, border1 = 5/3, border2 = 2*5/3):
    if x < border1:
        return 1
    elif x < border2:
        return 2
    else:
        return 3
    
def ternary_collapsing(preds, border1 = 5/3, border2 = 2*5/3): 
    return np.array([collapse_to_three(x, border1 = border1, border2 = border2) for x in preds])
    
def collapse_to_two(x, border1 = 5/2):
    if x < border1:
        return 1
    else:
        return 2
    
def binary_collapsing(preds, border1 = 5/2): 
    return np.array([collapse_to_two(x, border1 = border1) for x in preds])

def thresh2binary(x, border1 = 0.5):
    if x < border1:
        return 1
    else:
        return 2

def apply_threshold_to_binary(preds, border1 = 0.5):
    """
    For minmax normalized predictions
    """
    return np.array([thresh2binary(x, border1 = border1) for x in preds])

def thresh2trinary(x, border1 = 1/3, border2 = 2/3):
    if x < border1:
        return 1
    elif x < border2:
        return 2
    else:
        return 3

def apply_threshold_to_trinary(preds, border1 = 1/3, border2 = 2/3):
    """
    For minmax normalized predictions
    """
    return np.array([thresh2trinary(x, border1 = border1, border2=border2) for x in preds])

def my_r2(v1, v2): 
    return pearsonr(v1, v2)[0]**2    

def f1_macro(v1, v2): 
    return f1_score(v1, v2, average='macro')

def bootstrap_score(gold, preds, fun=mean_squared_error, n_iter=10000):
    n_obs = len(gold)
    true = fun(gold, preds)
    out = []
    for _ in range(n_iter):
        mask = np.random.randint(n_obs, size=n_obs)
        try:
            tmp = fun(gold[mask], preds[mask])
        except ValueError:
            continue
        out.append(tmp)
    return true, np.quantile(out, 0.025), np.quantile(out, 0.975)

def get_empirical_quantiles(train, target='cl_combined', minmax=False):
    res = dict()
    
    y_train = np.array(train[target])
    
    if minmax:
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(y_train.reshape(-1, 1))
        y_train = mm_scaler.transform(y_train.reshape(-1, 1))[:, 0]
    
    for i in range(101):
        res[i] = np.quantile(y_train, i/100)
    
    return res

def random_baseline_mcc_binary(train, test, target='cl_combined', border1=5/2):
    
    X_train, y_train, X_test, y_test = preproc(train, test, target=target)
    
    y_train_binary = binary_collapsing(y_train, border1=border1)
    
    preds = np.array([mode(binary_collapsing(train[target]))[0][0]]*y_train.shape[0])
    gold = binary_collapsing(y_train, border1=border1)
    
    return preds, gold

def random_baseline_mcc_trinary(train, test, target='cl_combined', border1 = 5/3, border2 = 2*5/3):
    
    X_train, y_train, X_test, y_test = preproc(train, test, target=target)
    
    y_train_binary = ternary_collapsing(y_train, border1=border1, border2=border2)
    
    preds = np.array([mode(binary_collapsing(train[target]))[0][0]]*y_train.shape[0])
    gold = binary_collapsing(y_train, border1=border1)
    
    return preds, gold

def run_model_training(train, test, target='cl_combined', n_searches=25, ignore_warnings=True, imputing_strategy='knn'):
    
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    
    emp_quant = get_empirical_quantiles(train.copy(), target=target)
    emp_quant_minmax = get_empirical_quantiles(train.copy(), target=target, minmax=True)
    
    dat = dict()
    models = dict()
    
    dat['random'], dat['gold'] = random_baseline(train.copy(), test.copy(), target=target)
    
    models['linreg'], _, dat['linreg'], _ = run_lasso(train.copy(), test.copy(), n_random_searches=n_searches, target=target, imputing_strategy=imputing_strategy)
    
    models['rf'], _, dat['rf'], _ = run_randomforest(train.copy(), test.copy(), n_random_searches=n_searches, target=target, imputing_strategy=imputing_strategy)

    models['nn'], _, dat['nn'], _ = run_neuralnet(train.copy(), test.copy(), n_random_searches=n_searches, target=target, imputing_strategy=imputing_strategy)
    
    models['nn-minmax'], _, dat['nn-minmax'], _ = run_neuralnet(train.copy(), test.copy(), n_random_searches=n_searches, target=target, minmax=True, imputing_strategy=imputing_strategy)
    
    models['xgb'], _, dat['xgb'], _  = run_xgb(train.copy(), test.copy(), target=target, n_random_searches=n_searches, imputing_strategy=imputing_strategy)
    
    models['enet'], _, dat['enet'], _ = run_elasticnet(train.copy(), test.copy(), target=target, n_random_searches=n_searches, imputing_strategy=imputing_strategy)
    
    models['svm'], _, dat['svm'], _  = run_svm(train.copy(), test.copy(), target=target, n_random_searches=n_searches, imputing_strategy=imputing_strategy)

    dat['ensemble'] = np.mean([dat['linreg'], dat['rf'], dat['nn'], dat['xgb'], dat['enet'], dat['svm']], axis=0)
        
    for method in ['random', 'gold', 'linreg', 'rf', 'nn', 'nn-minmax', 'xgb', 'enet', 'svm', 'ensemble']:
        try:
            dat[method+'_cap'] = metric_score_capping(dat[method], upper_bound=5, lower_bound=1)
        except:
            pass
        try:
            if method == 'random':
                dat[method+'_trinary'], _ = random_baseline_mcc_trinary(train.copy(), test.copy(), target=target, border1 = 5/3, border2 = 2*5/3)
                dat[method+'_binary'], _ = random_baseline_mcc_binary(train.copy(), test.copy(), target=target, border1=5/2)
                dat[method+'_trinary_emp'], _ = random_baseline_mcc_trinary(train.copy(), test.copy(), target=target, border1 = emp_quant[33], border2 = emp_quant[66])
                dat[method+'_binary_emp'], _ = random_baseline_mcc_binary(train.copy(), test.copy(), target=target, border1=emp_quant[50])
            elif method == 'nn-minmax':
                dat[method+'_trinary'] = apply_threshold_to_trinary(dat[method], border1 = 1/3, border2 = 2/3)
                dat[method+'_binary'] = apply_threshold_to_binary(dat[method], border1 = 0.5)
                dat[method+'_trinary_emp'] = apply_threshold_to_trinary(dat[method], border1 = emp_quant_minmax[33], border2 = emp_quant_minmax[66])
                dat[method+'_binary_emp'] = apply_threshold_to_binary(dat[method], border1 = emp_quant_minmax[50])
            else:
                dat[method+'_trinary'] = ternary_collapsing(dat[method], border1 = 5/3, border2 = 2*5/3)
                dat[method+'_binary'] = binary_collapsing(dat[method], border1 = 5/2)
                dat[method+'_trinary_emp'] = ternary_collapsing(dat[method], border1 = emp_quant[33], border2 = emp_quant[66])
                dat[method+'_binary_emp'] = binary_collapsing(dat[method], border1 = emp_quant[50])
        except:
            pass
        
    def metric_correspondence(s: str) -> list:
        if 'binary' in s:
            return [accuracy_score, roc_auc_score, f1_macro]
        elif 'trinary' in s:
            return [accuracy_score, f1_macro]
        else:
            return [mean_absolute_error]

    references, estimate, lower, upper = [], [], [], []
    for transform in ['', '_cap', '_trinary', '_binary', '_trinary_emp', '_binary_emp']:
        for eval_function in metric_correspondence(transform):
            for method in ['random', 'linreg', 'rf', 'nn', 'nn-minmax', 'xgb', 'enet', 'svm', 'ensemble']:
                try:
                    point, l, u = bootstrap_score(dat['gold'+transform], dat[method+transform], 
                                                  fun=eval_function, n_iter=1000)
                    references.append(method+transform+'_'+str(eval_function.__name__))
                    estimate.append(point)
                    lower.append(l)
                    upper.append(u)
                except:
                    pass
                    
    res = pd.DataFrame({
        'reference': references,
        'estimate': estimate,
        'lower_95conf': lower,
        'upper_95conf': upper
    })

    return res, models

#### MODEL EVALUATION ####

def get_sorted_model_result_table(prelim, label: str):
    d = prelim[label][0].copy()

    d['model'] = d['reference'].map(lambda s: s.split('_')[0])
    d['metric'] = d['reference'].map(lambda s: '_'.join(s.split('_')[1:]))
    d = d[['model', 'metric', 'estimate']]

    d = d.pivot(index='model',columns='metric')[['estimate']]
    d.columns = d.columns.droplevel(0)

    bv = [c for c in sorted(d.columns) if 'binary' in c]
    tv = [c for c in sorted(d.columns) if 'trinary' in c]

    d = d[['mean_absolute_error'] + bv + tv]
    
    d['target'] = label
    d.reset_index(inplace=True)
    d.set_index('target', inplace=True)
    d.reset_index(inplace=True)
    
    d = d.sort_values(by=['mean_absolute_error']) 
    
    # Difference to random
    baseline = d[d['model']=='random'].mean_absolute_error.values[0]
    d.insert(2, 'mean_absolute_error_diff', baseline - d['mean_absolute_error'] )
    
    return round(d, 3)

def cap_scores_test(xs, lower=1, upper=5):
    return [5 if x>5 else 1 if x<1 else x for x in xs]

#### EXTRAPOLATION ####

def apply_model(prelim, train, test, target='tl1', model_ref='linreg', imputing_strategy='control variables'):
    # Preproc
    if model_ref in ['random', 'linreg', 'rf', 'enet', 'svm']:
        X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), target=target, imputing_strategy=imputing_strategy)
    elif model_ref in ['xgb']:
        data_dmatrix, X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), 
                                                                 target=target, is_xgb=True, imputing_strategy=imputing_strategy)
    elif model_ref in ['nn']:
        X_train, y_train, X_test, y_test = preproc(train.copy(), test.copy(), 
                                                        target=target, keras=True, imputing_strategy=imputing_strategy)
    else:
        return None, None
    
    # Get model    
    if model_ref == 'nn':
        # Init build function
        temp = prelim[target][1][model_ref]
        n_unit = temp['n_unit']
        n_hidden = temp['n_hidden']
        dropout = temp['dropout']
        output_activation_function = 'relu'
        loss_function = 'mean_squared_error'
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='sigmoid'))
            model.add(Dropout(dropout))
            # First hidden layer (minimum is 1)
            model.add(Dense(n_unit, input_dim=X_train.shape[1], activation='sigmoid'))
            model.add(Dropout(dropout))
            # Additional layers
            for _ in range(n_hidden-1):
                model.add(Dense(n_unit, input_dim=n_unit, activation='sigmoid'))
                model.add(Dropout(dropout))
            model.add(Dense(1, activation=output_activation_function))

            # Compile model
            model.compile(loss=loss_function, optimizer='adam')
            return model
        callback = EarlyStopping(monitor='loss', patience=4, min_delta=0)
        model = KerasRegressor(build_fn=create_model, epochs=temp['epochs'], batch_size=temp['batch_size'], 
                               verbose=temp['verbose'], callbacks=[callback])
        model.fit(X_train, y_train)
        preds = cap_scores_test(model.predict(X_test))
        gold = y_test
        return preds, gold
    elif model_ref == 'random':
        preds = np.array([np.mean(y_train)]*y_test.shape[0])
        gold = y_test
        return preds, gold
    else:
        model = prelim[target][1][model_ref]
    
    # Get preds
    preds = cap_scores_test(model.predict(X_test[X_train.columns]))
    gold = y_test
    
    return preds, gold
