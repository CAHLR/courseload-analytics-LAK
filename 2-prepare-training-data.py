#!/usr/bin/env python
# coding: utf-8

# # Steps
# 
# 1) Preprocess the survey data export (which also includes individualized student features)
# 
# 2) Export an overview of all courses rated in the survey
# 
# 3) Join Spring 2021 LMS variables based on these courses

# In[6]:


import numpy as np
import pandas as pd

import ast
import itertools
import json
import os
import re
import sys
import warnings

from datetime import timedelta
from tqdm import tqdm
from utils import *


# ## Preprocess Survey Data

# In[7]:


# 1 - Select Cases and Features --------------------------------------------------------------------------

# Read in survey data, merge course-level and student-level data
path = '../research-data/s_courseload2_anon_record_filtered.tsv'
dat = pd.read_csv(path, sep="\t")
dat.columns = dat.columns.str.replace('#','n_')

path = '../research-data/s_courseload2_anon_student_filtered.tsv'
dat_student = pd.read_csv(path, sep="\t")
dat_student = dat_student.rename({'ts': 'ts_student'}, axis=1)

dat = dat.merge(dat_student, on='anon', how='left')

# Clean course name and section number
dat = dat.rename({'cid4grades': 'course_name_number', 'section_number': 'section_num'}, axis=1)
dat.section_num = dat.section_num.astype(str)
dat['section_num'] = dat['section_num'].apply(lambda x: x.zfill(3)) # add leading 0s until len of string is 3

# If a specific student rated the same course twice
# -> keep one instance (first entry), if the responses are the same
# -> omit all instances, if the responses differ

# All instances of student x course combinations that were rated more than once
dups = dat[dat.duplicated(subset=['anon', 'course_name_number', 'secondary_section_number'], keep=False)]

# Get indexes of responses that do not differ
dups_keep = dups[dups.duplicated(subset=['anon', 'course_name_number', 'secondary_section_number', 
                             'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'], keep=False)]

# Get unique responses out of responses that do not differ
dups_keep = dups_keep.drop_duplicates(subset=['anon', 'course_name_number', 'secondary_section_number', 
                             'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'], keep='first')

# Apply
indexes_to_omit = set(dups.index) - set(dups_keep.index)
dat = dat[~dat.index.isin(list(indexes_to_omit))]

# 2 - Data Cleaning ---------------------------------------------------------------------------------------

dat['tl1'] = dat.apply(answers_to_numeric, col='a1', axis=1)
dat['tl2'] = dat.apply(answers_to_numeric, col='a2', axis=1)
dat['tl_manage'] = dat.apply(answers_to_numeric, col='a3', axis=1)
dat['me'] = dat.apply(answers_to_numeric, col='a4', axis=1)
dat['me_manage'] = dat.apply(answers_to_numeric, col='a5', axis=1)
dat['ps'] = dat.apply(answers_to_numeric, col='a6', axis=1)
dat['ps_manage'] = dat.apply(answers_to_numeric, col='a7', axis=1)

dat['tl1_diff'] = dat['tl1'] - dat['tl_manage'] 
dat['tl2_diff'] = dat['tl2'] - dat['tl_manage']
dat['me_diff'] = dat['me'] - dat['me_manage']
dat['ps_diff'] = dat['ps'] - dat['ps_manage']

dat['tl_importance'] = dat.apply(answers_to_numeric, col='aTimeload', axis=1)
dat['me_importance'] = dat.apply(answers_to_numeric, col='aMental', axis=1)
dat['ps_importance'] = dat.apply(answers_to_numeric, col='aPsycho', axis=1)

d_stem_courses = pd.DataFrame.from_dict(d_stem_courses, orient='index')\
    .reset_index()\
    .rename({'index': 'course_name_number', 0: 'is_stem_course'}, axis=1)

dat = dat.merge(d_stem_courses, on='course_name_number', how='left')

d_stem_majors = pd.DataFrame.from_dict(d_stem_majors, orient='index')\
    .reset_index()\
    .rename({'index': 'major', 0: 'is_stem_student'}, axis=1)

dat = dat.merge(d_stem_majors, on='major', how='left')

# Collapse and standardize category 6 of tl1 to 5
dat['tl1'] = dat['tl1'].map(lambda x: 5 if x>5 else x)

# Create additional variables 
dat['holds_secondary_sections'] = dat['secondary_section_number'] == '[]'
dat['is_non_letter_grade_course'] = dat['avg_grade'].map(lambda x: x == 'Non Letter Grade')
dat['course_student_stem_match'] = dat['is_stem_course'] == dat['is_stem_student']
dat['cl_combined'] = dat[['tl1', 'me', 'ps']].mean(axis=1)
dat['cl_combined_manage'] = dat[['tl_manage', 'me_manage', 'ps_manage']].mean(axis=1)
dat['combined_importance'] = dat[['tl_importance', 'me_importance', 'ps_importance']].mean(axis=1)

# Create potentially more reliable estimate of scales
recode_scale = {
    1: 5,
    2: 4,
    3: 3,
    4: 2,
    5: 1
}
dat['tl_manage_rev'] = dat['tl_manage'].map(recode_scale)
dat['me_manage_rev'] = dat['me_manage'].map(recode_scale)
dat['ps_manage_rev'] = dat['ps_manage'].map(recode_scale)
dat['tl_agg'] =  dat[['tl1', 'tl_manage_rev']].mean(axis=1)
dat['me_agg'] =  dat[['me', 'me_manage_rev']].mean(axis=1)
dat['ps_agg'] =  dat[['ps', 'ps_manage_rev']].mean(axis=1)
dat['cl_agg'] =  dat[['tl1', 'tl_manage_rev',
                      'me', 'me_manage_rev',
                      'ps', 'ps_manage_rev']].mean(axis=1)

# Export disaggregated data with student vars for later references
dat.to_csv('../research-data/processed/lak22-courseload-surveydata.csv', index=False)

# Average course ratings and label smoothing
# Simple smoothing via student-level rating average
student_ref = dat.groupby('anon')[['tl1', 'tl2', 'me', 'ps', 'cl_combined']].mean().reset_index()
student_ref.columns = ['anon', 'tl1_student_avg', 'tl2_student_avg', 'me_student_avg', 
                       'ps_student_avg', 'cl_combined_student_avg']
dat = dat.merge(student_ref, on='anon', how='left')

# Smoothing via LMM random intercepts
random_intercepts = pd.read_csv('../research-data/processed/random-intercepts-for-smoothing.csv')
dat = dat.merge(random_intercepts, on='anon', how='left')

dat['percent_satisfied_prereqs_2021_Spring'] = dat['n_satisfied_prereqs_2021_Spring'] / dat['n_prereqs']
dat['percent_satisfied_prereqs_all_past_semesters'] = dat['n_satisfied_prereqs_all_past_semesters'] / dat['n_prereqs']
dat.rename(columns={'avg_gpa': 'student_gpa', 'avg_major_gpa': 'student_gpa_major'}, inplace=True)

# Average across multiple course ratings
dat = dat.groupby('course_name_number')[
    ['tl1', 'tl2', 'me', 'ps', 'cl_combined',
     'tl1_student_avg', 'tl2_student_avg', 'me_student_avg', 
     'ps_student_avg', 'cl_combined_student_avg',
     'tl_sensitivity', 'me_sensitivity', 'ps_sensitivity', 'cl_sensitivity',
     'is_stem_course', 'is_stem_student', 'course_student_stem_match',
     'n_satisfied_prereqs_2021_Spring', 'n_satisfied_prereqs_all_past_semesters',
    'percent_satisfied_prereqs_2021_Spring', 'percent_satisfied_prereqs_all_past_semesters',
    'is_non_letter_grade_course', 'student_gpa', 'student_gpa_major', 
    'tl_importance', 'me_importance', 'ps_importance', 'combined_importance',
    'tl_manage', 'me_manage', 'ps_manage', 'cl_combined_manage',
     'tl_agg', 'me_agg', 'ps_agg', 'cl_agg']
].agg(lambda x: np.nanmean(x)).reset_index()

dat['tl1_smoothed_lmm'] = dat['tl1'] + dat['tl_sensitivity']
dat['me_smoothed_lmm'] = dat['me'] + dat['me_sensitivity']
dat['ps_smoothed_lmm'] = dat['ps'] + dat['ps_sensitivity']
dat['cl_smoothed_lmm'] = dat['cl_combined'] + dat['cl_sensitivity']

dat['tl1_smoothed_student_average'] = dat['tl1'] - dat['tl1_student_avg']
dat['me_smoothed_student_average'] = dat['me'] - dat['me_student_avg']
dat['ps_smoothed_student_average'] = dat['ps'] - dat['ps_student_avg']
dat['cl_smoothed_student_average'] = dat['cl_combined'] - dat['cl_combined_student_avg']

dat = dat.drop(columns=['tl1_student_avg', 'tl2_student_avg', 'me_student_avg', 
     'ps_student_avg', 'cl_combined_student_avg'])

# Export labels
dat.to_csv('../research-data/processed/lak22-courseload-labels.csv', index=False)


# ## Export an overview of all courses rated in the survey

# In[18]:


# Course LMS and enrollment data for Spring 2021, keep indiv columns from survey data
df_21 = pd.read_csv('../research-data/processed/course-features-2021 Spring.csv')
df_21 = df_21[[c for c in df_21.columns if c not in dat.columns or c == 'course_name_number']]

df_final = dat.merge(df_21.drop_duplicates(subset=['course_name_number'], keep='first'), on='course_name_number', how='left')


# In[21]:


df_final.to_csv('../research-data/processed/lak22-courseload-final-studydata.csv', index=False)

