#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


semesters = ['2017 Spring', '2017 Fall', '2018 Spring', 
             '2018 Fall', '2019 Spring', '2019 Fall', '2020 Spring',
             '2020 Fall', '2021 Spring']


# In[ ]:


def main(reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_course_feature_engineering(reference=reference, 
                outf=f'../research-data/processed/course-features-{reference}.csv',
                semester_start=semester_frames[reference][0], semester_end=semester_frames[reference][1],
                aggregate_indiv_variables=True, verbose=True, debug=False)

if __name__ == '__main__':
    print(sys.argv[1])
    main(sys.argv[1])
