# -*- coding:utf-8 -*-
import math
import os
import shutil
from collections import ChainMap, deque

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from gensim.models import Word2Vec
from joblib import Parallel, delayed

class Struc2Vec():
    def __init__(self, nx_graph):
        self.graph = nx_graph
