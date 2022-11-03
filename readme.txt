环境: python 3.7.9

import pandas as pd #(v-1.2.1)
import json #(v-2.0.9)
import re #(v-2.2.1)
import numpy as np #(v-1.19.2)
import torch #(v-1.11.0)
import collections
import os
import scipy #(v-1.5.2)
import matplotlib as plt #(v-3.5.1)
import zhon #(v-1.1.5)
import string
import LAC
import time
import datetime
import Levenshtein #(v-0.15.0)
import transformers #(v-4.5.0)
import random #(v-4.5.0)
import lime 
import math


baidu_stopwords.txt 文件是公开的百度停用词表,可从此处找到: https://github.com/goto456/stopwords


代码运行说明: 

如果从头训练模型，先后运行MacBERT_training_and_original_prediction.py 和 LIME_scores.py， 再运行model.py

如果从已训练的模型中一键获得结果，直接运行 model.py

最终提交结果存放在以下目录中  ./lime_outputB/sim_rationale.txt  





