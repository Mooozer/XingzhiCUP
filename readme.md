Environment: python 3.7.9

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


baidu_stopwords.txt can be found here: https://github.com/goto456/stopwords

![www-msq-biobert](https://github.com/Mooozer/www-msq-biobert/blob/main/MSQ.png)


If you train the model from scratch, run MacBERT_training_and_original_prediction.py, then run LIME_scores.py, and finally run model.py

If you get the results from the trained model with one click, run model.py directly

The final submission results are stored in the following directory . /lime_outputB/sim_rationale.txt  





