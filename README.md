# DATA1030-Radhika-Mehrotra
Stock procurement is often a challenge for businesses since it requires predicting future consumer demand. Demand patterns that are a result of interplay between a multitude of factors make the business problem more complex. Businesses seek to find patterns in consumption as they serve as useful indicators to solve such business problems.

I use Seoul’s bike sharing demand dataset to understand how weather conditions, and holidays affect bike rental demand.

# Packages Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split 
from  sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from  sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.model_selection import ParameterGrid
import shap
