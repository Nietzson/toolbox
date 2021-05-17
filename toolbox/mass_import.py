def mass_import():
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge

    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split

    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer

    from sklearn.svm import SVC


    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder

    from sklearn.svm import SVC
    from sklearn.svm import SVR

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import make_scorer

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.linear_model import SGDRegressor, LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.pipeline import FeatureUnion

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.ensemble import StackingRegressor

    from sklearn.decomposition import PCA

    from sklearn.cluster import KMeans

    from sklearn import set_config; set_config(display='diagram')

    log_model = LogisticRegression(max_iter=1000)

    from tensorflow.keras import models
    from tensorflow.keras import Sequential, layers, regularizers

    from tensorflow.keras.callbacks import EarlyStopping