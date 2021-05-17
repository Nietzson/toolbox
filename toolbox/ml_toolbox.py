"""           Scaling Exploration                 """

# Define num_features and plot for each one a histplot and boxplot 
def scaling_exploration(X):
    
    for _ in list(X.columns):
        fig, ax=plt.subplots(1,2,figsize=(20,8))
        ax[0].set_title(f"Distribution of: {_}")
        sns.histplot(data= X[_],kde=True, ax = ax[0])
        ax[1].set_title(f"Boxplot of: {_}")
        sns.boxplot(data = X[_], ax=ax[1])
    fig.show() 

    
"""           Null Exploration                 """


def null_epxloration(data):
    
    import pandas as pd
    
    
    # NaN count for each column 
    data.isnull().sum().sort_values(ascending=False) 
    # DataFrame 
    pd.DataFrame(data.isnull().sum().sort_values(ascending=False), columns=["is_null"]) 
    # NaN percentage for each column data.isnull().sum().sort_values(ascending=False)/len(data) 
    # getting both nan count & percentage, in a DataFrame 
    return pd.DataFrame([ data.isnull().sum(),
                data.isnull().sum()/len(data) ], 
                index=["null_count", "null_share"]).T.sort_values(by="null_share", ascending=False)
    

"""                    Export model                     """

def export_model(tuned_pipe):

    import pickle
    # Export pipeline as pickle file

    with open("pipeline.pkl", "wb") as file:
        pickle.dump(tuned_pipe, file)


def load_model(tuned_pipe):

    import pickle
    # Load pipeline from pickle file
    my_pipeline = pickle.load(open("pipeline.pkl","rb"))

    return my_pipeline

"""                      Time series                        """


def  k_means_predictions(Xp):
    
    
    ### Predict the humber of clusters 
    inertias = []
    ks = range(1,10)
    for k in ks:
        km_test = KMeans(n_clusters=k).fit(Xp)
        inertias.append(km_test.inertia_)
    plt.plot(ks, inertias)
    plt.xlabel('k cluster number')





# We define here a "Plot forecast vs. real", which also shows historical train set

def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    # Prepare plot series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(train, label='training', color='black')
    plt.plot(test, label='actual', color='black', ls='--')
    plt.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8);
    
    
    
"""                  Check Residuals for inference validity                      """
    
def check_residuals(arima):
     
        
    residuals = pd.DataFrame(arima.resid)

    fig, ax = plt.subplots(1,2, figsize=(16,3))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1]);