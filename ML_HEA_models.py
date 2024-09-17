'''
Module for storing ML models for HEA leagues. The data used must be clean, in a format to be received
by the models and separated between training and testing data.

In order to use this module:
    from ML_HEA_models import *
'''
# ----------------------------------------------------------------------------------------------------
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# ----------------------------------------------------------------------------------------------------
'''
Support functions - metrics and plots
'''
# R² adjusted
def calc_r2_adj(r2, y, X):
    return 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)

# Real vs Predicted plot
def plot_real_pred(y_test, y_pred):
    
    x = np.linspace(0, y_test.max())
    y = x

    plt.title("Real target vs Predicted target")
    plt.plot(x, y, color = "red", ls = ":")
    sns.scatterplot(x = y_test, y = y_pred)
    plt.xlabel("Real")
    plt.ylabel("Predict")
    plt.show()

# Metrics
def calc_metrics(model, X, y, label = "", plot = False, dist_resids=True, print_stuff=True):
    # model prediction
    y_pred = model.predict(X)

    if print_stuff:
        print(f"\nEvaluation metrics for {label}:\n")
    
    if plot:
        plot_real_pred(y, y_pred)
    
    #  Metrics
    r2 = r2_score(y, y_pred)
    r2_adj = calc_r2_adj(r2, y, X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # Printing error metrics 
    if print_stuff:
        print(f"R2: {r2:.2f}")
        print(f"R2 adjusted: {r2_adj:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    
    if dist_resids:
        residuos = y - y_pred
        print(f"\nResidual distribution of {label}:\n")
        print(residuos.describe())
    
    if plot:
        sns.histplot(residuos, kde = True)
        plt.show()
    
    metrics_dict = {"r2": r2,
                   "r2_adj": r2_adj,
                   "mae": mae,
                   "rmse": rmse}
    
    return metrics_dict

# Plot Linear graph
def plot_reglin_model(model, X_train, y_train):
    
    plt.title("Linear Regression Model")
    plt.scatter(X_train, y_train)
    x_plot_model = np.linspace(X_train.min(), X_train.max(),100_000)

    y_plot_model = model.intecept_
    for n, b_n in enumerate(model.coef_):
        y_plot_model = y_plot_model + b_n * (x_plot_model**(n+1))
    plt.plot(x_plot_model, y_plot_model, color = 'red')
    plt.show()
# ----------------------------------------------------------------------------------------------------
'''
Default model: Multiple Linear and Nonlinear Regression Model with LASSO Regularization
'''
def regression_poly_features_regularized(X_train, y_train, X_test, y_test,
                                         deg = 1,
                                         type_regularization = None,
                                         alpha = 1,
                                         iter_max = 1000,
                                         plot = True, scale_mms = False,
                                         train_metrics = True, # Trocar isso
                                         dist_resids = True,
                                         plot_model = False):
    '''
    Main arguments:
        - deg : polynomial degree
        - type_regularization : None, "l1" (Lasso), "l2" (Ridge)
        - alpha : penalty strength
        - plot : if True, shows plot
        - train_metrics : True to compare train data and False for test data
        - scale_mms = True for scaling - for most models
    '''
    
    # input data dimensions
    data_dim = X_train.shape[1]

    # saving original data
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()

    if deg > 1:
        # transform data to higher degrees
        pf = PolynomialFeatures(degree = deg, include_bias=False).fit(X_train)
        X_train = pf.transform(X_train)
        X_test = pf.transform(X_test)

        # Information
        print(f"\nFeature space transformed\n")
        print(f"\nOriginal features: {pf.n_features_in_}")
        print(f"\nTransformed features: {pf.n_output_features_}")
        print("="*50)

    if scale_mms or type_regularization:
        mms = MinMaxScaler().fit(X_train)
        X_train = mms.transform(X_train)
        X_test = mms.transform(X_test)
    
    # build model
    if type_regularization == "l1":
        model = Lasso(alpha = alpha, max_iter=iter_max).fit(X_train, y_train)
    elif type_regularization == "l2":
        model = Ridge(alpha = alpha, max_iter=iter_max).fit(X_train, y_train)
    elif type_regularization == None:
        model = LinearRegression().fit(X_train, y_train)
    # Caso não tenha
    else:
        reg_options = ["l1", "l2", None]
        raise ValueError(f"Choose one of the list {reg_options}")

    # Model evaluation
    if train_metrics:
        metrics_train = calc_metrics(model, X_train, y_train,
                                     label = "train", plot = plot,
                                     dist_resids = dist_resids,
                                     print_stuff = True)
        print()
        print("#"*50)
    else:
        metrics_train = None
    metrics_test = calc_metrics(model, X_train, y_train,
                                label = "test", plot = plot,
                                dist_resids = dist_resids,
                                print_stuff = True)
    if plot_model and data_dim == 1:
        plot_reglin_model(model, X_train_original, y_train, X_test_original, y_test)
    
    return model, metrics_train, metrics_test
# ----------------------------------------------------------------------------------------------------
'''
O modelo dentro da função é model. usar .coef_ e .feature_names_in_
Colocar dentro de um dataframe : df1 = pd.DataFrame(values = abs(model.coef_), index = model.feature_names_in_,
columns = ["Feature Importance"])
df1.sort_values(by="Feature importance", ascending=False) # organizar do maior para o menor.
'''