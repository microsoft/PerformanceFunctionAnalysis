import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from src.amue_performance_function import AMUEPerfFunc


def fit_nd_eval_amue_model_diff_test(train_df, test_df, lang, pivot_size):
    """
    Fits the training performance data with AMUE Performance Function
    and evaluates the model on test data

    Inputs:
        - train_df (pd.DataFrame): Pandas Dataframe containing data configurations and corresponding performance values
            | en_tgt_trans_pivot_size | tgt_pivot_size | train_mode | f1_score |
            |-------------------------|----------------|------------|----------|

            where
                - `en_tgt_trans_pivot_size` column contains the amount of translated data
                - `tgt_pivot_size` column contains the amount of manual data
                - `train_mode` column contains which category the configuration falls into eg. zero-shot, few-shot etc.
                - `f1_score` column contains the f1_score obtained on the test data on training the model with a given data configuration
    
        - test_df (pd.DataFrame): Similar to `train_df` but to be used for evaluating the model
        - lang (str) : Language for which train_df and test_df were created
        - pivot_size (int) : Pivot size for which the configuration was created

    Returns
        - theta (np.ndarray): Parameters of the AMUE performance Function
        - pred_nd_error_df (pd.DataFrame) : Pandas dataframe containing the predictions and errors of the fitted model
    """

    X = train_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y = train_df["f1_score"].values
    modes = train_df["train_mode"].values
    amue_model = AMUEPerfFunc()
    amue_model.fit(X, y)
    theta = amue_model.theta

    X_test = test_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y_test = test_df["f1_score"].values
    y_pred = amue_model(X_test)
    errors = abs(y_pred - y_test)
    sq_errors = (y_pred - y_test) ** 2
    pred_nd_error_df = pd.DataFrame(
        {
            "Target Language": [lang for _ in range(len(X_test))],
            "English Pivot Size": [pivot_size for _ in range(len(X_test))],
            "Translation Size": X_test[:, 0],
            "Labelled Data Size": X_test[:, 1],
            "F1-Score": y_test,
            "Predicted F1-Score": y_pred,
            "Absolute Errors": errors,
            "Squared Errors": sq_errors,
        }
    )
    return theta, pred_nd_error_df


def fit_nd_eval_amue_model(perf_df, lang, pivot_size):
    """
    Fits the training performance data with AMUE Performance Function
    and evaluates the model on test data

    Inputs:
        - perf_df (pd.DataFrame): Pandas Dataframe containing data configurations and corresponding performance values
            | en_tgt_trans_pivot_size | tgt_pivot_size | train_mode | f1_score |
            |-------------------------|----------------|------------|----------|

            where
                - `en_tgt_trans_pivot_size` column contains the amount of translated data
                - `tgt_pivot_size` column contains the amount of manual data
                - `train_mode` column contains which category the configuration falls into eg. zero-shot, few-shot etc.
                - `f1_score` column contains the f1_score obtained on the test data on training the model with a given data configuration
    
        - lang (str) : Language for which train_df and test_df were created
        - pivot_size (int) : Pivot size for which the configuration was created

    Returns
        - theta (np.ndarray): Parameters of the AMUE performance Function
        - pred_nd_error_df (pd.DataFrame) : Pandas dataframe containing the predictions and errors of the fitted model
    """

    X = perf_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y = perf_df["f1_score"].values
    modes = perf_df["train_mode"].values
    amue_model = AMUEPerfFunc()
    amue_model.fit(X, y)
    theta = amue_model.theta
    y_pred = amue_model(X)
    errors = abs(y_pred - y)
    sq_errors = (y_pred - y) ** 2
    pred_nd_error_df = pd.DataFrame(
        {
            "Train Mode": modes,
            "Target Language": [lang for _ in range(len(X))],
            "English Pivot Size": [pivot_size for _ in range(len(X))],
            "Translation Size": X[:, 0],
            "Labelled Data Size": X[:, 1],
            "F1-Score": y,
            "Predicted F1-Score": y_pred,
            "Absolute Errors": errors,
            "Squared Errors": sq_errors,
        }
    )
    return theta, pred_nd_error_df


def fit_nd_eval_gpr_model_diff_test(train_df, test_df, lang, pivot_size):
    """
    Fits the training performance data with AMUE Performance Function
    and evaluates the model on test data

    Inputs:
        - train_df (pd.DataFrame): Pandas Dataframe containing data configurations and corresponding performance values
            | en_tgt_trans_pivot_size | tgt_pivot_size | train_mode | f1_score |
            |-------------------------|----------------|------------|----------|

            where
                - `en_tgt_trans_pivot_size` column contains the amount of translated data
                - `tgt_pivot_size` column contains the amount of manual data
                - `train_mode` column contains which category the configuration falls into eg. zero-shot, few-shot etc.
                - `f1_score` column contains the f1_score obtained on the test data on training the model with a given data configuration
    
        - test_df (pd.DataFrame): Similar to `train_df` but to be used for evaluating the model
        - lang (str) : Language for which train_df and test_df were created
        - pivot_size (int) : Pivot size for which the configuration was created

    Returns
        - gpr (sklearn.gaussian_process.GaussianProcessRegressor): Fitted GPR Model
        - pred_nd_error_df (pd.DataFrame) : Pandas dataframe containing the predictions and errors of the fitted model
    """
    X_train = train_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y_train = train_df["f1_score"].values
    modes = train_df["train_mode"].values

    kernel = RBF(length_scale=1, length_scale_bounds=(1e-05, 100000.0)) + WhiteKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42
    )
    gpr.fit(X_train, y_train)

    X_test = test_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y_test = test_df["f1_score"].values

    y_pred = gpr.predict(X_test)
    errors = abs(y_pred - y_test)
    sq_errors = (y_pred - y_test) ** 2
    pred_nd_error_df = pd.DataFrame(
        {
            "Target Language": [lang for _ in range(len(X_test))],
            "English Pivot Size": [pivot_size for _ in range(len(X_test))],
            "Translation Size": X_test[:, 0],
            "Labelled Data Size": X_test[:, 1],
            "F1-Score": y_test,
            "Predicted F1-Score": y_pred,
            "Absolute Errors": errors,
            "Squared Errors": sq_errors,
        }
    )
    return gpr, pred_nd_error_df


def fit_nd_eval_gpr_model(perf_df, lang, pivot_size):

    """
    Fits the training performance data with AMUE Performance Function
    and evaluates the model on test data

    Inputs:
        - perf_df (pd.DataFrame): Pandas Dataframe containing data configurations and corresponding performance values
            | en_tgt_trans_pivot_size | tgt_pivot_size | train_mode | f1_score |
            |-------------------------|----------------|------------|----------|

            where
                - `en_tgt_trans_pivot_size` column contains the amount of translated data
                - `tgt_pivot_size` column contains the amount of manual data
                - `train_mode` column contains which category the configuration falls into eg. zero-shot, few-shot etc.
                - `f1_score` column contains the f1_score obtained on the test data on training the model with a given data configuration
    
        - lang (str) : Language for which train_df and test_df were created
        - pivot_size (int) : Pivot size for which the configuration was created

    Returns
        - gpr (sklearn.gaussian_process.GaussianProcessRegressor): Fitted GPR Model
        - pred_nd_error_df (pd.DataFrame) : Pandas dataframe containing the predictions and errors of the fitted model
    """

    X = perf_df[["en_tgt_trans_pivot_size", "tgt_pivot_size"]].values
    y = perf_df["f1_score"].values
    modes = perf_df["train_mode"].values

    kernel = RBF(length_scale=1, length_scale_bounds=(1e-05, 100000.0)) + WhiteKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42
    )
    gpr.fit(X, y)

    y_pred = gpr.predict(X)
    errors = abs(y_pred - y)
    sq_errors = (y_pred - y) ** 2
    pred_nd_error_df = pd.DataFrame(
        {
            "Train Mode": modes,
            "Target Language": [lang for _ in range(len(X))],
            "English Pivot Size": [pivot_size for _ in range(len(X_test))],
            "Translation Size": X[:, 0],
            "Labelled Data Size": X[:, 1],
            "F1-Score": y,
            "Predicted F1-Score": y_pred,
            "Absolute Errors": errors,
            "Squared Errors": sq_errors,
        }
    )
    return gpr, pred_nd_error_df
