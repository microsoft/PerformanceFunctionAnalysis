import os
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.args import build_parser
from src.fit_performance_functions import (
    fit_nd_eval_amue_model_diff_test,
    fit_nd_eval_amue_model,
    fit_nd_eval_gpr_model_diff_test,
    fit_nd_eval_gpr_model,
)
from src.perf_func_analysis import get_expansion_path


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s"
)
logger = logging.getLogger("PFA")
logger.setLevel(logging.INFO)


def fit_nd_eval_perf_func(perf_train_df, perf_test_df=None, perf_func="amue"):
    pred_error_dfs = []
    lang2ps2parms = {}

    for lang, lang_sub_train_df in perf_train_df.groupby("tgt_lang"):
        if lang == "en":
            continue
        lang2ps2parms[lang] = {}
        for pivot_size, lang_ps_sub_train_df in lang_sub_train_df.groupby(
            "en_pivot_size"
        ):
            print(
                f"Lang: {lang}, Pivot Size: {pivot_size}, Length of Training Data: {len(lang_ps_sub_train_df)}"
            )
            try:
                if perf_test_df is not None:
                    lang_ps_sub_test_df = perf_test_df[
                        (perf_test_df["tgt_lang"] == lang)
                        & (perf_test_df["en_pivot_size"] == pivot_size)
                    ]
                    if perf_func == "amue":
                        params, pred_nd_error_df = fit_nd_eval_amue_model_diff_test(
                            lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size
                        )
                    else:
                        params, pred_nd_error_df = fit_nd_eval_gpr_model_diff_test(
                            lang_ps_sub_train_df, lang_ps_sub_test_df, lang, pivot_size
                        )

                else:
                    if perf_func == "amue":
                        params, pred_nd_error_df = fit_nd_eval_amue_model(
                            lang_ps_sub_train_df, lang, pivot_size
                        )
                    else:
                        params, pred_nd_error_df = fit_nd_eval_gpr_model(
                            lang_ps_sub_train_df, lang, pivot_size
                        )
            except ValueError:
                continue
            pred_error_dfs.append(pred_nd_error_df)
            if isinstance(params, np.ndarray):
                lang2ps2parms[lang][pivot_size] = params.tolist()
            else:
                lang2ps2parms[lang][pivot_size] = params

    pred_error_df = pd.concat(pred_error_dfs, axis=0)
    mae, mse = (
        pred_error_df["Absolute Errors"].mean(),
        pred_error_df["Squared Errors"].mean() ** (1 / 2),
    )
    r2 = r2_score(
        pred_error_df["F1-Score"].values, pred_error_df["Predicted F1-Score"].values,
    )

    return (pred_error_df, lang2ps2parms, mae, mse, r2)


def main():

    args = build_parser()

    perf_filepath = os.path.join(args.data_dir, args.performance_file)
    logger.info(f"Loading Data from {perf_filepath}")
    perf_df = pd.read_csv(perf_filepath)

    logger.info("Creating Output Directories if not exists")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    langs = (
        [args.lang]
        if args.lang != "all"
        else sorted(list(perf_df["tgt_lang"].unique()))
    )
    pivot_sizes = (
        [int(args.pivot_size)]
        if args.pivot_size != "all"
        else sorted(list(perf_df["en_pivot_size"].unique()))
    )
    # pivot_sizes = [2000, 3696]

    perf_df = perf_df[
        (perf_df["tgt_lang"].isin(langs)) & (perf_df["en_pivot_size"].isin(pivot_sizes))
    ]

    if "fit_nd_eval" in args.mode:

        logger.info("Splitting Performance data into train and test data")
        perf_train_df, perf_test_df = train_test_split(
            perf_df, test_size=args.test_split_frac, random_state=args.seed
        )

        logger.info("Fitting And Evaluating AMUE Performance Function")
        (
            amue_pred_error_df,
            amue_lang2ps2parms,
            amue_mae,
            amue_mse,
            amue_r2,
        ) = fit_nd_eval_perf_func(
            perf_train_df, perf_test_df=perf_test_df, perf_func="amue"
        )
        logger.info(
            f"Done Fitting and Evaluating AMUE | MAE: {amue_mae} | RMSE: {amue_mse} | R^2: {amue_r2}"
        )

        logger.info("Saving Prediction and Errors")
        pred_dir = os.path.join(args.output_dir, "fit_results")
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        pred_file = os.path.join(
            pred_dir,
            f"amue_pred_nd_errors_lang{args.lang}_pivotSize{args.pivot_size}.csv",
        )
        amue_pred_error_df.to_csv(pred_file)

        logger.info("Saving Parameters")
        params_file = os.path.join(
            pred_dir, f"amue_params_lang{args.lang}_pivotSize{args.pivot_size}.json"
        )

        with open(params_file, "w") as f:
            json.dump(amue_lang2ps2parms, f, indent=4, ensure_ascii=False)

        (_, _, gpr_mae, gpr_mse, gpr_r2,) = fit_nd_eval_perf_func(
            perf_train_df, perf_test_df=perf_test_df, perf_func="gpr"
        )
        logger.info(
            f"Done Fitting and Evaluating GPR | MAE: {gpr_mae} | MSE: {gpr_mse} | R^2: {gpr_r2}"
        )

    if "expansion_paths" in args.mode:
        logger.info("Fitting AMUE Performance Function on entire dataset")
        (_, amue_lang2ps2parms, _, _, _,) = fit_nd_eval_perf_func(
            perf_df, perf_test_df=None, perf_func="amue"
        )
        for lang in langs:
            if lang == "en":
                continue
            for pivot_size in pivot_sizes:
                if pivot_size == 0:
                    continue
                params = amue_lang2ps2parms[lang][pivot_size]
                a0 = params[0]
                max_y = int(
                    perf_df[
                        (perf_df["tgt_lang"] == lang)
                        & (perf_df["en_pivot_size"] == pivot_size)
                    ]["f1_score"].max()
                )
                ys = np.linspace(np.ceil(a0), max_y, int((max_y - a0) // 1))
                get_expansion_path(
                    params,
                    ys,
                    args.c12,
                    lang,
                    pivot_size,
                    plot=True,
                    save_dir=os.path.join(args.output_dir, "exp_paths"),
                )


if __name__ == "__main__":
    main()
