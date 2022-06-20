import argparse


def build_parser():
    parser = argparse.ArgumentParser("Run Performance Function Experiments")
    parser.add_argument(
        "-l",
        "--lang",
        default="all",
        help="Which language to run experiments on. If `all` provided then the experiments will be run for all supported languages",
    )
    parser.add_argument(
        "-p",
        "--pivot_size",
        default="all",
        help="What pivot size to use to run experiments. Default = `all` meaning all supported pivot sizes will be used",
    )
    parser.add_argument(
        "--c12",
        type=float,
        default=0.1,
        help="Ratio between unit translation and unit manual data cost",
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs="+",
        default="fit_nd_eval",
        help="What mode to run the experiments. `fit_nd_eval` for fiting performance functions on performance data, `expansion_paths` for generating expansion paths. One or more of the options can be selected.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="performance_data/",
        help="Directory containing performance data",
    )
    parser.add_argument(
        "-f",
        "--performance_file",
        type=str,
        default="tydiqa_mbert_results.csv",
        help="csv file containing performance data",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to store outputs",
    )
    parser.add_argument("--test_split_frac", type=int, default=0.2)
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    return args
