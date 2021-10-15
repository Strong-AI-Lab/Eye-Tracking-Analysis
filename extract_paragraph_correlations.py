import pandas as pd
from collections import OrderedDict
from scipy.stats import spearmanr
from os import path, scandir

from utils import create_output_dir

INPUT_DIR = "output/normalized_attention_data/paragraphs/"
OUTPUT_DIR = "output/correlation_data/paragraphs/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"
ZUCO_DATSET = "ZuCo"


def get_gaze_data(filepath):
    gaze_df = pd.read_csv(filepath)
    gaze_df = gaze_df.groupby("WORD_ID").mean()

    return gaze_df


def get_transformer_data(filepath):
    transformer_df = pd.read_csv(filepath).set_index("WORD_ID")

    return transformer_df


def merge_data(gaze_df, transformer_df):
    full_df = pd.merge(gaze_df, transformer_df, left_index=True, right_index=True)
    full_df = full_df[full_df["L1_attention-mean_word-mean_norm"] < 1]

    return full_df


def extract_correlations(df, eye_col, model):
    mean_dict = OrderedDict()
    max_dict = OrderedDict()

    mean_cols = [col for col in df.columns if
                 all(include in col for include in ["_norm", "attention", "mean_word-mean"]) and all(
                     exclude not in col for exclude in ["cls", "min", "sum"])]
    max_cols = [col for col in df.columns if
                all(include in col for include in ["_norm", "attention", "max_word-max"]) and all(
                    exclude not in col for exclude in ["cls", "min", "sum"])]

    for col in mean_cols:
        mean_dict[col] = spearmanr(df[col], df[eye_col], nan_policy="omit")

    for col in max_cols:
        max_dict[col] = spearmanr(df[col], df[eye_col], nan_policy="omit")

    mean_df = pd.DataFrame.from_dict(mean_dict)
    mean_df.index = [f"{model}", f"{model} p-val"]

    max_df = pd.DataFrame.from_dict(max_dict)
    max_df.index = [f"{model}", f"{model} p-val"]

    return mean_df, max_df


def create_datasets(gaze_data, transformer_files, eye_col, output_path):
    mean_dfs = []
    max_dfs = []

    gaze_df = get_gaze_data(gaze_data)

    for file in transformer_files:
        model = file.split("/")[-1][:-4]
        print(model)
        transformer_df = get_transformer_data(file)
        data_df = merge_data(gaze_df, transformer_df)
        mean_df, max_df = extract_correlations(data_df, eye_col, model)
        mean_dfs.append(mean_df)
        max_dfs.append(max_df)

    mean_df = pd.concat(mean_dfs)
    mean_data = mean_df.loc[["p-val" not in index for index in mean_df.index]]
    mean_p = mean_df.loc[["p-val" in index for index in mean_df.index]]

    max_df = pd.concat(max_dfs)
    max_data = max_df.loc[["p-val" not in index for index in max_df.index]]
    max_p = max_df.loc[["p-val" in index for index in max_df.index]]

    mean_data.to_csv(f"{output_path}/mean_df.csv")
    max_data.to_csv(f"{output_path}/max_df.csv")
    mean_p.to_csv(f"{output_path}/mean_p_val_df.csv")
    max_p.to_csv(f"{output_path}/max_p_val_df.csv")
    print("Done")


def create_sood_et_al_corr_table(dataset):
    print(dataset)
    eye_col = "Gaze event duration"
    transformer_datapath = f"{INPUT_DIR}{dataset}/"

    output_path = create_output_dir(f"{dataset}/Study_1/", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/sood_et_al_2020/normed_study1_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)
                         if "study_1" in file.name]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)


    output_path = create_output_dir(f"{dataset}/Study_2/", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/sood_et_al_2020/normed_study2_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)
                         if "study_2" in file.name]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)


def create_mishra_sarcasm_corr_table(dataset):
    print(dataset)
    eye_col = "Fixation_Duration"
    transformer_datapath = f"{INPUT_DIR}{dataset}/"

    output_path = create_output_dir(f"{dataset}", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/Mishra/Eye-tracking_and_SA-II_released_dataset/normed_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)


def create_geco_corr_table(dataset):
    print(dataset)
    eye_col = "WORD_GAZE_DURATION"
    transformer_datapath = f"{INPUT_DIR}{dataset}/"

    output_path = create_output_dir(f"{dataset}", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/GECO/normed_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)


def create_zuco_corr_table(dataset):
    print(dataset)
    eye_col = "Fixation_Duration"
    transformer_datapath = f"{INPUT_DIR}{dataset}/"

    output_path = create_output_dir(f"{dataset}/Study_1/", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/ZuCo/normed_task1_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)
                         if "task_1" in file.name]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)

    output_path = create_output_dir(f"{dataset}/Study_2/", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/ZuCo/normed_task2_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)
                         if "task_2" in file.name]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)

    output_path = create_output_dir(f"{dataset}/Study_3/", OUTPUT_DIR)
    gaze_data = "output/normalized_gaze_data/ZuCo/normed_task3_sentences.csv"
    transformer_files = [f"{transformer_datapath}{file.name}" for file in scandir(transformer_datapath)
                         if "task_3" in file.name]
    create_datasets(gaze_data, transformer_files, eye_col, output_path)


def main():
    if not path.isdir(path.join(INPUT_DIR, SOOD_DATASET)):
        print(f"Cannot find {SOOD_DATASET} - skipping creation")
    else:
        create_sood_et_al_corr_table(SOOD_DATASET)

    if not path.isdir(path.join(INPUT_DIR, SARCASM_DATASET)):
        print(f"Cannot find {SARCASM_DATASET} - skipping creation")
    else:
        create_mishra_sarcasm_corr_table(SARCASM_DATASET)

    if not path.isdir(path.join(INPUT_DIR, GECO_DATASET)):
        print(f"Cannot find {GECO_DATASET} - skipping creation")
    else:
        create_geco_corr_table(GECO_DATASET)

    if not path.isdir(path.join(INPUT_DIR, ZUCO_DATSET)):
        print(f"Cannot find {ZUCO_DATSET} - skipping creation")
    else:
        create_zuco_corr_table(ZUCO_DATSET)


if __name__ == "__main__":
    main()