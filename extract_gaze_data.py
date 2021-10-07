import pandas as pd
from os import scandir, path
from xlsx2csv import Xlsx2csv
from utils import create_output_dir

INPUT_DIR = "data/"
OUTPUT_DIR = "output/gaze_data/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"


def process_participant_gaze(file, dataset=None):
    df = pd.read_csv(file,
                     sep="\t",
                     usecols=["Recording name", "Presented Stimulus name", "word_index", "word", "Gaze event duration"],
                     low_memory=False)

    df["Participant name"] = file.split("/")[-1][:-4]

    df = df[(df["word_index"] > -1)].groupby(
        ["Participant name", "Recording name", "Presented Stimulus name", "word_index", "word"]).sum()[
        'Gaze event duration'].reset_index()

    return df


def create_gaze_dataset(dataset=None, files=None):
    dfs = []
    df = None

    if dataset == SARCASM_DATASET:
        df = pd.read_csv("data/Mishra/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv")
        df = df[~(df["Word_ID"] == 1)]
        df["word_index"] = df["Word_ID"] - 2
        df = df.groupby(["Participant_ID", "Text_ID", "word_index", "Word"]).sum()['Fixation_Duration'].reset_index()

    if dataset == SOOD_DATASET:
        for file in files:
            try:
                df = process_participant_gaze(file, dataset=dataset)
                dfs.append(df)
                print(file)
            except Exception as e:
                print(f"{file}: {e}")

        df = pd.concat(dfs).reset_index()

    return df


def create_sood_et_al_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    output_file = f"{output_path}/study1_gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        data = "data/sood_et_al_2020/release24_2/study1_data/"
        files = [f"{data}{file.name}" for file in scandir(data) if ".tsv" in file.name]
        df = create_gaze_dataset(dataset=dataset, files=files)
        df.to_csv(output_file, index=False)
        print(f"{output_file} done")

    output_file = f"{output_path}/study2_gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        data = "data/sood_et_al_2020/release24_2/study2_data/"
        files = [f"{data}{file.name}" for file in scandir(data) if ".tsv" in file.name]
        df = create_gaze_dataset(dataset=dataset, files=files)
        df.to_csv(output_file, index=False)
        print(f"{output_file} done")


def create_mishra_sarcasm_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    output_file = f"{output_path}/gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        df = create_gaze_dataset(dataset)
        df.to_csv(output_file, index=False)
        print(f"{output_file} done")


def create_geco_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    output_file = f"{output_path}/MonolingualReadingData.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        Xlsx2csv("data/GECO/MonolingualReadingData.xlsx", outputencoding="utf-8").convert(output_file, sheetid=1)
        print(f"{output_file} done")

    output_file = f"{output_path}/L2ReadingData.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        Xlsx2csv("data/GECO/L2ReadingData.xlsx", outputencoding="utf-8").convert(output_file, sheetid=1)
        print(f"{output_file} done")


def main():
    if not path.isdir(path.join(INPUT_DIR, SOOD_DATASET)):
        print(f"Cannot find {SOOD_DATASET} - skipping creation")
    else:
        create_sood_et_al_gaze_data(SOOD_DATASET)

    if not path.isdir(path.join(INPUT_DIR, SARCASM_DATASET)):
        print(f"Cannot find {SARCASM_DATASET} - skipping creation")
    else:
        create_mishra_sarcasm_gaze_data(SARCASM_DATASET)

    if not path.isdir(path.join(INPUT_DIR, GECO_DATASET)):
        print(f"Cannot find {GECO_DATASET} - skipping creation")
    else:
        create_geco_gaze_data(GECO_DATASET)


if __name__ == "__main__":
    main()
