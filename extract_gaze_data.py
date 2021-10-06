import pandas as pd
from os import scandir, makedirs, path

INPUT_DIR = "data"
OUTPUT_DIR = "output/gaze_data/"


def create_output_dir(dataset):
    output_path = path.join(OUTPUT_DIR, dataset)
    if not path.isdir(output_path):
        makedirs(output_path)

    return output_path


def process_participant(file, dataset=None):
    df = pd.read_csv(file,
                     sep="\t",
                     usecols=["Recording name", "Presented Stimulus name", "word_index", "word", "Gaze event duration"],
                     low_memory=False)

    df["Participant name"] = file.split("/")[-1][:-4]

    df = df[(df["word_index"] > -1)].groupby(
        ["Participant name", "Recording name", "Presented Stimulus name", "word_index", "word"]).sum()[
        'Gaze event duration'].reset_index()
    return df


def create_gaze_dataset(files, dataset=None):
    dfs = []

    for file in files:
        try:
            df = process_participant(file, dataset=dataset)
            dfs.append(df)
            print(file)

        except Exception as e:
            print(f"{file}: {e}")

    return pd.concat(dfs).reset_index()


def create_sood_et_al_data(dataset):
    output_path = create_output_dir(dataset)
    output_file = f"{output_path}/study1_gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        data = "data/sood_et_al_2020/release24_2/study1_data/"
        files = [f"{data}{file.name}" for file in scandir(data) if ".tsv" in file.name]
        processed_df = create_gaze_dataset(files, dataset=dataset)
        processed_df.to_csv(output_file, index=False)
        print(f"{output_file} done")

    output_file = f"{output_path}/study2_gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")

    else:
        data = "data/sood_et_al_2020/release24_2/study2_data/"
        files = [f"{data}{file.name}" for file in scandir(data) if ".tsv" in file.name]
        processed_df = create_gaze_dataset(files, dataset=dataset)
        processed_df.to_csv(output_file, index=False)
        print(f"{output_file} done")


def create_mishra_sarcasm_data(dataset):
    output_path = create_output_dir(dataset)
    output_file = f"{output_path}/gaze_durations.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        df = pd.read_csv("data/Mishra/Eye-tracking_and_SA-II_released_dataset/Fixation_sequence.csv")
        df = df[~(df["Word_ID"] == 1)]
        df["word_index"] = df["Word_ID"] - 2
        df = df.groupby(["Participant_ID", "Text_ID", "word_index", "Word"]).sum()['Fixation_Duration'].reset_index()
        df.to_csv(output_file, index=False)
        print(f"{output_file} done")


def main():
    sood_dataset = "sood_et_al_2020"
    sarcasm_dataset = "Mishra/Eye-tracking_and_SA-II_released_dataset"

    if not path.isdir(path.join(INPUT_DIR, sood_dataset)):
        print(f"Cannot find {sood_dataset} - skipping creation")
    else:
        create_sood_et_al_data(sood_dataset)

    if not path.isdir(path.join(INPUT_DIR, sarcasm_dataset)):
        print(f"Cannot find {sarcasm_dataset} - skipping creation")
    else:
        create_mishra_sarcasm_data(sarcasm_dataset)


if __name__ == "__main__":
    main()
