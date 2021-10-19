import pandas as pd
from os import path

from utils import convert_plus_fill, create_output_dir

INPUT_DIR = "data/"
OUTPUT_DIR = "output/normalized_gaze_data/"
GAZE_DIR = "output/gaze_data/"
TEXT_DIR = "output/text_data/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"
ZUCO_DATSET = "ZuCo"
PROVO_DATASET = "Provo"
FRANK_DATASET = "Frank_et_al_2013"


def modify_gaze_df(gaze_df, dataset):
    if dataset == SOOD_DATASET:
        gaze_df["PARAGRAPH_ID"] = gaze_df["Presented Stimulus name"].apply(lambda s: s.split(".")[0])
        gaze_df["WORD_ID"] = gaze_df["PARAGRAPH_ID"] + "-" + gaze_df["word_index"].astype(str)

    if dataset in [SARCASM_DATASET, ZUCO_DATSET]:
        gaze_df["PARAGRAPH_ID"] = gaze_df["Text_ID"]
        gaze_df["WORD_ID"] = gaze_df["PARAGRAPH_ID"].astype(str) + "-" + gaze_df["word_index"].astype(str)

    if dataset == PROVO_DATASET:
        gaze_df["PARAGRAPH_ID"] = gaze_df["Text_ID"]
        gaze_df["WORD_ID"] = gaze_df["word_index"].astype(str)

    if dataset == FRANK_DATASET:
        gaze_df["PARAGRAPH_ID"] = 0
        gaze_df["WORD_ID"] = gaze_df["PARAGRAPH_ID"].astype(str) + "-" + gaze_df["Text_ID"].astype(str) + "-" + \
                             gaze_df["word_index"].astype(str)

    return gaze_df


def modify_words_df(words_df, dataset):
    words_df["PARAGRAPH_ID"] = words_df["PARAGRAPH_ID"].astype(str)

    if dataset == PROVO_DATASET:
        words_df["INDEX"] = words_df["WORD_ID"].apply(lambda s: "".join(filter(str.isdigit, s))).astype(int)
    elif dataset == FRANK_DATASET:
        words_df["INDEX"] = words_df["WORD_ID"].apply(lambda s: s.split(".")[-1]).astype(int)
    else:
        words_df["INDEX"] = words_df["WORD_ID"].apply(lambda s: s.split("-")[-1]).astype(int)

    return words_df


def merge_word_data(gaze_df, words_df, dataset):
    df = pd.merge(gaze_df, words_df, on="WORD_ID", how="outer")

    if dataset == SOOD_DATASET:
        df["Participant"] = df['Participant name'].unique()[0]
        df["Gaze event duration"] = df["Gaze event duration"].fillna(0)
        df = df.drop(["index", "Participant name", "Recording name", "Presented Stimulus name", "word_index", "word",
                      "PARAGRAPH_ID_x"], axis=1)

    if dataset in [SARCASM_DATASET, ZUCO_DATSET, PROVO_DATASET, FRANK_DATASET]:
        if dataset in [ZUCO_DATSET, PROVO_DATASET]:
            df["Participant"] = df['Participant_ID']
        else:
            df["Participant"] = df['Participant_ID'].unique()[0]
        df["Gaze event duration"] = df["Fixation_Duration"].fillna(0)
        df = df.drop(["Participant_ID", "word_index", "PARAGRAPH_ID_x"], axis=1)

    df = df.sort_values("INDEX")

    if dataset in [SARCASM_DATASET, SOOD_DATASET, FRANK_DATASET]:
        df = df.iloc[:words_df.shape[0]]

    return df


def combine_dfs(gaze_df, words_df, dataset):
    dfs = []
    participant_col = ""
    gaze_df = modify_gaze_df(gaze_df, dataset)
    words_df = modify_words_df(words_df, dataset)

    if dataset == SOOD_DATASET:
        participant_col = "Participant name"

    if dataset in [SARCASM_DATASET, FRANK_DATASET]:
        participant_col = "Participant_ID"

    for paragraph_id in words_df["PARAGRAPH_ID"].unique():
        print(paragraph_id)

        words_mask = words_df["PARAGRAPH_ID"] == paragraph_id
        current_words = words_df[words_mask]

        if dataset in [SARCASM_DATASET, FRANK_DATASET]:
            paragraph_id = int(paragraph_id)

        for name in gaze_df[participant_col].unique():
            gaze_mask = (gaze_df[participant_col] == name) & (gaze_df["PARAGRAPH_ID"] == paragraph_id)


            if gaze_mask.sum() == 0:
                continue

            current_gaze = gaze_df[gaze_mask]
            dfs.append(merge_word_data(current_gaze, current_words, dataset))

    df = pd.concat(dfs)

    return df


def normalize_gaze_data(df, dataset, paragraph=False):
    normalised_dfs = []
    normal_col = "SENTENCE_ID"

    if dataset == GECO_DATASET:
        participant_col = "PP_NR"

    if dataset in [SARCASM_DATASET, SOOD_DATASET, ZUCO_DATSET, PROVO_DATASET, FRANK_DATASET]:
        participant_col = "Participant"

    if paragraph:
        normal_col = "PARAGRAPH_ID"

        if dataset in [SARCASM_DATASET, SOOD_DATASET, ZUCO_DATSET, PROVO_DATASET, FRANK_DATASET]:
            normal_col = "PARAGRAPH_ID_y"

    for participant in df[participant_col].unique():
        print(participant)
        mask = df[participant_col] == participant
        person_df = df[mask]
        norm_data_id = df[mask][normal_col].unique()
        pp_ids = df[mask][normal_col]

        for data_id in norm_data_id:
            data_mask = pp_ids == data_id
            current_df = person_df[data_mask].set_index(["WORD_ID", participant_col]).select_dtypes(exclude="object")
            print(data_id)
            normalised_dfs.append(current_df / current_df.sum())

    df = pd.concat(normalised_dfs)

    return df


def create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file):
    sentence_df = normalize_gaze_data(df, dataset)
    sentence_df.to_csv(sentence_output_file)
    print(f"{sentence_output_file} done")

    paragraph_df = normalize_gaze_data(df, dataset, paragraph=True)
    paragraph_df.to_csv(paragraph_output_file)
    print(f"{paragraph_output_file} done")


def normalize_sood_et_al_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_study1_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_study1_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        study_df = pd.read_csv(f"{GAZE_DIR}{dataset}/study1_gaze_durations.csv")
        study_words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/study1_words.csv")
        df = combine_dfs(study_df, study_words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)

    sentence_output_file = f"{output_path}/normed_study2_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_study2_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        study_df = pd.read_csv(f"{GAZE_DIR}{dataset}/study2_gaze_durations.csv")
        study_words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/study2_words.csv")
        df = combine_dfs(study_df, study_words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def normalize_mishra_sarcasm_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        study_df = pd.read_csv(f"{GAZE_DIR}{dataset}/gaze_durations.csv")
        study_words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/words.csv")
        df = combine_dfs(study_df, study_words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def normalize_geco_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        mono_df = pd.read_csv(f"{GAZE_DIR}{dataset}/MonolingualReadingData.csv")
        bilingual_df = pd.read_csv(f"{GAZE_DIR}{dataset}/L2ReadingData.csv")
        df = pd.concat([bilingual_df, mono_df])
        keep_cols = [col for col in df.columns[11:] if
                     any(word in col for word in ["%", "COUNT", "DURATION", "AVERAGE", "SKIP"])]
        keep_cols = list(df.columns[:11]) + keep_cols
        df = df[keep_cols]

        for col in df.columns[11:]:
            df[col] = convert_plus_fill(df[col])

        df.loc[df["WORD"].isnull(), "WORD"] = "null"
        sentence_df = pd.read_csv(f"{TEXT_DIR}{dataset}/EnglishMaterialALL.csv",
                                  usecols=['WORD_ID', "SENTENCE_ID"])
        df = pd.merge(df, sentence_df, on="WORD_ID").sort_values(["PP_NR", "SENTENCE_ID"])
        df["PARAGRAPH_ID"] = df["WORD_ID"].apply(lambda s: "-".join(s.split("-")[:2]))
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def normalize_zuco_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_task1_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_task1_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        gaze_df = pd.read_csv(f"{GAZE_DIR}{dataset}/t1_gaze_duration.csv")
        gaze_df = modify_gaze_df(gaze_df, dataset)
        words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/t1_words.csv")
        words_df = modify_words_df(words_df, dataset)

        df = merge_word_data(gaze_df, words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)

    sentence_output_file = f"{output_path}/normed_task2_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_task2_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        gaze_df = pd.read_csv(f"{GAZE_DIR}{dataset}/t2_gaze_duration.csv")
        gaze_df = modify_gaze_df(gaze_df, dataset)
        words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/t2_words.csv")
        words_df = modify_words_df(words_df, dataset)

        df = merge_word_data(gaze_df, words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)

    sentence_output_file = f"{output_path}/normed_task3_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_task3_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        gaze_df = pd.read_csv(f"{GAZE_DIR}{dataset}/t3_gaze_duration.csv")
        gaze_df = modify_gaze_df(gaze_df, dataset)
        words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/t3_words.csv")
        words_df = modify_words_df(words_df, dataset)

        df = merge_word_data(gaze_df, words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def normalize_provo_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        gaze_df = pd.read_csv(f"{GAZE_DIR}{dataset}/gaze_durations.csv")
        gaze_df = modify_gaze_df(gaze_df, dataset)
        words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/words.csv")
        words_df = modify_words_df(words_df, dataset)

        df = merge_word_data(gaze_df, words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def normalize_frank_gaze_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    sentence_output_file = f"{output_path}/normed_sentences.csv"
    paragraph_output_file = f"{output_path}/normed_paragraphs.csv"

    if path.isfile(sentence_output_file) and path.isfile(paragraph_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        study_df = pd.read_csv(f"{GAZE_DIR}{dataset}/gaze_durations.csv")
        study_words_df = pd.read_csv(f"{TEXT_DIR}{dataset}/words.csv")
        df = combine_dfs(study_df, study_words_df, dataset)
        create_normalized_files(df, dataset, sentence_output_file, paragraph_output_file)


def method_chooser(dataset):
    if dataset == SOOD_DATASET:
        normalize_sood_et_al_gaze_data(dataset)
    elif dataset == SARCASM_DATASET:
        normalize_mishra_sarcasm_gaze_data(dataset)
    elif dataset == GECO_DATASET:
        normalize_geco_gaze_data(dataset)
    elif dataset == ZUCO_DATSET:
        normalize_zuco_gaze_data(dataset)
    elif dataset == PROVO_DATASET:
        normalize_provo_gaze_data(dataset)
    elif dataset == FRANK_DATASET:
        normalize_frank_gaze_data(dataset)


def main():
    for dataset in [SOOD_DATASET, SARCASM_DATASET, GECO_DATASET, ZUCO_DATSET, PROVO_DATASET, FRANK_DATASET]:
        if not path.isdir(path.join(INPUT_DIR, dataset)):
            print(f"Cannot find {dataset} - skipping creation")
        else:
            method_chooser(dataset)


if __name__ == "__main__":
    main()
