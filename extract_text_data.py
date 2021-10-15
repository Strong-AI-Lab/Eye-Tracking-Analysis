import shutil

import pandas as pd
import glob
from collections import OrderedDict
from os import path
from xlsx2csv import Xlsx2csv
from utils import create_output_dir, get_text

INPUT_DIR = "data/"
OUTPUT_DIR = "output/text_data/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"
ZUCO_DATSET = "ZuCo"


def process_text(text_id, text):
    text_dict = OrderedDict()
    sentence_dict = OrderedDict()
    words = text.split()
    sentence = 0
    current_sentence = []

    for i, word in enumerate(words):
        text_dict[i] = {"PARAGRAPH_ID": text_id, "SENTENCE_ID": f"{text_id}-{sentence}", "WORD_ID": f"{text_id}-{i}",
                        "WORD": word}
        current_sentence.append(word)

        if any(end in word for end in [".", "?", "!"]):
            sentence_dict[sentence] = {"PARAGRAPH_ID": text_id, "SENTENCE_ID": f"{text_id}-{sentence}",
                                       "SENTENCE": " ".join(current_sentence)}
            current_sentence = []
            sentence = sentence + 1

    return text_dict, sentence_dict


def create_text_dfs(data, text=None):
    if text is None:
        text_id = data.split("/")[-1][:-4]
        text = get_text(data)
    else:
        text_id = data

    text_dict, sentence_dict = process_text(text_id, text)
    text_df = pd.DataFrame.from_dict(text_dict, orient="index")
    sentence_df = pd.DataFrame.from_dict(sentence_dict, orient="index")

    return text_df, sentence_df


def extract_format_text(dataset=None, texts=None):
    word_dfs = []
    sentence_dfs = []

    if dataset == SOOD_DATASET:
        for filepath in texts:
            text_df, sentence_df = create_text_dfs(filepath)
            word_dfs.append(text_df)
            sentence_dfs.append(sentence_df)

    if dataset == SARCASM_DATASET:
        raw_sentence_df = pd.read_csv(f"{INPUT_DIR}/{dataset}/text_and_annorations.csv",
                                      index_col=0)
        for text_id, text in raw_sentence_df["Text"].iteritems():
            text_df, sentence_df = create_text_dfs(text_id, text=text)
            word_dfs.append(text_df)
            sentence_dfs.append(sentence_df)

    words_df = pd.concat(word_dfs)
    sentences_df = pd.concat(sentence_dfs)

    return words_df, sentences_df


def create_sood_et_al_text_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    words_output_file = f"{output_path}/study1_words.csv"
    sentences_output_file = f"{output_path}/study1_sentences.csv"

    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} study 1 files already exist - skipping creation")
    else:
        texts = [text.replace("\\", "/") for text in
                 glob.glob(f'{INPUT_DIR}/{dataset}/release24_2/stimuli/study1/exp3/**/*.txt',
                           recursive=True) if "QA" not in text]
        words_df, sentences_df = extract_format_text(dataset=dataset, texts=texts)
        words_df.to_csv(words_output_file, index=False)
        sentences_df.to_csv(sentences_output_file, index=False)
        print(f"{output_path} study 1 files done")

    output_path = create_output_dir(dataset, OUTPUT_DIR)
    words_output_file = f"{output_path}/study2_words.csv"
    sentences_output_file = f"{output_path}/study2_sentences.csv"

    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} study 2 files already exist - skipping creation")
    else:
        texts = [text for text in glob.glob(f'{INPUT_DIR}/{dataset}/release24_2/stimuli/study2/**/*.txt',
                                            recursive=True)]
        words_df, sentences_df = extract_format_text(dataset=dataset, texts=texts)
        words_df.to_csv(words_output_file, index=False)
        sentences_df.to_csv(sentences_output_file, index=False)
        print(f"{output_path} study 2 files done")


def create_mishra_sarcasm_text_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    words_output_file = f"{output_path}/words.csv"
    sentences_output_file = f"{output_path}/sentences.csv"

    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} files already exist - skipping creation")
    else:
        words_df, sentences_df = extract_format_text(dataset=dataset)
        words_df.to_csv(words_output_file, index=False)
        sentences_df.to_csv(sentences_output_file, index=False)
        print(f"{output_path} files done")


def create_geco_text_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    output_file = f"{output_path}/EnglishMaterialALL.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        Xlsx2csv("data/GECO/EnglishMaterial.xlsx", outputencoding="utf-8").convert(f"{output_file[:-4]}_pre-clean.csv",
                                                                                   sheetid=1)

        # Clean the English data file
        original_df = pd.read_csv(f"{output_file[:-4]}_pre-clean.csv")
        original_df.loc[original_df['WORD_ID'] == "2-28-46", "WORD"] = "null"
        original_df.loc[original_df['WORD_ID'] == "1-41-67", "SENTENCE_ID"] = "1-281"
        original_df.loc[original_df['WORD_ID'] == "1-150-20", "SENTENCE_ID"] = "1-1238"
        part_one_error_start = 3170
        part_one_error_end = 14338
        original_df.loc[part_one_error_start + 1:part_one_error_end - 1, "SENTENCE_ID"] = original_df.loc[
                                                                                          part_one_error_start:part_one_error_end - 2,
                                                                                          "SENTENCE_ID"].values
        original_df.loc[original_df['WORD_ID'] == "1-41-60", "SENTENCE_ID"] = "1-280"
        original_df.loc[original_df['WORD_ID'] == "1-45-24", "SENTENCE_ID"] = "1-321"
        original_df.loc[original_df['WORD_ID'] == "1-46-30", "SENTENCE_ID"] = "1-328"
        original_df.loc[original_df['WORD_ID'] == "1-46-30", "SENTENCE_ID"] = "1-328"
        original_df.loc[original_df['WORD_ID'] == "1-46-96", "SENTENCE_ID"] = "1-333"
        original_df.loc[original_df['WORD_ID'] == "1-46-101", "SENTENCE_ID"] = "1-334"
        original_df.loc[original_df['WORD_ID'] == "1-47-52", "SENTENCE_ID"] = "1-339"
        original_df.loc[original_df['WORD_ID'] == "1-51-61", "SENTENCE_ID"] = "1-361"
        original_df.loc[original_df['WORD_ID'] == "1-52-27", "SENTENCE_ID"] = "1-363"
        original_df.loc[original_df['WORD_ID'] == "1-55-55", "SENTENCE_ID"] = "1-379"
        original_df.loc[original_df['WORD_ID'] == "1-55-72", "SENTENCE_ID"] = "1-381"
        original_df.loc[original_df['WORD_ID'] == "1-56-14", "SENTENCE_ID"] = "1-384"
        original_df.loc[original_df['WORD_ID'] == "1-57-1", "SENTENCE_ID"] = "1-386"
        original_df.loc[original_df['WORD_ID'] == "1-63-34", "SENTENCE_ID"] = "1-440"
        original_df.loc[original_df['WORD_ID'] == "1-63-41", "SENTENCE_ID"] = "1-442"
        original_df.loc[original_df['WORD_ID'] == "1-66-35", "SENTENCE_ID"] = "1-465"
        original_df.loc[original_df['WORD_ID'] == "1-67-58", "SENTENCE_ID"] = "1-478"
        original_df.loc[original_df['WORD_ID'] == "1-69-5", "SENTENCE_ID"] = "1-491"
        original_df.loc[original_df['WORD_ID'] == "1-71-17", "SENTENCE_ID"] = "1-510"
        original_df.loc[original_df['WORD_ID'] == "1-73-40", "SENTENCE_ID"] = "1-526"
        original_df.loc[original_df['WORD_ID'] == "1-73-83", "SENTENCE_ID"] = "1-532"
        original_df.loc[original_df['WORD_ID'] == "1-74-76", "SENTENCE_ID"] = "1-543"
        original_df.loc[original_df['WORD_ID'] == "1-75-47", "SENTENCE_ID"] = "1-547"
        original_df.loc[original_df['WORD_ID'] == "1-78-14", "SENTENCE_ID"] = "1-563"
        original_df.loc[original_df['WORD_ID'] == "1-78-46", "SENTENCE_ID"] = "1-566"
        original_df.loc[original_df['WORD_ID'] == "1-92-43", "SENTENCE_ID"] = "1-657"
        original_df.loc[original_df['WORD_ID'] == "1-92-64", "SENTENCE_ID"] = "1-660"
        original_df.loc[original_df['WORD_ID'] == "1-92-93", "SENTENCE_ID"] = "1-665"
        original_df.loc[original_df['WORD_ID'] == "1-93-1", "SENTENCE_ID"] = "1-668"
        original_df.loc[original_df['WORD_ID'] == "1-93-31", "SENTENCE_ID"] = "1-671"
        original_df.loc[original_df['WORD_ID'] == "1-93-65", "SENTENCE_ID"] = "1-675"
        original_df.loc[original_df['WORD_ID'] == "1-94-49", "SENTENCE_ID"] = "1-685"
        original_df.loc[original_df['WORD_ID'] == "1-95-81", "SENTENCE_ID"] = "1-706"
        original_df.loc[original_df['WORD_ID'] == "1-96-1", "SENTENCE_ID"] = "1-708"
        original_df.loc[original_df['WORD_ID'] == "1-96-17", "SENTENCE_ID"] = "1-712"
        original_df.loc[original_df['WORD_ID'] == "1-97-33", "SENTENCE_ID"] = "1-724"
        original_df.loc[original_df['WORD_ID'] == "1-99-101", "SENTENCE_ID"] = "1-745"
        original_df.loc[original_df['WORD_ID'] == "1-100-14", "SENTENCE_ID"] = "1-750"
        original_df.loc[original_df['WORD_ID'] == "1-100-44", "SENTENCE_ID"] = "1-754"
        original_df.loc[original_df['WORD_ID'] == "1-103-69", "SENTENCE_ID"] = "1-779"
        original_df.loc[original_df['WORD_ID'] == "1-108-44", "SENTENCE_ID"] = "1-824"
        original_df.loc[original_df['WORD_ID'] == "1-108-60", "SENTENCE_ID"] = "1-827"
        original_df.loc[original_df['WORD_ID'] == "1-108-68", "SENTENCE_ID"] = "1-830"
        original_df.loc[original_df['WORD_ID'] == "1-111-49", "SENTENCE_ID"] = "1-864"
        original_df.loc[original_df['WORD_ID'] == "1-112-56", "SENTENCE_ID"] = "1-875"
        original_df.loc[original_df['WORD_ID'] == "1-118-40", "SENTENCE_ID"] = "1-928"
        original_df.loc[original_df['WORD_ID'] == "1-119-68", "SENTENCE_ID"] = "1-939"
        original_df.loc[original_df['WORD_ID'] == "1-120-82", "SENTENCE_ID"] = "1-949"
        original_df.loc[original_df['WORD_ID'] == "1-123-69", "SENTENCE_ID"] = "1-971"
        original_df.loc[original_df['WORD_ID'] == "1-125-68", "SENTENCE_ID"] = "1-984"
        original_df.loc[original_df['WORD_ID'] == "1-128-44", "SENTENCE_ID"] = "1-1007"
        original_df.loc[original_df['WORD_ID'] == "1-130-1", "SENTENCE_ID"] = "1-1020"
        original_df.loc[original_df['WORD_ID'] == "1-150-33", "SENTENCE_ID"] = "1-1240"
        original_df.loc[original_df['WORD_ID'] == "1-150-98", "SENTENCE_ID"] = "1-1248"
        original_df.loc[original_df['WORD_ID'] == "1-151-49", "SENTENCE_ID"] = "1-1252"
        original_df.loc[original_df['WORD_ID'] == "1-152-40", "SENTENCE_ID"] = "1-1258"
        original_df.loc[original_df['WORD_ID'] == "1-152-49", "SENTENCE_ID"] = "1-1260"
        original_df.loc[original_df['WORD_ID'] == "1-155-4", "SENTENCE_ID"] = "1-1291"
        original_df.loc[original_df['WORD_ID'] == "1-155-65", "SENTENCE_ID"] = "1-1296"
        original_df.loc[original_df['WORD_ID'] == "1-158-54", "SENTENCE_ID"] = "1-1325"
        original_df.loc[original_df['WORD_ID'] == "1-160-36", "SENTENCE_ID"] = "1-1352"
        original_df.loc[original_df['WORD_ID'] == "1-161-97", "SENTENCE_ID"] = "1-1364"
        original_df.loc[original_df['WORD_ID'] == "1-162-75", "SENTENCE_ID"] = "1-1373"
        original_df.loc[original_df['WORD_ID'] == "1-163-32", "SENTENCE_ID"] = "1-1374"
        original_df.loc[original_df['WORD_ID'] == "1-163-33", "SENTENCE_ID"] = "1-1374"
        original_df.loc[original_df['WORD_ID'] == "1-163-34", "SENTENCE_ID"] = "1-1374"
        original_df.loc[original_df['WORD_ID'] == "1-163-35", "SENTENCE_ID"] = "1-1374"
        original_df.loc[original_df['WORD_ID'] == "1-128-44", "SENTENCE_ID"] = "1-1007"
        original_df.loc[original_df['WORD_ID'] == "3-101-18", "SENTENCE_ID"] = "3-976"
        original_df.loc[original_df['WORD_ID'] == "3-101-28", "SENTENCE_ID"] = "3-977"
        original_df.loc[original_df['WORD_ID'] == "3-101-35", "SENTENCE_ID"] = "3-978"
        original_df.loc[original_df['WORD_ID'] == "3-101-36", "SENTENCE_ID"] = "3-978"
        original_df.loc[original_df['WORD_ID'] == "3-101-47", "SENTENCE_ID"] = "3-979"
        original_df.loc[original_df['WORD_ID'] == "3-101-48", "SENTENCE_ID"] = "3-979"
        original_df.loc[original_df['WORD_ID'] == "3-101-60", "SENTENCE_ID"] = "3-980"
        original_df.loc[original_df['WORD_ID'] == "3-101-61", "SENTENCE_ID"] = "3-980"
        original_df.loc[original_df['WORD_ID'] == "3-101-62", "SENTENCE_ID"] = "3-980"
        original_df.loc[original_df['WORD_ID'] == "3-101-72", "SENTENCE_ID"] = "3-981"
        original_df.loc[original_df['WORD_ID'] == "3-101-72", "SENTENCE_ID"] = "3-981"
        original_df.loc[original_df['WORD_ID'] == "3-101-72", "SENTENCE_ID"] = "3-981"
        original_df.loc[original_df['WORD_ID'] == "3-101-89", "SENTENCE_ID"] = "3-982"
        original_df.loc[original_df['WORD_ID'] == "3-101-90", "SENTENCE_ID"] = "3-982"
        original_df.loc[original_df['WORD_ID'] == "3-101-91", "SENTENCE_ID"] = "3-982"
        original_df.loc[original_df['WORD_ID'] == "3-101-92", "SENTENCE_ID"] = "3-982"
        original_df.loc[original_df['WORD_ID'] == "3-101-73", "SENTENCE_ID"] = "3-981"
        original_df.loc[original_df['WORD_ID'] == "3-101-74", "SENTENCE_ID"] = "3-981"
        original_df.loc[original_df['WORD_ID'] == "3-20-35", "WORD"] = "tack"
        original_df.loc[original_df['WORD_ID'] == "4-86-98", "WORD"] = "Poison"
        original_df = original_df.drop(
            [14273, 14274, 14275, 14276, 29440, 35158, 35159, 35160, 35161, 35162, 35163, 35164, 35165,
             35166, 35167, 35168, 35169, 35170, 35171, 35172, 40982, 54360, 54361])

        original_df.to_csv(output_file, index=False)
        print(f"{output_file} done")

    output_file = f"{output_path}/EnglishMaterialSENTENCE.csv"

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        Xlsx2csv("data/GECO/EnglishMaterial.xlsx", outputencoding="utf-8").convert(output_file, sheetid=3)
        print(f"{output_file} done")


def create_zuco_text_data(dataset):
    output_path = create_output_dir(dataset, OUTPUT_DIR)
    words_output_file = f"{output_path}/t1_words.csv"
    sentences_output_file = f"{output_path}/t1_sentences.csv"

    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} task 1 files already exist - skipping creation")
    else:
        words_df = pd.read_csv(f"{INPUT_DIR}{dataset}/Task_1/words.csv")
        words_df["WORD_ID"] = words_df["SENTENCE_ID"].astype(str) + "-" + words_df["WORD_ID"].astype(str)
        words_df.to_csv(words_output_file, index=False)
        shutil.copyfile(f"{INPUT_DIR}{dataset}/Task_1/sentences.csv", sentences_output_file)
        print(f"{output_path} task 1 files done")

    words_output_file = f"{output_path}/t2_words.csv"
    sentences_output_file = f"{output_path}/t2_sentences.csv"


    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} task 2 files already exist - skipping creation")
    else:
        words_df = pd.read_csv(f"{INPUT_DIR}{dataset}/Task_2/words.csv")
        words_df["WORD_ID"] = words_df["SENTENCE_ID"].astype(str) + "-" + words_df["WORD_ID"].astype(str)
        words_df.to_csv(words_output_file, index=False)
        shutil.copyfile(f"{INPUT_DIR}{dataset}/Task_2/sentences.csv", sentences_output_file)
        print(f"{output_path} task 2 files done")

    words_output_file = f"{output_path}/t3_words.csv"
    sentences_output_file = f"{output_path}/t3_sentences.csv"

    if path.isfile(words_output_file) and path.isfile(sentences_output_file):
        print(f"{output_path} task 3 files already exist - skipping creation")
    else:
        words_df = pd.read_csv(f"{INPUT_DIR}{dataset}/Task_3/words.csv")
        words_df["WORD_ID"] = words_df["SENTENCE_ID"].astype(str) + "-" + words_df["WORD_ID"].astype(str)
        words_df.to_csv(words_output_file, index=False)
        shutil.copyfile(f"{INPUT_DIR}{dataset}/Task_3/sentences.csv", sentences_output_file)
        print(f"{output_path} task 3 files done")


def main():
    if not path.isdir(path.join(INPUT_DIR, SOOD_DATASET)):
        print(f"Cannot find {SOOD_DATASET} - skipping creation")
    else:
        create_sood_et_al_text_data(SOOD_DATASET)

    if not path.isdir(path.join(INPUT_DIR, SARCASM_DATASET)):
        print(f"Cannot find {SARCASM_DATASET} - skipping creation")
    else:
        create_mishra_sarcasm_text_data(SARCASM_DATASET)

    if not path.isdir(path.join(INPUT_DIR, GECO_DATASET)):
        print(f"Cannot find {GECO_DATASET} - skipping creation")
    else:
        create_geco_text_data(GECO_DATASET)

    if not path.isdir(path.join(INPUT_DIR, ZUCO_DATSET)):
        print(f"Cannot find {ZUCO_DATSET} - skipping creation")
    else:
        create_zuco_text_data(ZUCO_DATSET)


if __name__ == "__main__":
    main()
