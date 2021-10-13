import argparse
from os import path
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead
from utils import word_formatter, create_output_dir
from collections import OrderedDict

INPUT_DIR = "output/text_data/"
OUTPUT_DIR = "output/attention_data/sentences/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="GECO")
parser.add_argument("-m", "--model", default="bert-base-cased")
args = parser.parse_args()


def load_models(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name, output_attentions=True)

    return tokenizer, model


def load_data(sentence_file, word_file):
    sentence_df = pd.read_csv(sentence_file)
    word_df = pd.read_csv(word_file)
    word_df['norm word'] = word_df['WORD'].astype(str).apply(word_formatter)

    return sentence_df, word_df


def word_intersection(sample, original):
    return set(sample).intersection(original)


def convert_to_regex(s):
    s = f"^{s}$"
    return s


def process_sample(sentence, tokenizer, special_tokens=True):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=special_tokens)
    individual_tokens = tokenizer.batch_encode_plus(sentence.split(), add_special_tokens=False)['input_ids']

    tokens_per_word = list(zip(sentence.split(), [len(item) for item in individual_tokens]))
    tokens = inputs['input_ids']

    return tokens, tokens_per_word


def process_attention_scores(inputs, model, special_tokens=True):
    processed_scores = {}
    model_output = model(inputs)
    if "attentions" in model_output.keys():
        attentions = model(inputs)['attentions']
    elif "encoder_attentions" in model_output.keys():
        attentions = model(inputs)['encoder_attentions']

    for i in range(len(attentions)):
        processed_scores[f"L{i + 1}_attention-mean"] = attentions[i][0].mean(0).mean(0).detach().numpy()
        processed_scores[f"L{i + 1}_attention-max"] = torch.max(torch.max(attentions[i][0], 0).values,
                                                                0).values.detach().numpy()
        processed_scores[f"L{i + 1}_attention-min"] = torch.min(torch.min(attentions[i][0], 0).values,
                                                                0).values.detach().numpy()

        if special_tokens:
            processed_scores[f"L{i + 1}_cls-attention-mean"] = attentions[i][0, :, 0].mean(0).detach().numpy()
            processed_scores[f"L{i + 1}_cls-attention-max"] = torch.max(attentions[i][0, :, 0],
                                                                        0).values.detach().numpy()
            processed_scores[f"L{i + 1}_cls-attention-min"] = torch.min(attentions[i][0, :, 0],
                                                                        0).values.detach().numpy()

    processed_df = pd.DataFrame.from_dict(processed_scores)
    if special_tokens:
        processed_df = processed_df.iloc[1:-1].reset_index(drop=True)

    return processed_df


def get_attention_scores(sample, model, tokenizer, device, special_tokens=True):
    inputs, tokens_per_word = process_sample(sample, tokenizer, special_tokens=special_tokens)
    inputs.to(torch.device(device))
    print(inputs.device)
    attention_scores = process_attention_scores(inputs, model, special_tokens=special_tokens)

    index = 0
    final_scores = []
    word_id = 0

    for i, (word, tokens) in enumerate(tokens_per_word):
        current_word_scores = OrderedDict()
        scores = attention_scores.iloc[index:index + tokens]
        index = index + tokens

        current_word_scores['max'] = scores.apply(np.max)
        current_word_scores['mean'] = scores.apply(np.mean)
        current_word_scores['sum'] = scores.apply(np.sum)

        for key, score in current_word_scores.items():
            score.index = map(lambda s: f"{s}_word-{key}", score.index)

        current_word_df = pd.concat([current_word_scores[key] for key in current_word_scores.keys()])
        current_word_df['WORD'] = word
        word_id = word_id + 1
        final_scores.append(current_word_df.to_frame().transpose())

    return pd.concat(final_scores).reset_index(drop=True)


def get_final_df(final_df, orignal_df):
    final_df['WORD'] = final_df['WORD'].apply(word_formatter)

    sample_word_set = final_df['WORD'].unique()
    original_word_set = orignal_df["norm word"].unique()
    keep_words = word_intersection(sample_word_set, original_word_set)
    keep_words = list(map(convert_to_regex, keep_words))

    pattern = '|'.join(keep_words)
    mask = final_df['WORD'].str.contains(pattern)

    return final_df[mask]


def create_df_from_sentence(sentence_id, sentence_df, word_df, model, mask, tokenizer, device, special_tokens=True):
    sentence = sentence_df.loc[sentence_df['SENTENCE_ID'] == sentence_id, 'SENTENCE'].iloc[0]
    attention_scores = get_attention_scores(sentence, model, tokenizer, device, special_tokens=special_tokens)

    original_df = word_df[mask]

    df = get_final_df(attention_scores, original_df)

    # check new df words are the same as old
    if df.shape[0] != original_df.shape[0]:
        new_idx = 0
        orig_idx = 0

        while (new_idx < df.shape[0]) and (orig_idx < original_df.shape[0]):
            if df['WORD'].iloc[new_idx].lower() == original_df['norm word'].iloc[orig_idx]:
                new_idx = new_idx + 1
                orig_idx = orig_idx + 1
            else:
                to_drop = df.index[new_idx]
                df = df.drop(to_drop)

        while new_idx < df.shape[0]:
            to_drop = df.index[new_idx]
            df = df.drop(to_drop)
            new_idx = new_idx + 1

    df = df.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    if df.shape[0] > original_df.shape[0]:
        df = df.loc[:original_df.shape[0] - 1]
    elif df.shape[0] < original_df.shape[0]:
        original_df = original_df.loc[:df.shape[0] - 1]

    assert (df['WORD'] == original_df['norm word']).sum() == \
           original_df.shape[0]
    df["WORD_ID"] = original_df["WORD_ID"].reset_index(drop=True)

    return df


def extraction_loop(model_name, sentence_df, word_df, device):
    tokenizer, model = load_models(model_name)
    model.to(torch.device(device))

    dfs = []
    errors = []

    for i, sentence_id in sentence_df["SENTENCE_ID"].iteritems():
        print(f"{i + 1} of {sentence_df.shape[0]}")
        mask = word_df['SENTENCE_ID'] == sentence_id
        if mask.sum() == 0:
            errors.append(f"{sentence_id} - Zero!")
            continue

        try:
            df = create_df_from_sentence(sentence_id, sentence_df, word_df, model, mask, tokenizer, device)
            dfs.append(df)

        except Exception as e:
            errors.append(f"{sentence_id} - {e}")

    return dfs, errors


def run_extraction(model_name, dataset, sentence_file, word_file, output_file):
    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")

    else:
        sentence_df, word_df = load_data(sentence_file, word_file)
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"

        dfs, errors = extraction_loop(model_name, sentence_df, word_df, device)

        if len(errors) > 0:
            output_path = create_output_dir(dataset, f"{OUTPUT_DIR}/errors/")
            error_file = open(f"{output_path}/{model_name.replace('/','-')}.txt", "w")
            for error in errors:
                error_file.write(error + "\n")
            error_file.close()

        processed_df = pd.concat(dfs)
        processed_df = pd.merge(word_df, processed_df, on="WORD_ID")
        processed_df.to_csv(output_file)
        print(f"{output_file} Done!")


def control_extraction(model_name, dataset, special_tokens=True):
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    if dataset == GECO_DATASET:
        output_path = create_output_dir(dataset, OUTPUT_DIR)
        sentence_file = f"{INPUT_DIR}{dataset}/EnglishMaterialSENTENCE.csv"
        word_file = f"{INPUT_DIR}{dataset}/EnglishMaterialALL.csv"
        output_file = f"{output_path}/{model_name.replace('/','-')}-{special_tokens}.csv"
        run_extraction(model_name, dataset, sentence_file, word_file, output_file)

    if dataset == SOOD_DATASET:
        output_path = create_output_dir(dataset, OUTPUT_DIR)
        sentence_file = f"{INPUT_DIR}{dataset}/study1_sentences.csv"
        word_file = f"{INPUT_DIR}{dataset}/study1_words.csv"
        output_file = f"{output_path}/study_1_{model_name.replace('/','-')}-{special_tokens}.csv"
        run_extraction(model_name, dataset, sentence_file, word_file, output_file)

        sentence_file = f"{INPUT_DIR}{dataset}/study2_sentences.csv"
        word_file = f"{INPUT_DIR}{dataset}/study2_words.csv"
        output_file = f"{output_path}/study_2_{model_name.replace('/','-')}-{special_tokens}.csv"
        run_extraction(model_name, dataset, sentence_file, word_file, output_file)

    if dataset == SARCASM_DATASET:
        output_path = create_output_dir(dataset, OUTPUT_DIR)
        sentence_file = f"{INPUT_DIR}{dataset}/sentences.csv"
        word_file = f"{INPUT_DIR}{dataset}/words.csv"
        output_file = f"{output_path}/{model_name.replace('/','-')}-{special_tokens}.csv"
        run_extraction(model_name, dataset, sentence_file, word_file, output_file)


def main():
    control_extraction(args.model, args.data)


if __name__ == "__main__":
    main()
