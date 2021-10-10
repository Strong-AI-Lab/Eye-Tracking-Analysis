import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead
from utils import word_formatter
from collections import OrderedDict

INPUT_DIR = "output/text_data/"
OUTPUT_DIR = "output/normalized_attention_data/"
SOOD_DATASET = "sood_et_al_2020"
SARCASM_DATASET = "Mishra/Eye-tracking_and_SA-II_released_dataset"
GECO_DATASET = "GECO"


def load_models(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name, output_attentions=True)

    return tokenizer, model


def load_data(sentence_file, word_file):
    sentence_df = pd.read_csv(sentence_file)
    word_df = pd.read_csv(word_file)
    word_df['norm word'] = word_df['WORD'].apply(word_formatter)

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
    attentions = model(inputs)['attentions']

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


def get_attention_scores(sample, tokenizer, special_tokens=True):
    inputs, tokens_per_word = process_sample(sample, tokenizer, special_tokens=special_tokens)
    attention_scores = process_attention_scores(inputs, special_tokens=special_tokens)

    index = 0
    final_scores = []
    word_id = 0

    for i, (word, tokens) in enumerate(tokens_per_word):
        current_word_scores = OrderedDict()
        scores = attention_scores.iloc[index:index + tokens]
        index = index + tokens

        current_word_scores['min'] = scores.apply(np.min)
        current_word_scores['max'] = scores.apply(np.max)
        current_word_scores['mean'] = scores.apply(np.mean)
        current_word_scores['median'] = scores.apply(np.median)
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


def create_df_from_sentence(sentence_id, sentence_df, word_df, mask, tokenizer, special_tokens=True):
    sentence = sentence_df.loc[sentence_df['SENTENCE_ID'] == sentence_id, 'SENTENCE'].iloc[0]
    attention_scores = get_attention_scores(sentence, tokenizer, special_tokens=special_tokens)

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

    assert (df['WORD'].reset_index(drop=True) == original_df['norm word'].reset_index(drop=True)).sum() == \
           original_df.shape[0]
    df["WORD_ID"] = original_df["WORD_ID"].reset_index(drop=True)

    return df


def run_experiment(model_name, data):
    pass


def main():
    pass


if __name__ == "__main__":
    "Run from run experiment"
