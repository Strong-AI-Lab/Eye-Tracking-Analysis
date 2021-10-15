import glob
import pandas as pd
from os import path, makedirs

INPUT_DIR = "output/attention_data/paragraphs/"
OUTPUT_DIR = "output/normalized_attention_data/paragraphs/"
FILE_INDEX = len(INPUT_DIR.split("/")) - 1


def normalize_data(filepath):
    split_path = filepath.split("/")
    output_dir = f'{OUTPUT_DIR}{"/".join(split_path[FILE_INDEX:-1])}/'
    output_file = output_dir + split_path[-1]
    print(output_file)

    if path.isfile(output_file):
        print(f"{output_file} already exists - skipping creation")
    else:
        transformer_df = pd.read_csv(filepath)
        normalised_dfs = []

        for paragraph in transformer_df["PARAGRAPH_ID"].unique():
            print(paragraph)

            mask = transformer_df["PARAGRAPH_ID"] == paragraph
            current_df = transformer_df[mask].select_dtypes(exclude="object")
            normalised_dfs.append(current_df / current_df.sum())

        normal_trans_df = pd.concat(normalised_dfs)

        if normal_trans_df.shape[0] == transformer_df.shape[0]:
            full_normal_trans = pd.concat([transformer_df, normal_trans_df], ignore_index=True, axis=1)
            full_normal_trans.columns = list(transformer_df.columns) + list(
                map(lambda s: f"{s}_norm", list(normal_trans_df.columns)))

            if not path.isdir(output_dir):
                makedirs(output_dir)

            full_normal_trans.to_csv(output_file)


def main():
    filepaths = [file.replace("\\", "/") for file in glob.glob(f'{INPUT_DIR}/**/*.csv', recursive=True)]
    
    for filepath in filepaths:
        normalize_data(filepath)
    
    print("Done!")


if __name__ == "__main__":
    main()
