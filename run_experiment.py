import argparse
from extract_transformer_attention import control_extraction

data_list = None
model_list = None

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default="all")
parser.add_argument("-m", "--model", default="all")
args = parser.parse_args()

if args.data == "all":
    data_file = open("texts-english.txt", "r")
    data_list = [data.replace("\n", "").lower() for data in data_file.readlines()]
    data_file.close()
else:
    data_list = [args.data.lower()]

if args.model == "all":
    model_file = open("models-english.txt", "r")
    model_list = [model.replace("\n", "") for model in model_file.readlines()]
    model_file.close()
else:
    model_list = [args.model]


def main():
    for model in model_list:
        for dataset in data_list:
            control_extraction(model, dataset)


if __name__ == "__main__":
    main()
