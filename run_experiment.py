import argparse

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
    if "geco" in data_list:
        print("GECO is here")

    print(model_list)


if __name__ == "__main__":
    main()
