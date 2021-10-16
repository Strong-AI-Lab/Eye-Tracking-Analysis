from extract_transformer_attention_by_paragraph import control_extraction

data_file = open("texts-english.txt", "r")
data_list = [data.replace("\n", "") for data in data_file.readlines()]
data_file.close()

model_file = open("models-english.txt", "r")
model_list = [model.replace("\n", "") for model in model_file.readlines()]
model_file.close()


def main():
    errors = []

    for model in model_list:
        for dataset in data_list:
            try:
                control_extraction(model, dataset)
            except Exception as e:
                errors.append(f"{model} - {dataset} - {e}")

    error_file = open("paragraph_extraction_errors.txt", "w")
    for error in errors:
        error_file.write(error + "\n")
    error_file.close()



if __name__ == "__main__":
    main()
