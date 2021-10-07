from os import path, makedirs


def create_output_dir(dataset, output_dir):
    output_path = path.join(output_dir, dataset)
    if not path.isdir(output_path):
        makedirs(output_path)

    return output_path
