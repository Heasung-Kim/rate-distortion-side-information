
import re

def parse_txt_file(filename):
    # Initialize variables to store parsed values
    last_directory = None
    last_distortion = None
    last_rate = None
    last_BER = None
    last_BLER = None

    # Define a regular expression pattern to match numbers
    number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'

    # Open the file and read its contents
    with open(filename, 'r') as file:
        content = file.read()

    # Use regular expressions to find and extract information
    matches = re.findall(r'ckpt dir:\s*(.*?)\s*\(distortion,\s*rate\):\s*\(({}),\s*({})\)\s*BER:\s*\[([^\]]+)\]\s*BLER:\s*\[([^\]]+)\]'.format(number_pattern, number_pattern), content)

    if matches:
        # Extract the last set of values
        last_match = matches[-1]
        last_directory = last_match[0]
        last_distortion = float(last_match[1])
        last_rate = float(last_match[2])
        last_BER = [float(x) for x in re.findall(number_pattern, last_match[3])]
        last_BLER = [float(x) for x in re.findall(number_pattern, last_match[4])]

    return last_distortion, last_rate, last_BER, last_BLER

def parse_last_distortion_rate(file_path):
    last_distortion = None
    last_rate = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("(distortion,rate):"):
                distortion, rate = map(float, line.split(":")[1].strip()[1:-1].split(","))
                last_distortion = distortion
                last_rate = rate

    return last_distortion, last_rate


def get_dataset(config):
    if "2WGNAVG" in config["dataset"]:
        from data.loader.WGNAVG_dataset_loader import get_2WGNAVG_dataset
        return get_2WGNAVG_dataset(config=config)

    elif "2WGN" in config["dataset"]:
        from data.loader.WGN_dataset_loader import get_2WGN_dataset
        return get_2WGN_dataset(config=config)

    else:
        raise NotImplementedError
