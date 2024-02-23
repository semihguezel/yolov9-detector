import yaml
import random


def generate_random_rgb_color_list(n):
    """
    Generate a list of random RGB colors.

    Args:
    - n (int): Number of colors to generate.

    Returns:
    - list: A list of randomly generated RGB colors.
    """
    color_list = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]
    return color_list


def read_yaml_file(file_path):
    """
    Read YAML file and return its content.

    Args:
    - file_path (str): Path to the YAML file.

    Returns:
    - dict: Dictionary containing the data from the YAML file.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None


def get_labels_from_txt(file_path):
    # Initialize an empty dictionary
    my_dict = {}

    # Open the file and read its content
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into key and value (assuming they are separated by a delimiter, e.g., column)
            key, value = line.strip().split(':')
            # Add key-value pair to the dictionary
            my_dict[int(key)] = value

    # Now, my_dict contains the data from the text file as a dictionary
    return my_dict
