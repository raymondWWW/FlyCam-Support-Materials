# Testing out reading and writing to YAML
# Sources:
# https://stackabuse.com/reading-and-writing-yaml-to-a-file-in-python/
# https://pyyaml.org/wiki/PyYAMLDocumentation
#  Recommends using libYAML

# Import libraries
import yaml

# Load YAML files
with open("fruits.yaml") as file:
    # The fullloader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    fruits_list = yaml.load(file, Loader=yaml.FullLoader)
    print(fruits_list)

with open("categories.yaml") as file:
    documents = yaml.full_load(file)

    for item, doc in documents.items():
        print(item, ":", doc)

# Writing YAML Files
dict_file = [{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}]

with open("store_file.yaml", "w") as file:
    documents = yaml.dump(dict_file, file)