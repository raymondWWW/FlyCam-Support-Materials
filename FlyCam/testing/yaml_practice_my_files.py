# Import libraries
import yaml

# Load YAML files
with open("connection_settings.yaml") as file:
    # The fullloader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    connection_settings_dict = yaml.load(file, Loader=yaml.FullLoader)
    # How to access dict items
    # print(connection_settings_dict["cell_sensor"]["monoprice"]["device_path"])

    print(connection_settings_dict["cell_sensor"]["monoprice"]["max"]["test"][2])

