# Convert a list of lists (or a matrix-like) or coordinates into GCode formatted strings

# Import libraries
import yaml


# Function that takes in a list, returns a similar dimension list of gcode formatted strings
def convert_list_to_gcode_strings(location_list):
    X = 0; Y = 1; Z = 2

    print("convert_list_to_gcode_strings")
    position = "G0"
    gcode_string_list = []
    number_of_rows = len(location_list)
    number_of_columns = len(location_list[0])

    # TODO: Can I achieve this with a list comprehension?
    # For loop goes through each row
    for row_num in range(number_of_rows):
        print("row_num:", row_num)
        # For loop goes through each column
        for col_num in range(number_of_columns):
            print("col_num:", col_num)
            print(location_list[row_num][col_num])
            current_location = location_list[row_num][col_num]
            x = current_location[X]
            y = current_location[Y]
            z = current_location[Z]
            print(x, y, z)
            gcode_string = "{}X{}Y{}Z{}".format(position, x, y, z)
            print(gcode_string)
            gcode_string_list.append(gcode_string)
    print(gcode_string_list)
    pass


# Function loads yaml file, extracts location list, converts to a list of lists
def get_location_list_from_yaml(yaml_file):
    # print("get_location_list_from_yaml")
    with open(yaml_file) as file:
        location_list = yaml.load(file, Loader=yaml.FullLoader)
        print(location_list)
    return location_list


def main():
    yaml_file = "location_list_2x3_all.yaml"
    location_list = get_location_list_from_yaml(yaml_file)
    convert_list_to_gcode_strings(location_list)
    pass


main()

