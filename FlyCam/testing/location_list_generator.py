# Generates location list based on starting location and well distance.

# Load Library
import yaml
import pandas as pd

# CONSTANTS
# TODO: Move to a settings or common file
PATH_CSV_FILE_NAME = "sample_path"
LOCATION_CSV_FILE_NAME = ""

# Load YAML File
# For now, dummy data
# sample_plate_specifications = {
#     "number_of_rows": 2,
#     "number_of_columns": 3,
#     "starting_location": (50, 140, 50),
#     "well_distance": (38, 38)
# }


# Generate location list
# Given the known starting location, number of rows, number of columns, and well distance (x and y)
# get the absolute position of all wells.
# Usage: A Matrix representing all well locations in mm
def get_location_list(sample_plate_specifications):
    # TODO: Prepare for float, or change to float values
    X = 0; Y = 1; Z = 2
    # print(sample_plate_specifications)
    number_of_rows = sample_plate_specifications["number_of_rows"]
    number_of_columns = sample_plate_specifications["number_of_columns"]
    starting_x = sample_plate_specifications["starting_location"][X]
    starting_y = sample_plate_specifications["starting_location"][Y]
    starting_z = sample_plate_specifications["starting_location"][Z]
    well_distance_x = sample_plate_specifications["well_distance"][X]
    well_distance_y = sample_plate_specifications["well_distance"][Y]
    number_of_wells = number_of_rows * number_of_columns

    # Create matrix of locations to match the actual well locations
    # TODO: Should I use NumPy?
    # Format (if a 2x3):
    #   [ (x, y, z), (x, y, z), (x, y, z)
    #     (x, y, z), (x, y, z), (x, y, z) ]
    location_list = []

    # for loop that goes through each row
    for row_num in range(number_of_rows):
        # print("row_num", row_num)
        # Reset/set temp x
        temp_x = starting_x
        #   temp_y changes each iteration here (add a well_distance_y)
        temp_y = starting_y + (well_distance_y * row_num)
        #   temp_z stays the same throughout
        temp_z = starting_z
        # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)

        # Create row
        location_list.append([])
        #   for loop that goes through each column
        for col_num in range(number_of_columns):
            # print("col_num:", col_num)
            #     temp_x changes each iteration here
            temp_x = starting_x + (well_distance_x * col_num)
            # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)
            temp_location = [temp_x, temp_y, temp_z]
            location_list[row_num].append(temp_location)
        # print("Just finished a row")
        # print(location_list)
    return location_list

# Function: Create a path list of well locations based on sample plate specifications, then returns that list
# Usage: Exact locations to go to (then take pic or video)
def get_path_list(sample_plate_specifications):
    X = 0;
    Y = 1;
    Z = 2
    # print(sample_plate_specifications)
    number_of_rows = sample_plate_specifications["number_of_rows"]
    number_of_columns = sample_plate_specifications["number_of_columns"]
    starting_x = sample_plate_specifications["starting_location"][X]
    starting_y = sample_plate_specifications["starting_location"][Y]
    starting_z = sample_plate_specifications["starting_location"][Z]
    well_distance_x = sample_plate_specifications["well_distance"][X]
    well_distance_y = sample_plate_specifications["well_distance"][Y]
    number_of_wells = number_of_rows * number_of_columns

    # Create matrix of locations to match the actual well locations
    # TODO: Should I use NumPy?
    # Format (if a 2x3):
    #   [ (x, y, z), (x, y, z), (x, y, z)
    #     (x, y, z), (x, y, z), (x, y, z) ]
    # Or Maybe don't use tuples, use lists instead
    # TODO: Test how YAML saves tuples, if at all
    path_list = []

    # for loop that goes through each row
    for row_num in range(number_of_rows):
        # print("row_num", row_num)
        # Reset/set temp x
        temp_x = starting_x
        #   temp_y changes each iteration here (add a well_distance_y)
        temp_y = starting_y - (well_distance_y * row_num)
        #   temp_z stays the same throughout
        temp_z = starting_z
        # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)

        # Create row
        # location_list.append([])
        #   for loop that goes through each column
        for col_num in range(number_of_columns):
            # print("col_num:", col_num)
            #     temp_x changes each iteration here
            temp_x = starting_x + (well_distance_x * col_num)
            # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)
            temp_location = [temp_x, temp_y, temp_z]
            path_list.append(temp_location)
        # print("Just finished a row")
        # print(location_list)
    return path_list

# TODO: Create new function that saves to a PANDAS file instead.
# Function: Create a path list of well locations based on sample plate specifications, then returns that list
# Usage: Exact locations to go to (then take pic or video)
def get_path_dataframe(sample_plate_specifications):
    # Extract data from Sample Plate Specs, just like in the previous function

    # Setup Function Constants
    X = 0;
    Y = 1;
    Z = 2
    column_titles_list = ["X", "Y", "Z"]


    # print(sample_plate_specifications)
    number_of_rows = sample_plate_specifications["number_of_rows"]
    number_of_columns = sample_plate_specifications["number_of_columns"]
    starting_x = sample_plate_specifications["starting_location"][X]
    starting_y = sample_plate_specifications["starting_location"][Y]
    starting_z = sample_plate_specifications["starting_location"][Z]
    well_distance_x = sample_plate_specifications["well_distance"][X]
    well_distance_y = sample_plate_specifications["well_distance"][Y]
    number_of_wells = number_of_rows * number_of_columns

    # Create Empty Dataframe
    dataframe_path = pd.DataFrame()

    # Use for loop to go through each row
    for row_num in range(number_of_rows):
        # Reset/set temp x
        temp_x = starting_x

        #   temp_y changes each iteration here (add a well_distance_y)
        temp_y = starting_y - (well_distance_y * row_num)

        #   temp_z stays the same throughout
        temp_z = starting_z

        # Uncomment to see what each row value is at the beginning
        # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)
        # Use nested for loop to go through each column
        for col_num in range(number_of_columns):
            # Increase x as it goes to each column, y and z stay the same (Assumes level sample plate)
            temp_x = starting_x + (well_distance_x * col_num)

            # Uncomment to see each location.
            # Questions to consider:
            #  Is y going towards zero as the rows increase (meaning the bottom left is 0)?
            #  Is z staying the same?
            # print("temp_x:", temp_x, "temp_y:", temp_y, "temp_z:", temp_z)

            # Create Dictionary with column titles, will be used for DataFrame
            data = {column_titles_list[X]: [temp_x], column_titles_list[Y]: [temp_y], column_titles_list[Z]: [temp_z]}

            # Create a single row for dataframe
            df = pd.DataFrame(data)

            # Append new row to dataframe_path variable
            dataframe_path = dataframe_path.append(df)

            # TODO: Consider creating column vectors, then creating dataframe at the end
            #  Note: Current method recreates DataFrame over and over, may be memory intensive
    return dataframe_path


def main():
    sample_plate_specifications = {
        "number_of_rows": 2,
        "number_of_columns": 3,
        "starting_location": [50, 140, 50],
        "well_distance": [38, 38]
    }

    # Start Location List Creation
    # location_list = get_location_list(sample_plate_specifications)
    # print(location_list)
    #
    # number_of_rows = sample_plate_specifications["number_of_rows"]
    # number_of_columns = sample_plate_specifications["number_of_columns"]
    #
    # # file name example: location_list_2x3_all.yaml
    # filename_yaml = "location_list_{}x{}_all.yaml".format(number_of_rows, number_of_columns)
    #
    # # dict_file = []
    # # TODO: Save a location as a Python List: [50, 178, 50] instead of something like this:
    # #   - - - 50
    # #       - 140
    # #       - 50
    # #     - - 88
    #
    # with open(filename_yaml, "w") as file:
    #     documents = yaml.dump(location_list, file)

    # Old Way, the YAML Way
    # Start Path List Creation
    # path_list = get_path_list(sample_plate_specifications)
    # print(path_list)

    # Save Path List to a YAML file
    # number_of_rows = sample_plate_specifications["number_of_rows"]
    # number_of_columns = sample_plate_specifications["number_of_columns"]
    #
    # # file name example: location_list_2x3_all.yaml
    # filename_yaml = "path_list_{}x{}_all.yaml".format(number_of_rows, number_of_columns)
    #
    # # dict_file = []
    # # TODO: Save to CSV File
    #
    # with open(filename_yaml, "w") as file:
    #     documents = yaml.dump(path_list, file)

    # New Way, the PANDAS/CSV way
    # path_list = get_path_list(sample_plate_specifications)
    # print(path_list)

    dataframe = get_path_dataframe(sample_plate_specifications)
    print(dataframe)

    dataframe.to_csv("file2.csv")




main()

