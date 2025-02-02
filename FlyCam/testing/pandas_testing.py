# Testing for PANDAS creation, saving, and opening of CSV files
# Source: https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm

# Import PANDAS Library
import pandas as pd


# Create pandas variable
# pandas.DataFrame( data, index, columns, dtype, copy)

# Create Empty Data Frame
# df = pd.DataFrame()

# Create DataFrame with data
# data = [1, 2, 3, 4, 5]
# df = pd.DataFrame(data)


# Create DataFrame with data and column headers
# Data is a list of row vectors
# data = [["Alex", 10], ["Bob", 12], ["Clarke", 13]]
# df = pd.DataFrame(data, columns=["Name", "Age"])

# Create DataFrame with data and column headers and data type
# Data is a list of row vectors
# data = [["Alex", 10], ["Bob", 12], ["Clarke", 13]]
# df = pd.DataFrame(data, columns=["Name", "Age"], dtype=float)

# Notes: Possible to pass in a list of dictionaries, may want to do that for location lists


# print(df)

# Save to CSV
# Source: https://www.geeksforgeeks.org/saving-a-pandas-dataframe-as-a-csv/
# https://www.geeksforgeeks.org/how-to-export-pandas-dataframe-to-a-csv-file/
# df.to_csv("file1.csv")

# df.to_csv("file1.csv", sep="\t")

# Open CSV
# index_col=0 to remove the "Unnamed" column title
# test = pd.read_csv("file1.csv", index_col=0)
# print(test)

# Sample Data:
data = {"x": [10, 20, 30, 40, 50], "y": [15, 25, 35, 45, 55], "z": [18, 28, 38, 48, 58]}
df = pd.DataFrame(data)
print(df)

# TODO: Try iterating over rows or parse CSV into list of lists? Or generate into GCODE String right away. [DONE]
# https://www.tutorialspoint.com/python_pandas/python_pandas_iteration.htm

for row_index, row in df.iterrows():
    print("==============")
    # print(row_index, row)
    # print("x:", row[0])
    # print("y:", row[1])
    # print("z:", row[2])
    sample_gcode = "G0X{}Y{}Z{}".format(row[0], row[1], row[2])
    print("sample_gcode:", sample_gcode)

