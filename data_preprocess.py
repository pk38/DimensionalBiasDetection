import pandas as pd
from utils_movie_bias import str_to_list
import numpy as np

class DataCollect:
    def __init__(self, location) -> None:
        """
        Loading data, converting and cleaning it
        """
        # Load data into local environment
        self.data = pd.read_csv(location)
        for col in ['Directors', 'Cast', 'genres']:
            self.simplify_str(col)
        self.clean_data()
        for col in ['Directors', 'Cast', 'genres']:
            self.drop_wrong_col_type(col, str)

    def simplify_str(self, col):
        """
        Converting string Columns to suitable lists.
        """
        # apply function to 'col' column
        self.data[col] = self.data[col].apply(str_to_list)

    def clean_data(self):
        """
        Dropping all Nan rows
        """
        # Replace empty lists with NaN values
        self.data = self.data.replace(to_replace=[], value=np.nan)

        # Remove rows with NaN values
        self.data = self.data.dropna()

    def drop_wrong_col_type(self, col, type_col):
        """
        Cleaning step - Dropping rows with no information of wrong formats
        """
        # Iterate over each row of the dataframe
        for index, row in self.data.iterrows():
            # Check if the values in the row list are all strings
            if all(isinstance(val, type_col) for val in row[col]):
                # If they are all strings, do nothing
                pass
            else:
                # If not, drop the row
                self.data = self.data.drop(index)