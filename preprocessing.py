import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Preprocessing transformers


def categorical_data_to_numerical_encoding(df_cat):
    # Encode categorical variables
    # Create an instance of the OneHotEncoder with sparse=False to obtain a dense array
    encoder = OneHotEncoder(sparse=False)
    encoded_data_cat = encoder.fit_transform(df_cat)

    # Specify the categorical columns to be encoded
    categorical_columns = df_cat.columns[[0, 1, 2, 3, 4]]

    # Create a new DataFrame 'encoded_df' with the encoded data and column names
    encoded_df_cat = pd.DataFrame(encoded_data_cat, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_df_cat.head()

    return encoded_data_cat

def df_to_normalized_ndarray(df_num, df):
    # Normalize numerical columns
    # Create an instance of the MinMaxScaler for normalization
    scaler = MinMaxScaler()

    # Create an empty DataFrame to store the normalized numerical columns
    normalized_numerical = pd.DataFrame()

    # Iterate over each numerical column
    for column in df_num:
        try:
            # Normalize the column values using the scaler and flattening the result
            normalized_column = scaler.fit_transform(df[[column]])
            normalized_numerical[column] = normalized_column.flatten()
        except ValueError:
            # Skip the column if it contains non-numeric values
            print(f"Skipping column {column} due to non-numeric values.")

    # Convert the normalized_numerical DataFrame to a numerical array
    data_array = np.array(normalized_numerical)

    return data_array

def preprocessing():
    dataset_paths = ["Data/07_17_2023_v1_GeneID_Query__SSTR1.csv"]
    preprocessed_data = []

    for path in dataset_paths:
        result_data = []

        # Load data and preparing the dataset
        # Load data from a CSV file into a pandas DataFrame
        df = pd.read_csv(path)
        df.drop_duplicates(inplace=True)
        if "Unnamed: 0" in df.columns.values:
            df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

        # Dataframe
        df = df.set_index("REM_ID")

        # Label vector
        df_lbl = df['Predicted_function']                           #(('REM0639547', 'activating')...

        # Dataset without labels
        df_nlbl = df.drop(columns=['Predicted_function'], axis=1)

        # Creating a DataFrame df_cat containing only the categorical attributes
        df_cat = df_nlbl.select_dtypes(exclude=["number"])

        # Creating a list df_num of column names for the numeric attributes
        df_num = df_nlbl.select_dtypes(include=["number"]).columns.tolist()

        # All predicted function categories
        all_cat = []                                                #['activating', 'repressing']
        for i in df_lbl:
            if i not in all_cat:
                all_cat.append(i)

        # Normalized the df_num
        data_array = df_to_normalized_ndarray(df_num, df)

        # Encode the df_cat
        encoded_data_cat = categorical_data_to_numerical_encoding(df_cat)

        # Concatenate encoded categorical data with numerical data array
        concatenated_data = np.concatenate((encoded_data_cat, data_array), axis=1)

        # Result preparation
        result_data.append(concatenated_data)
        result_data.append(df_lbl)
        result_data.append(all_cat)
        preprocessed_data.append(result_data)

        # Writing results in a file
        with open('CCA_preprocessing.txt', 'w') as saveFile:
            saveFile.write("--------Successful Run--------")
            saveFile.write("\n")
            saveFile.write("--------Dataframe columns--------")
            saveFile.write("\n")
            saveFile.write(str(df.columns.values))
            saveFile.write("\n")
            saveFile.write("--------Dataframe--------")
            saveFile.write("\n")
            saveFile.write(str(df))
            saveFile.write("\n")
            saveFile.write("--------Preprocessed data shape--------")
            saveFile.write("\n")
            saveFile.write(str(concatenated_data.shape))
            saveFile.write("\n")

    with open("CCA_preprocessing.txt", "a") as output:
        for row in preprocessed_data:
            output.write("--------Preprocessed data--------")
            output.write('\n')
            s = " ".join(map(str, row))
            output.write(s + '\n')

    return preprocessed_data


if __name__ == '__main__':
    preprocessing()