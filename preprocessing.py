import numpy as np  # Numerical computing library
import pandas as pd  # Data manipulation library
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Preprocessing transformers


def categorical_data_to_numerical_encoding(df_cat):
    # 5--> Encoding categorical variables

    # Creating an instance of the OneHotEncoder with sparse=False to obtain a dense array
    encoder = OneHotEncoder(sparse=False)
    encoded_data_cat = encoder.fit_transform(df_cat)

    # Printing the head (first few rows) of the df_cat DataFrame
    print(df_cat.head())

    # Printing the encoded representation of the first row of the categorical data
    print(encoded_data_cat[0])

    # Specifying the categorical columns to be encoded
    categorical_columns = ['Gene_ID', 'Gene_Symbol', 'REM_ID', 'Chr', 'Predicted_function', 'CREM_ID']

    # Creating a new DataFrame 'encoded_df' with the encoded data and column names
    encoded_df_cat = pd.DataFrame(encoded_data_cat, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_df_cat.head()

    return encoded_data_cat

def df_to_normalized_ndarray(df_num, df):
    # 3--> Normalizing numerical columns

    # Creating an instance of the MinMaxScaler for normalization
    scaler = MinMaxScaler()

    # Creating an empty DataFrame to store the normalized numerical columns
    normalized_numerical = pd.DataFrame()

    # Iterating over each numerical column
    for column in df_num:
        try:
            # Normalizing the column values using the scaler and flattening the result
            normalized_column = scaler.fit_transform(df[[column]])
            normalized_numerical[column] = normalized_column.flatten()
        except ValueError:
            # Skipping the column if it contains non-numeric values
            print(f"Skipping column {column} due to non-numeric values.")

    # Displaying the head of the normalized_numerical DataFrame
    normalized_numerical.head()

    # 4--> Converting remaining data to a numerical array

    # Converting the normalized_numerical DataFrame to a numerical array
    data_array = np.array(normalized_numerical)

    # Printing the shape (dimensions) of the data array
    print(data_array.shape)

    # Accessing and printing the first row of the data array
    print(data_array[0])

    return data_array

def preprocessing():
    dataset_paths = ["Data/07_17_2023_v1_GeneID_Query__SSTR1.csv"]
    preprocessed_data = []

    for path in dataset_paths:
        result_data = []

        # 2--> Loading data and preparing the dataset
        # Loading the dataset from a CSV file into a pandas DataFrame
        df = pd.read_csv(path)
        df.drop_duplicates(inplace=True)
        df = df.set_index("REM_ID")

        y = df['Predicted_function']  # label vector 0=No Attack, 1=Attack
        # y_cat = df['attack_cat']  # Label vector with attack categories or "Normal" for no attack
        X = df.drop(columns=['Predicted_function'])  # Dataset without labels

        # Creating a DataFrame 'df_cat' containing only the categorical attributes
        df_cat = X.select_dtypes(exclude=["number"])  # Categorical attributes only

        # Creating a list 'df_num' of column names for the numeric attributes
        df_num = X.select_dtypes(include=["number"]).columns.tolist()  # Numeric attributes only

        # Print all Attack Categories
        """    all_cat = []
        for i in y_cat:
            if i not in all_cat:
                print(i)
                all_cat.append(i)"""

        data_array = df_to_normalized_ndarray(df_num, df)
        encoded_data_cat = categorical_data_to_numerical_encoding(df_cat)

        # 6--> Concatenating encoded categorical data with numerical data array

        # Concatenate encoded categorical data with numerical data array
        concatenated_data = np.concatenate((encoded_data_cat, data_array), axis=1)
        # Print the shape of the concatenated data
        print("Shape of concatenated data:", concatenated_data.shape)

        # Print the first row of the concatenated data
        print("First row of concatenated data:", concatenated_data[0])

        result_data.append(concatenated_data)
        # result_data.append(y_cat)
        # result_data.append(all_cat)
        preprocessed_data.append(result_data)

    return preprocessed_data


if __name__ == '__main__':
    preprocessing()