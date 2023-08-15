import numpy as np
import random as rd
import preprocessing
import argparse
from CCA import ChengChurchAlgorithm
from iofile import _biclustering_to_dict


def logarithmic_transformation(data_test):
    # Do Logaritmic transformations on dataset that are suggested in the paper
    nonzero_mask = data_test != 0
    data_test_log_norm = np.zeros_like(data_test)
    data_test_log_norm[nonzero_mask] = np.log10(data_test[nonzero_mask] * (10 ** 5)) * 100

    return data_test_log_norm


def get_sample(data_test, y_cat, sample_size=None):

    if not sample_size:
        y_cat = y_cat.reset_index()
        return data_test, y_cat
    else:
        # Define sample of data to use CCA on smaller dataset.
        num_rows = len(data_test)
        random_indices = rd.sample(range(num_rows), sample_size)
        sample_data = data_test[random_indices]
        sample_labels = y_cat.iloc[random_indices]
        sample_labels = sample_labels.reset_index()
        print(sample_labels)
        return sample_data, sample_labels


def get_distribution_of_sample(sample_labels):

    labels = {"activating": [], "repressing": []}

    for label in labels:
        count = sample_labels["Predicted_function"].value_counts().get(label, 0)
        labels[label] = count


def run_cca(sample_data):
    # Run CCA on the sample data
    # missing value imputation suggested by Cheng and Church
    missing_sample = np.where(sample_data < 0.0)
    sample_data[missing_sample] = np.random.randint(low=0, high=800, size=len(missing_sample[0]))

    min_value = np.min(sample_data)
    max_value = np.max(sample_data)

    msr_thr = (((max_value - min_value) ** 2) / 12) * 0.005
    print("The used threshold for the msr is:", msr_thr)

    print("Initializing CCA")

    # creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
    cca = ChengChurchAlgorithm(num_biclusters=100)

    print("Starting Algorithm")
    biclustering_test = cca.run(sample_data)
    print(biclustering_test)

    return biclustering_test


def format_cca_results(biclustering_test, all_cat, sample_labels):
    biclusters_as_dict = _biclustering_to_dict(biclustering_test)
    results = []
    # Evaluating data having labels
    """for bicluster in biclusters_as_dict["biclusters"]:
        bicluster_values = {}
        for cat in all_cat:
            bicluster_values[cat] = 0

        for row in bicluster[0]:
            bicluster_values[sample_labels.loc[row]["Predicted_function"]] += 1

        results.append(bicluster_values)"""

    # Evaluating data with coherence measures
    for bicluster in biclusters_as_dict["biclusters"]:
        # np.array(bicluster)
        results.append(np.array(bicluster[0]))
        results.append(np.array(bicluster[1]))


    return results

def main(sample_size=None):
    # Preprocessing EpiRego dataset
    data = preprocessing.preprocessing()

    # Getting data as a source, label and all categorical data
    test_data = data[0][0]
    y_cat_test = data[0][1]
    all_cat_test = data[0][2]

    # Do transformation according to CCA original paper
    test_data = logarithmic_transformation(test_data)

    # Get part of data for large datasets
    sample_data, sample_labels = get_sample(test_data, y_cat_test, sample_size=sample_size)

    # Distribution of data according to label
    get_distribution_of_sample(sample_labels)

    # Run CCA algorithm
    biclustering_test = run_cca(test_data)

    # Format the resulting biclusters
    results = format_cca_results(biclustering_test, all_cat_test, sample_labels)


if __name__ == '__main__':
    # Using argparse to specify the sample_size parameter when running the code
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_size", type=int, nargs='?', default=None, help="Sample size for CCA")
    args = parser.parse_args()

    main(sample_size=None)
