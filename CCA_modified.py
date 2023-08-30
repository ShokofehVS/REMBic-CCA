import time
import numpy as np
import random as rd
import pandas as pd
import sklearn.metrics as metrics
import evaluation
import preprocessing
import argparse
from CCA import ChengChurchAlgorithm
from iofile import _biclustering_to_dict, _dict_to_biclustering
from prelic import prelic_relevance, prelic_recovery
from subspace import clustering_error, relative_non_intersecting_area


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

        return sample_data, sample_labels


def get_distribution_of_sample(sample_labels):

    labels = {"activating": [], "repressing": []}

    labels_binary = {"activating": 0, "repressing": 0}
    for label in labels:
        count = sample_labels["Predicted_function"].value_counts().get(label, 0)
        labels[label] = count

        if label == "activating":
            labels_binary["activating"] = count
        else:
            labels_binary["repressing"] += count

    return labels, labels_binary

def run_cca(sample_data):
    # Run CCA on the sample data
    # missing value imputation suggested by Cheng and Church
    missing_sample = np.where(sample_data < 0.0)
    sample_data[missing_sample] = np.random.randint(low=0, high=800, size=len(missing_sample[0]))

    min_value = np.min(sample_data)
    max_value = np.max(sample_data)
    msr_thr = (((max_value - min_value) ** 2) / 12) * 0.005

    # creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
    cca = ChengChurchAlgorithm(num_biclusters=100)
    biclustering_test  = cca.run(sample_data)

    return biclustering_test, msr_thr


def format_cca_results(biclustering_test, all_cat, sample_labels):
    biclusters_as_dict = _biclustering_to_dict(biclustering_test)
    results = []
    # Evaluating data having labels
    for bicluster in biclusters_as_dict["biclusters"]:
        bicluster_values = {}
        for cat in all_cat:
            bicluster_values[cat] = 0

        for row in bicluster[0]:
            bicluster_values[sample_labels.loc[row]["Predicted_function"]] += 1

        results.append(bicluster_values)

    # Evaluating data with coherence measures
    for bicluster in biclusters_as_dict["biclusters"]:
        # np.array(bicluster)
        results.append(bicluster)

    return results

def write_results_in_file(number_of_runs, sample_size, msr_thr, multiple_node_deletion_tresh, data_min_col,
                          number_of_biclusters, biclustering):

    # Save biclustering results to a file
    biclustering_dict = _biclustering_to_dict(biclustering)
    biclustering_list = biclustering_dict["biclusters"]
    with open(
            f'CCA_modified_{number_of_runs}_{sample_size}_{round(msr_thr)}_{multiple_node_deletion_tresh}_{data_min_col}_'
            f'{number_of_biclusters}.out', 'w') as saveFile:
        for bicluster in biclustering_list:
            saveFile.write(str(bicluster))
            saveFile.write("\n")

def get_sample_binary_labels(sample_labels):
    # Convert multi-class labels to binary labels (activating/repressing)
    sample_labels_binary = sample_labels.copy()
    for index, row in sample_labels_binary.iterrows():
        if row['Predicted_function'] != "activating":
            sample_labels_binary.at[index, 'Predicted_function'] = "repressing"

    return sample_labels_binary

def format_results_for_f1(biclustering_test, all_cat, sample_labels):
    # Formate Bicluster result, for easier calculation of f1-score
    biclusters_as_dict = _biclustering_to_dict(biclustering_test)
    results = []
    classified_rows = set()
    number_of_bicl = 0

    for bicluster in biclusters_as_dict["biclusters"]:
        number_of_bicl += 1
        bicluster_values = {}
        for cat in all_cat:
            # Initialize dictionary to later count data of each predicted function in this bicluster.
            bicluster_values[cat] = 0

        for row in bicluster[0]:
            # Count and accumulate occurrences of each attack category
            classified_rows.add(row)
            bicluster_values[sample_labels.loc[row]["Predicted_function"]] += 1

        results.append(bicluster_values)

    return results, len(classified_rows)

def calc_f1_multi_classification(results, true_labels):
    purity_numerator = 0
    amount_of_data_biclustering = 0
    avg_precision, avg_recall, avg_f1, avg_acc = 0, 0, 0, 0

    for i, result in enumerate(results):
        count_labels = 0
        # Count occurence of each label in the bicluster
        for label in result:
            count_labels += result[label]

        # Determine most featured label
        max_key = max(result, key=result.get)
        # max_value is equal to the definition of true positives
        max_value = result[max_key]
        sum_values = sum(result.values())

        amount_of_data_biclustering += sum_values
        true_negatives = 0
        acc_denumerator = 0
        for label in true_labels:
            # Iterate through real labels of the data
            if label == max_key:
                # True positives and false negatives added to acc_denumerator.
                acc_denumerator += true_labels[label]
            else:
                # True Negatives and false positives added to acc_denumerator and true negatives
                acc_denumerator += true_labels[label]
                true_negatives += true_labels[label]

        # Subtract false positives from true negatives to get real value.
        true_negatives = true_negatives - (count_labels - max_value)

        precision = (max_value / sum_values)
        recall = max_value / true_labels[max_key]
        accuracy = (max_value + true_negatives) / acc_denumerator
        f1_score = (2 * recall * precision) / (recall + precision)

        purity_numerator += result[max_key]
        avg_acc += accuracy
        avg_f1 += f1_score
        avg_recall += recall
        avg_precision += precision
        number_of_bicluster = i + 1

    avg_precision = avg_precision / number_of_bicluster
    avg_recall = avg_recall / number_of_bicluster
    avg_f1 = avg_f1 / number_of_bicluster
    avg_acc = avg_acc / number_of_bicluster
    purity_multi = purity_numerator / amount_of_data_biclustering

    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1_Score: {avg_f1}")
    print(f"Average Accuracy: {avg_acc}")
    print("Purity Multiclassification:", purity_multi)

    return purity_multi, avg_precision, avg_recall, avg_f1, avg_acc

def calc_f1_binary_classification(results, true_labels):
    # Code is really similar to calculation of f1 on multi classification.
    binary_results = []
    avg_precision, avg_recall, avg_f1, avg_acc = 0, 0, 0, 0

    for result in results:
        # Format results into binary results.
        binary_result = {
            "activating": result.get("activating", 0),
            "repressing": sum(value for key, value in result.items() if key != "activating")
        }
        binary_results.append(binary_result)

    purity_numerator_binary = 0
    amount_of_data_biclustering = 0

    for j, binary_result in enumerate(binary_results):
        count_labels = 0
        # Count occurence of each label in the bicluster
        for label in binary_result:
            count_labels += binary_result[label]

        max_key_binary = max(binary_result, key=binary_result.get)
        max_value_binary = binary_result[max_key_binary]
        sum_values_binary = sum(binary_result.values())

        purity_numerator_binary += binary_result[max_key_binary]
        amount_of_data_biclustering += sum_values_binary

        true_negatives = 0
        overall_amount = 0
        for label in true_labels:
            if label == max_key_binary:
                overall_amount += true_labels[label]
            else:
                overall_amount += true_labels[label]
                true_negatives += true_labels[label]
        true_negatives = true_negatives - (count_labels - max_value_binary)

        # Calculating accuracy denumerator is easier for binary classification
        accuracy = (max_value_binary + true_negatives) / (true_labels["activating"] + true_labels["repressing"])
        recall = max_value_binary / true_labels[max_key_binary]
        precision = max_value_binary / (binary_result["activating"] + binary_result["repressing"])
        f1_score = (2 * recall * precision) / (recall + precision)

        avg_acc += accuracy
        avg_f1 += f1_score
        avg_recall += recall
        avg_precision += precision
        number_of_bicluster = j + 1

    avg_precision = avg_precision / number_of_bicluster
    avg_recall = avg_recall / number_of_bicluster
    avg_f1 = avg_f1 / number_of_bicluster
    avg_acc = avg_acc / number_of_bicluster
    purity_binary = purity_numerator_binary / amount_of_data_biclustering

    print("Purity binary:", purity_binary)
    print(f"Average Precision Binary: {avg_precision}")
    print(f"Average Recall Binary: {avg_recall}")
    print(f"Average F1_Score Binary: {avg_f1}")
    print(f"Average Accuracy Binary: {avg_acc}")

    return purity_binary, avg_precision, avg_recall, avg_f1, avg_acc

def format_results_for_eval(biclustering_test, sample_size):
    # Format the results so that we have list with a category for each row. The different bicluster mark the
    # different category's. This is necessary to calculate the metrics with sklearn.
    biclusters_as_dict = _biclustering_to_dict(biclustering_test)

    label_pred = [0 for i in range(sample_size)]

    for count, bicluster in enumerate(biclusters_as_dict["biclusters"]):
        rows = bicluster[0]
        for row in rows:
            label_pred[row] = count + 1

    not_class_data = []
    label_pred_without_none = []

    for count, row in enumerate(label_pred):
        if row == 0:
            not_class_data.append(count)
        else:
            label_pred_without_none.append(row)

    return label_pred_without_none, not_class_data

def format_true_labels_for_eval(sample_labels, not_class_data, sample_labels_binary):
    # Create list where each element represents the category the row had in the original data.
    true_labels = [row['Predicted_function'] for index, row in sample_labels.iterrows()]
    true_labels_binary = [row['Predicted_function'] for index, row in sample_labels_binary.iterrows()]

    true_labels_without_none = []
    for index, value in enumerate(true_labels):
        if index not in not_class_data:
            true_labels_without_none.append(value)

    true_labels_without_none_binary = []
    for index_b, value_b in enumerate(true_labels_binary):
        if index_b not in not_class_data:
            true_labels_without_none_binary.append(value_b)

    return true_labels_without_none, true_labels_without_none_binary

def calulate_eval_without_reference_bicl(true_labels, pred_labels, true_labels_binary):
    # Calculate popular metrics from sklearn.
    rand_ind = metrics.rand_score(true_labels, pred_labels)
    rand_ind_adj = metrics.adjusted_rand_score(true_labels, pred_labels)
    v_measure = metrics.v_measure_score(true_labels, pred_labels)

    rand_ind_bin = metrics.rand_score(true_labels_binary, pred_labels)
    rand_ind_adj_bin = metrics.adjusted_rand_score(true_labels_binary, pred_labels)
    v_measure_bin = metrics.v_measure_score(true_labels_binary, pred_labels)

    print(f"Random_ind: {rand_ind}")
    print(f"Random_ind_adj: {rand_ind_adj}")
    print(f"V_Measure: {v_measure}")

    print(f"Random_ind Binary: {rand_ind_bin}")
    print(f"Random_ind_adj Binary: {rand_ind_adj_bin}")
    print(f"V_Measure Binary: {v_measure_bin}")

    return rand_ind, rand_ind_adj, v_measure, rand_ind_bin, rand_ind_adj_bin, v_measure_bin

def create_reference_biclustering(sample_labels, sample_labels_binary):
    # Create reference Biclustering for the evaluation methods designed for comparing biclusters. Reference Biclustering
    # consists of 10 Bicluster with each just featuring data from a single category.

    true_labels = [row['Predicted_function'] for index, row in sample_labels.iterrows()]
    biclusters_dict = {'__class__': 'Biclustering', '__module__': 'models', 'biclusters': []}
    label_values = ['activating', 'repressing']

    columns = [x for x in range(194)]
    for label_value in label_values:
        rows = []
        for index, row in enumerate(true_labels):
            if row == label_value:
                rows.append(index)
        biclusters_dict['biclusters'].append((rows, columns))

    true_label_bicluster = _dict_to_biclustering(biclusters_dict)

    true_labels_binary = [row['Predicted_function'] for index, row in sample_labels_binary.iterrows()]
    biclusters_dict_binary = {'__class__': 'Biclustering', '__module__': 'models',
                       'biclusters': []}
    label_values_binary = ['activating', 'repressing']

    columns = [x for x in range(194)]
    for label_value in label_values_binary:
        rows = []
        for index, row in enumerate(true_labels_binary):
            if row == label_value:
                rows.append(index)
        biclusters_dict_binary['biclusters'].append((rows, columns))

    true_label_bicluster_binary = _dict_to_biclustering(biclusters_dict_binary)

    return true_label_bicluster, true_label_bicluster_binary

def calculate_eval_with_reference_bicl(pred_bicluster, true_bicluster, true_bicluster_binary):
    # Calculate metrics to compare biclustering to "perfect" biclustering.

    prelic_rel = prelic_relevance(pred_bicluster, true_bicluster)
    prelic_rec = prelic_recovery(pred_bicluster, true_bicluster)
    ce = clustering_error(pred_bicluster, true_bicluster, 40000, 194)
    rnia = relative_non_intersecting_area(pred_bicluster, true_bicluster, 40000, 194)

    prelic_rel_bin = prelic_relevance(pred_bicluster, true_bicluster_binary)
    prelic_rec_bin = prelic_recovery(pred_bicluster, true_bicluster_binary)
    ce_bin = clustering_error(pred_bicluster, true_bicluster_binary, 40000, 194)
    rnia_bin = relative_non_intersecting_area(pred_bicluster, true_bicluster_binary, 40000, 194)

    print(f"prelic_rel: {prelic_rel}")
    print(f"prelic_rec: {prelic_rec}")
    print(f"clustering error: {ce}")
    print(f"rnia: {rnia}")
    print(f"prelic_rel Binary: {prelic_rel_bin}")
    print(f"prelic_rec Binary: {prelic_rec_bin}")
    print(f"clustering error Binary: {ce_bin}")
    print(f"rnia Binary: {rnia_bin}")

    return prelic_rel, prelic_rec, ce, rnia, prelic_rel_bin, prelic_rec_bin, ce_bin, rnia_bin

def update_results_dict_for_visualization(results_dict, metrics_list, msr_tresh, runtime_av, numb_class_data_sum,
                                          sample_size, number_of_runs):
    results_dict["rand_index"].append(metrics_list[1])
    results_dict["F-measure"].append(metrics_list[4])
    results_dict["Precision"].append(metrics_list[5])
    results_dict["Recall"].append(metrics_list[6])
    results_dict["Accuracy"].append(metrics_list[7])
    results_dict["runtime"].append(runtime_av)
    results_dict["coverage"].append((numb_class_data_sum / sample_size))
    results_dict["number_of_runs"].append(number_of_runs)

def initialize_results_dict():
    return {
        "rand_index": [],
        "F-measure": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": [],
        "runtime": [],
        "coverage": [],
        "number_of_runs": [],
    }

def main(sample_size=None):
    # Parameters initialization
    mult_purity_av, bin_purity_av, mult_rand_ind_av, mult_rand_ind_adj_av, mult_v_measure_av, mult_prelic_rel_av, \
        mult_prelic_rec_av, mult_ce_av, mult_rnia_av, mult_prec_av, mult_rec_av, mult_f1_av, mult_acc_av \
        = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    bin_rand_ind_av, bin_rand_ind_adj_av, bin_v_measure_av, bin_prelic_rel_av, bin_prelic_rec_av, bin_ce_av, \
        bin_rnia_av, bin_prec_av, bin_rec_av, bin_f1_av, bin_acc_av = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    runtime_av, numb_class_data_sum = 0, 0

    number_of_biclusters, data_min_col, multiple_node_deletion_tresh, msr_thresh, number_of_runs = 200, 200, 1.2,\
        "default", 10

    # Preprocess EpiRego dataset
    data = preprocessing.preprocessing()

    # for visualization purposes
    results_dict_for_visualization_mult = initialize_results_dict()
    results_dict_for_visualization_bin = initialize_results_dict()

    # Get data as a source, label and all categorical data
    test_data = data[0][0]
    y_cat_test = data[0][1]
    all_cat = data[0][2]

    # Do transformation according to CCA original paper
    test_data = logarithmic_transformation(test_data)

    for j in range(number_of_runs):
        # Get part of data for large datasets
        sample_data, sample_labels = get_sample(test_data, y_cat_test, sample_size=sample_size)
        sample_labels_binary = get_sample_binary_labels(sample_labels)

        # Distribution of data according to label
        labels_distribution, labels_distribution_binary = get_distribution_of_sample(sample_labels)

        start = time.time()

        # Run CCA algorithm
        biclustering, msr_thr = run_cca(test_data)

        runtime = time.time() - start
        runtime_av += runtime

        # Format results before evaluations
        predicted_labels, not_class_data = format_results_for_eval(biclustering, sample_size)
        true_labels, true_labels_binary = format_true_labels_for_eval(sample_labels, not_class_data,
                                                                      sample_labels_binary)

        # Evaluate biclusters based on metrics
        mult_rand_ind, mult_rand_ind_adj, mult_v_measure, rand_ind_bin, rand_ind_adj_bin, v_measure_bin = \
            calulate_eval_without_reference_bicl(true_labels, predicted_labels, true_labels_binary)

        true_labels_bicluster_format, true_labels_binary_bicluster_format = create_reference_biclustering(
            sample_labels, sample_labels_binary)

        mult_prelic_rel, mult_prelic_rec, mult_ce, mult_rnia, prelic_rel_bin, prelic_rec_bin, ce_bin, rnia_bin = \
            calculate_eval_with_reference_bicl(biclustering, true_labels_bicluster_format,
                                               true_labels_binary_bicluster_format)
        # Format results for f1-measure
        results_f1_format, number_of_classified_data = format_results_for_f1(biclustering, all_cat, sample_labels)

        # Evaluate based on f1-measure
        mult_purity, mult_prec, mult_rec, mult_f1, mult_acc = calc_f1_multi_classification(results_f1_format,
                                                                                           labels_distribution)

        bin_purity, bin_prec, bin_rec, bin_f1, bin_acc = calc_f1_binary_classification(results_f1_format,
                                                                                       labels_distribution_binary)
        # Results of evaluations
        mult_purity_av += mult_purity
        mult_rand_ind_av += mult_rand_ind
        mult_rand_ind_adj_av += mult_rand_ind_adj
        mult_v_measure_av += mult_v_measure
        mult_prec_av += mult_prec
        mult_rec_av += mult_rec
        mult_f1_av += mult_f1
        mult_acc_av += mult_acc
        mult_prelic_rel_av += mult_prelic_rel
        mult_prelic_rec_av += mult_prelic_rec
        mult_ce_av += mult_ce
        mult_rnia_av += mult_rnia

        bin_purity_av += bin_purity
        bin_rand_ind_av += rand_ind_bin
        bin_rand_ind_adj_av += rand_ind_adj_bin
        bin_v_measure_av += v_measure_bin
        bin_prec_av += bin_prec
        bin_rec_av += bin_rec
        bin_f1_av += bin_f1
        bin_acc_av += bin_acc
        bin_prelic_rel_av += prelic_rel_bin
        bin_prelic_rec_av += prelic_rec_bin
        bin_ce_av += ce_bin
        bin_rnia_av += rnia_bin
        numb_class_data_sum += number_of_classified_data

        update_results_dict_for_visualization(results_dict_for_visualization_bin, [
            bin_purity_av, rand_ind_bin, bin_rand_ind_adj_av, bin_v_measure_av, bin_f1, bin_prec,
            bin_rec, bin_acc, bin_prelic_rel_av, bin_prelic_rec_av, bin_ce_av, bin_rnia_av], msr_thresh, runtime_av,
                                              number_of_classified_data, sample_size, j)
        df_bin = pd.DataFrame(results_dict_for_visualization_bin)
        df_bin.to_csv(f"Data/Results_for_visualization_binary.csv", index=False)

    runtime_av /= number_of_runs
    numb_class_data_sum /= number_of_runs

    # Write the result on file
    write_results_in_file(number_of_runs, sample_size, msr_thr, multiple_node_deletion_tresh, data_min_col,
                          number_of_biclusters, biclustering)


if __name__ == '__main__':
    # Using argparse to specify the sample_size parameter when running the code
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_size", type=int, nargs='?', default=None, help="Sample size for CCA")
    args = parser.parse_args()

    main(sample_size=71)
