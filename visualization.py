import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_multi_class():
    df = pd.read_csv("Results_for_visualization_multi_multiple_thr2")

    new_palette = sns.color_palette("husl", n_colors=len(df.columns))

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    line_columns = df.columns[1:]

    for i, col in enumerate(line_columns):
        if col == 'runtime' or col == 'rand_index' or col == 'V-measure' or col == 'relevance_match_score' or col == 'recovery_match_score' \
                or col == 'relative_non_intersecting_area':
            continue
        sns.lineplot(x="multiple_thr", y=col, data=df, label=col, marker='o', color=new_palette[i])

    plt.title("Values for changing the multiple node deletion treshold ")
    plt.xlabel("Multiple Node Deletion Treshold")
    plt.ylabel("Value")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('multi_multiple_thr.png', format='png')

    plt.show()


def visualize_binary_class():
    df = pd.read_csv(os.path.join(os.getcwd(), 'Results_for_visualization_binary.csv'), delimiter=',')
    new_palette = sns.color_palette("husl", n_colors=len(df.columns))

    sns.set(style="whitegrid")
    line_columns = df.columns[1:]

    for i, col in enumerate(df.columns[1:len(df.columns)-1]):
        if col == 'runtime' or col == 'rand_index' or col == 'V-measure' or col == 'relevance_match_score' or col == 'recovery_match_score' \
                or col == 'relative_non_intersecting_area':
            continue
        sns.lineplot(x="number_of_runs", y=col, data=df, label=col, marker='o', color=new_palette[i])

    plt.title("Results of Evaluation Measures over CCA")
    plt.xlabel("Number of Runs")
    plt.ylabel("Values")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('bin_num_runs.png', format='png')

    plt.show()


def main():
    """
    Reads the values of the different evaluation metrics from a csv file and creates a diagram to visualize the results.
    :return:
    """
    visualization = "bin"

    if visualization == "bin":
        visualize_binary_class()
    else:
        visualize_multi_class()

if __name__ == '__main__':
    main()
