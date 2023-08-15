import gseapy
import numpy as np
import warnings
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')

# Methods for using further in evaluation
def _means(bicluster_data):
    mean = np.mean(bicluster_data)
    row_means = np.mean(bicluster_data, axis=1)
    col_means = np.mean(bicluster_data, axis=0)
    return mean, row_means, col_means

def _abs_corrcoef(X):
    c = np.ma.corrcoef(X)
    return np.ma.abs(c)

def _spearman(X):
    s, _ = spearmanr(X, axis=1)
    return s if isinstance(s, float) else np.ma.array(s, mask=np.isnan(s))

def _abs_spearman(X):
    return np.ma.abs(_spearman(X))

def _mean_residue(bicluster_data, func=lambda x: x ** 2):
    mean, row_means, col_means = _means(bicluster_data)
    residues = func(bicluster_data - row_means[:, np.newaxis] - col_means + mean)
    return np.mean(residues)

def isclose(a, b, tol=1e-12):
    return abs(a - b) <= tol

# *** START OF EVALUATION MEASURES ***
# Direct implementation from BiCon "https://github.com/biomedbigdata/BiCoN" and needs modification
def jaccard_index(self, true_labels):
    def jac(x, y):
        if len(x) > 0 and len(y) > 0:
            return len(set(x).intersection(set(y))) / len((set(x).union(set(y))))
        else:
            return (0)

    def jac_matrix(true, pred):
        res = np.zeros((len(true), len(true)))
        for i in range(len(true)):
            for j in range(len(true)):
                res[i, j] = jac(true[i], pred[j])
        cand1 = (res[0][0], res[1][1])
        cand2 = (res[0][1], res[1][0])
        if sum(cand1) > sum(cand2):
            return (cand1)
        else:
            return (cand2)

    ids = jac_matrix([self.patients1, self.patients2], true_labels)

    print("Jaccard indices for two groups are {0} and {1}".format(round(ids[0], 2), round(ids[1], 2)))
    # actually return the ids aswell
    return (round(ids[0], 2), round(ids[1], 2))


# Direct implementation from BiCon "https://github.com/biomedbigdata/BiCoN" and needs modification
def enrichment_analysis(self, library, output):
    """
    Saves the results of enrichment analysis

    Attributes:
    -----------
    library - Enrichr library to be used. Recommendations:
        - 'GO_Molecular_Function_2018'
        - 'GO_Biological_Process_2018'
        - 'GO_Cellular_Component_2018'
        for more options check available libraries by typing gseapy.get_library_name()

    output - directory name where results should be saved
    """
    libs = gseapy.get_library_name()
    assert library in libs, "the library is not available, check gseapy.get_library_name() for available options"
    assert (self.convert == True) or (
            self.origID == "symbol"), "EnrichR accepts only gene names as an input, thus please set 'convert' to True and indicate the original gene ID"

    genes1_name = [self.mapping[x] for x in self.genes1]
    genes2_name = [self.mapping[x] for x in self.genes2]
    all_genes_names = genes1_name + genes2_name
    res = gseapy.enrichr(gene_list=all_genes_names, description='pathway', gene_sets=library, cutoff=0.05,
                         outdir=output)
    return (res.results)

# Evaluating data having labels
def calc_multi_classification(results):
    information = {}

    precision_numerator = 0
    precision_denominator = 0
    max_keys = set()

    for i, result in enumerate(results):
        max_key = max(result, key=result.get)
        max_keys.add(max_key)
        max_value = result[max_key]
        sum_values = sum(result.values())
        percent = (max_value / sum_values) * 100

        information[f"Bicluster {i + 1}"] = {
            "Sum": sum_values,
            "Max Key": max_key,
            "Percentage": percent
        }

        precision_numerator += result[max_key]
        precision_denominator += sum_values

    accuracy = precision_numerator / precision_denominator

    print("Accuracy Multiclassification:", accuracy)

    return accuracy

# Evaluating data having labels
def calculate_binary_classification(results):
    binary_results = []

    for result in results:
        binary_result = {
            "Normal": result.get("Normal", 0),
            "Attack": sum(value for key, value in result.items() if key != "Normal")
        }
        binary_results.append(binary_result)

    precision_numerator_binary = 0
    precision_denominator_binary = 0

    for j, binary_result in enumerate(binary_results):
        max_key_binary = max(binary_result, key=binary_result.get)
        max_value_binary = binary_result[max_key_binary]
        sum_values_binary = sum(binary_result.values())

        precision_numerator_binary += binary_result[max_key_binary]
        precision_denominator_binary += sum_values_binary

    accuracy_binary = precision_numerator_binary / precision_denominator_binary

    print("Accuracy binary:", accuracy_binary)

    return accuracy_binary

# The rest of metrics are direct implementations from https://padilha.github.io/bracis-2018-suppl/
# Variance (VAR)
def var(bicluster_data):
    mean = np.mean(bicluster_data)
    VAR = np.sum((bicluster_data - mean) ** 2)
    assert VAR >= 0.0
    return VAR

# Mean Squared Residue (MSR)
def msr(bicluster_data):
    MSR = _mean_residue(bicluster_data)
    assert MSR >= 0.0
    return MSR

# Mean Absolute Residue (MAR)
def mar(bicluster_data):
    MAR = _mean_residue(bicluster_data, func=np.abs)
    assert MAR >= 0.0
    return MAR

# Scaling Mean Squared Residue (SMSR)
def smsr(bicluster_data):
    mean, row_means, col_means = _means(bicluster_data)
    scaling_residues = ((np.outer(row_means, col_means) - bicluster_data * mean) ** 2) / np.outer(row_means ** 2,
                                                                                                  col_means ** 2)
    SMSR = np.mean(scaling_residues)
    assert SMSR >= 0.0
    return SMSR

# Minimal Mean Squared Error (MMSE)
def mmse(bicluster_data):
    row_means = np.mean(bicluster_data, axis=1)
    B = bicluster_data - row_means[:, np.newaxis]
    n, m = B.shape
    S = np.dot(B, B.T) if n < m else np.dot(B.T, B)
    abs_eigvals = np.abs(np.linalg.eigvals(S))
    MMSE = (np.sum(B ** 2) - np.max(abs_eigvals)) / (n * m)
    assert MMSE >= 0.0 or isclose(MMSE, 0.0)
    return MMSE

# Overall Constancy (OC)
def oc(bicluster_data):
    def constancy(bicluster_data):
        dist = pdist(bicluster_data, metric='euclidean')
        return np.mean(dist)

    n, m = bicluster_data.shape
    Cr = constancy(bicluster_data)
    Cc = constancy(bicluster_data.T)
    OC = (n * Cr + m * Cc) / (n + m)
    assert OC >= 0.0
    return OC

# Relevance Index (RI)
def ri(bicluster_data, bicluster_col_global_data):
    assert bicluster_data.shape[1] == bicluster_col_global_data.shape[1]
    global_var = np.var(bicluster_col_global_data, axis=0, ddof=1)
    local_var = np.var(bicluster_data, axis=0, ddof=1)
    relevance = 1.0 - local_var / global_var
    return np.sum(relevance)

# Average Correlation (AC)
def ac(bicluster_data, corr=np.ma.corrcoef):
    n, m = bicluster_data.shape
    c = corr(bicluster_data)

    if isinstance(c, float):  # needed if the bicluster has only 2 rows
        return c

    diag = np.ma.diag(c)
    AC = (np.ma.sum(c) - np.ma.sum(diag)) / (n ** 2 - n)
    assert -1.0 <= AC <= 1.0 or isclose(AC, -1.0) or isclose(AC, 1.0)
    return AC

# Sub-Matrix Correlation Score (SCS)
def scs(bicluster_data):
    def score(bicluster_data):
        abs_corr = _abs_corrcoef(bicluster_data)

        if isinstance(abs_corr, float):  # needed if the bicluster has only 2 rows
            return 1 - abs_corr

        n, _ = abs_corr.shape
        row_scores = 1 - (np.ma.sum(abs_corr, axis=1) - np.ma.diag(abs_corr)) / (n - 1)
        return np.min(row_scores)

    row_score = score(bicluster_data)
    col_score = score(bicluster_data.T)
    SCS = min(row_score, col_score)
    assert 0.0 <= SCS <= 1.0 or isclose(SCS, 0.0) or isclose(SCS, 1.0)
    return SCS

# Average Correlation Value (ACV)
def acv(bicluster_data, corr=_abs_corrcoef):
    avg_row_corr = ac(bicluster_data, corr=corr)
    avg_col_corr = ac(bicluster_data.T, corr=corr)
    ACV = max(avg_row_corr, avg_col_corr)
    assert -1.0 <= ACV <= 1.0 or isclose(ACV, -1.0) or isclose(ACV, 1.0)
    return ACV

# Average Spearman’s Rho (ASR)
def asr(bicluster_data):
    ASR = acv(bicluster_data, corr=_spearman)
    assert -1.0 <= ASR <= 1.0 or isclose(ASR, -1.0) or isclose(ASR, 1.0)
    return ASR

# Spearman’s Biclustering Measure (SBM)
def sbm(bicluster_data, full_data, alpha_thr=9, beta_reliability=1.0):
    avg_row_corr = ac(bicluster_data, corr=_abs_spearman)
    avg_col_corr = ac(bicluster_data.T, corr=_abs_spearman)

    n, m = bicluster_data.shape
    N, M = full_data.shape

    if m > alpha_thr:
        alpha_reliability = 1.0
    else:
        alpha_reliability = m / M

    SBM = alpha_reliability * avg_row_corr * beta_reliability * avg_col_corr
    assert SBM >= 0.0
    return SBM

# Maximal Standard Area (MSA)
def msa(bicluster_data):
    bicluster_data = scale(bicluster_data, axis=1)
    upper = np.max(bicluster_data, axis=0)
    lower = np.min(bicluster_data, axis=0)
    n, m = bicluster_data.shape
    MSA = sum(abs(upper[j] - lower[j] + upper[j + 1] - lower[j + 1]) / 2 for j in range(m - 1))
    assert MSA >= 0.0
    return MSA

# Virtual Error (VE)
def virtual_error(bicluster_data):
    virtual_row = scale(np.mean(bicluster_data, axis=0))
    bicluster_data = scale(bicluster_data, axis=1)
    VE = np.mean(np.abs(bicluster_data - virtual_row))
    assert VE >= 0.0
    return VE

# Transposed Virtual Error (VEt)
def transposed_virtual_error(bicluster_data):
    return virtual_error(bicluster_data.T)