# REMBic-CCA

 This project implements a powerful algorithm for identifying biclusters and co-activation patterns within biological datasets. CCA can help uncover hidden relationships and functional insights in complex biological data.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Biclustering or simultaneous clustering of both genes and conditions as a new paradigm was introduced by Cheng and Church's Algorithm (CCA). The concept of bicluster refers to a subset of genes and a subset of conditions with a high similarity score, which measures the coherence of the genes and conditions in the bicluster. It also returns the list of biclusters for the given data set.
## Features

- **Biclustering:** Discover patterns of co-activated genes across subsets of conditions.
- **Co-Activation Analysis:** Identify relationships and interactions among genes.
- **Evaluation Metrics:** Evaluate the algorithm's performance using precision, recall, F1-score, accuracy, and purity metrics.
- **Cross-Validation:** Utilize cross-validation for robust evaluation.

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:ShokofehVS/REMBic-CCA.git
   cd REMBic-CCA
   python CCA_modified.py