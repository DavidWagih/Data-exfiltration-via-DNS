import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def corr_matrix(df):
    corr_mat = df.corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_mat, mask=mask,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, annot_kws={"size":10},fmt=".2f")