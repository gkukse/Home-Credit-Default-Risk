import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
pd.plotting.register_matplotlib_converters()


cmap='rocket'


def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    df = df.reset_index(drop=True)

    fontsize = 10

    for feature in df.columns:

        if df[feature].dtype.name in ['object', 'bool']:
            pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            print(f'{feature}')

            df.boxplot(column=feature, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min():.2f} to {df[feature].max():.2f}', fontsize=fontsize)

            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}', fontsize=fontsize)

            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature}', fontsize=fontsize)

            plt.tight_layout()
            plt.show()


def heatmap(df: pd.DataFrame, name: str, method: str) -> None:
    """ Plotting the heatmap of correlation matrix """
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f',
                vmin=-1, vmax=1, mask=mask)
    plt.title(f'Correlation {name.capitalize()} Attributes')
    plt.show()