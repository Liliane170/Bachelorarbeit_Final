import pandas as pd
import json
import matplotlib.pyplot as plt
from torch.utils import data
from scipy.stats import shapiro
from scipy.stats import anderson
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats import skew
import seaborn as sns
from scipy.stats import kstest
from scipy.stats import binomtest
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


class Analyser:
    def __init__(self, result):

        data = result

        self.metric_keys = [
            "flesch_original", "gunning_original", "dale_chall_original",
            "flesch_generalized", "gunning_generalized", "dale_chall_generalized",
            "flesch_tagged", "gunning_tagged", "dale_chall_tagged",
            "flesch_supressed", "gunning_supressed", "dale_chall_supressed",
            "flesch_randomized", "gunning_randomized", "dale_chall_randomized"
        ]

        metric_data = [{key: item.get(key, None) for key in self.metric_keys} for item in data]
        
        self.df = pd.DataFrame(metric_data)

        self.df_full = pd.DataFrame(result)

        self.flesch = ["flesch_tagged", "flesch_generalized", "flesch_supressed", "flesch_randomized"]

        self.gunning = ["gunning_tagged", "gunning_generalized", "gunning_supressed", "gunning_randomized"]

        self.dale_chall = ["dale_chall_tagged", "dale_chall_generalized", "dale_chall_supressed", "dale_chall_randomized"]
        
        
    def showData_diff(self, data, category):
        
        df = data.round(3)
        
        a = f"flesch_diff_{category.lower()}"
        
        b = f"gunning_diff_{category.lower()}"
        
        c = f"dale_chall_diff_{category.lower()}"

           
        colors = {
            'Flesch': '#4682B4',   
            'Gunning': '#66C2A5',           
            'Dale Chall': '#FFD92F'      
        }

        markers = {
            'Flesch': 'o',
            'Gunning': 's',
            'Dale Chall': '^'
        }
       

        plt.figure(figsize=(10, 6))

        def scatter(col, label):
            y = df[col]  
            x = df.index  
            plt.scatter(
                x, y,
                label=label,
                color=colors[label],
                marker=markers[label],
                alpha=0.8,
                edgecolor='black',
                s=30
            )

        scatter(a, 'Flesch')
        scatter(b, 'Gunning')
        scatter(c, 'Dale Chall')
 

        if category == 'Supressed':
            title = 'Suppression'
        elif category == 'Randomized':
            title = 'Randomisierung'
        elif category == 'Generalized':
            title = 'Generalisierung'
        else:
            title = 'Tagging'

        plt.legend(frameon=True, loc='best')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel(f'Differenz von Original und {title}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def normalality_for_t_test(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                attribute = (b.rpartition("_")[0]) + "_original"

                data = self.df[attribute] - self.df[b]

                statistic, pvalue = stats.shapiro(data)

                print(f'{b}: {pvalue}')

                if (b.rpartition("_")[0]) == 'flesch':
                    title1 = 'Flesch'
                elif (b.rpartition("_")[0]) == 'gunning':
                    title1 = 'Gunning'
                else:
                    title1 = 'Dale Chall'

                if b.rsplit("_", 1)[1] == "tagged":
                    title2 = 'Tagging'
                elif b.rsplit("_", 1)[1] == "supressed":
                    title2 = 'Suppression'
                elif b.rsplit("_", 1)[1] == "randomized":
                    title2 = 'Randomisierung'
                else:
                    title2 = 'Generalisierung'

                # Q-Q-Plot
                fig, ax = plt.subplots(figsize=(8, 5))

                stats.probplot(data, dist="norm", plot=ax)

                ax.get_lines()[0].set_color("#FC8D62")  
                ax.get_lines()[1].set_color("black")        
                ax.get_lines()[0].set_marker("o")           
                ax.get_lines()[0].set_markersize(4)
                ax.set_title("")
                ax.set_xlabel("Theoretische Quantile", fontsize=12)
                ax.set_ylabel("Beobachtete Quantile", fontsize=12)
                plt.tight_layout()

                plt.savefig(f"qq_{b.rsplit("_", 1)[1]}_{b.rpartition("_")[0]}_diff.png", dpi=300)

                plt.show()

    def normalality_for_anova(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                data = self.df[b]

                statistic, pvalue = stats.shapiro(data)

                print(pvalue)
            

                if (b.rpartition("_")[0]) == 'flesch':
                    title1 = 'Flesch'
                elif (b.rpartition("_")[0]) == 'gunning':
                    title1 = 'Gunning'
                else:
                    title1 = 'Dale Chall'

                if b.rsplit("_", 1)[1] == "tagged":
                    title2 = 'Tagging'
                elif b.rsplit("_", 1)[1] == "supressed":
                    title2 = 'Suppression'
                elif b.rsplit("_", 1)[1] == "randomized":
                    title2 = 'Randomisierung'
                else:
                    title2 = 'Generalisierung'

                # Q-Q-Plot
                fig, ax = plt.subplots(figsize=(8, 5))

                stats.probplot(data, dist="norm", plot=ax)

                ax.get_lines()[0].set_color("#FC8D62")   
                ax.get_lines()[1].set_color("black")       
                ax.get_lines()[0].set_marker("o")          
                ax.get_lines()[0].set_markersize(4)
                ax.set_title("")
                ax.set_xlabel("Theoretische Quantile", fontsize=12)
                ax.set_ylabel("Beobachtete Quantile", fontsize=12)
                plt.tight_layout()

                plt.savefig(f"qq_{b.rsplit("_", 1)[1]}_{b.rpartition("_")[0]}.png", dpi=300)

                #plt.show()

       
        
    def diff(self):
        
        print(self.df.head(10))

        df_diff = self.df
        
        df_diff["flesch_diff_tagged"] = df_diff["flesch_original"] - df_diff["flesch_tagged"]
        df_diff["gunning_diff_tagged"] = df_diff["gunning_original"] - df_diff["gunning_tagged"]
        df_diff["dale_chall_diff_tagged"] = df_diff["dale_chall_original"] - df_diff["dale_chall_tagged"]
        
        df_diff_tagged = df_diff[["flesch_diff_tagged", "gunning_diff_tagged", "dale_chall_diff_tagged"]]
        
        df_diff["flesch_diff_generalized"] = df_diff["flesch_original"] - df_diff["flesch_generalized"]
        df_diff["gunning_diff_generalized"] = df_diff["gunning_original"] - df_diff["gunning_generalized"]
        df_diff["dale_chall_diff_generalized"] = df_diff["dale_chall_original"] - df_diff["dale_chall_generalized"]
        
        df_diff_generalized = df_diff[["flesch_diff_generalized", "gunning_diff_generalized", "dale_chall_diff_generalized"]]
        
        df_diff["flesch_diff_supressed"] = df_diff["flesch_original"] - df_diff["flesch_supressed"]
        df_diff["gunning_diff_supressed"] = df_diff["gunning_original"] - df_diff["gunning_supressed"]
        df_diff["dale_chall_diff_supressed"] = df_diff["dale_chall_original"] - df_diff["dale_chall_supressed"]
        
        df_diff_supressed = df_diff[["flesch_diff_supressed", "gunning_diff_supressed", "dale_chall_diff_supressed"]]
        
        df_diff["flesch_diff_randomized"] = df_diff["flesch_original"] - df_diff["flesch_randomized"]
        df_diff["gunning_diff_randomized"] = df_diff["gunning_original"] - df_diff["gunning_randomized"]
        df_diff["dale_chall_diff_randomized"] = df_diff["dale_chall_original"] - df_diff["dale_chall_randomized"]
        
        df_diff_randomized = df_diff[["flesch_diff_randomized", "gunning_diff_randomized", "dale_chall_diff_randomized"]]
        
        self.showData_diff(df_diff_tagged, 'Tagged')
        self.showData_diff(df_diff_generalized, 'Generalized')
        self.showData_diff(df_diff_supressed, 'Supressed')
        self.showData_diff(df_diff_randomized, 'Randomized')

    def compare_data(self):

        df_compare = self.df
     

        df_compare_gunning = df_compare[["gunning_tagged", "gunning_generalized", "gunning_supressed", "gunning_randomized"]]

        df_compare_flesch = df_compare[["flesch_tagged", "flesch_generalized", "flesch_supressed", "flesch_randomized"]]

        df_compare_dale_chall = df_compare[["dale_chall_tagged", "dale_chall_generalized", "dale_chall_supressed", "dale_chall_randomized"]]

        self.showData_compare(df_compare_gunning, 'Gunning')
        self.showData_compare(df_compare_flesch, 'Flesch')
        self.showData_compare(df_compare_dale_chall, 'Dale_Chall')


    def showData_compare(self, data, category):
        
        df = data.round(3)

        a = f"{category.lower()}_generalized"
        b = f"{category.lower()}_tagged"
        c = f"{category.lower()}_randomized"
        d = f"{category.lower()}_supressed"
  
        colors = {
            'Generalisierung': '#4682B4',   
            'Tagging': '#66C2A5',          
            'Randomisierung': '#FFD92F',    
            'Suppression': '#FC8D62'       
        }

        markers = {
            'Generalisierung': 'o',
            'Tagging': 's',
            'Randomisierung': '^',
            'Suppression': 'D'
        }

        plt.figure(figsize=(10, 6))

        def scatter_sorted(col, label):
            sorted_vals = df[col].sort_values().reset_index(drop=True)
            x = sorted_vals.index
            plt.scatter(
                x, sorted_vals,
                label=label,
                color=colors[label],
                marker=markers[label],
                alpha=0.8,
                edgecolor='black',
                s=30  
            )

        scatter_sorted(a, 'Generalisierung')
        scatter_sorted(b, 'Tagging')
        scatter_sorted(c, 'Randomisierung')
        scatter_sorted(d, 'Suppression')

        title = 'Dale Chall' if category == 'Dale_Chall' else category

        plt.legend(frameon=True, loc='best')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel(f'Lesbarkeit nach {title}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def friedman(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]


        for a in metrics:
        
            statistic, pvalue = friedmanchisquare(self.df[a[0]], self.df[a[1]], self.df[a[2]], self.df[a[3]])

            
            print(pvalue)

        return 0

    def sign_test(self):
       
        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                attribute = (b.rpartition("_")[0]) + "_original"

                data = self.df[attribute] - self.df[b]

                n_positive = np.sum(data > 0)
                n_negative = np.sum(data < 0)
                n = n_positive + n_negative

                print(b)
                print(n_positive)
                print(n_negative)

                res_less = binomtest(k=n_positive, n=n, p=0.5, alternative='less')
                res_greater = binomtest(k=n_positive, n=n, p=0.5, alternative='greater')
                print("p-Wert für Less:", res_less.pvalue)
                print("p-Wert für Greater:", res_greater.pvalue)

    def showOutliers(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                df_sortiert = self.df_full.sort_values(by=b)

                df_filterd = df_sortiert 

                df_filterd.to_json(f'{b}.json', orient='records', lines=True)

    def showOutliers_diff(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                attribute = (b.rpartition("_")[0]) + "_original"

                df_diff = self.df_full

                df_diff[f"diff_original_{b}"] = df_diff[attribute] - df_diff[b]

                df_sortiert = df_diff.sort_values(by=f"diff_original_{b}")

                df_filterd = df_sortiert 

                df_filterd.to_json(f'{b}.json', orient='records', lines=True)

    def posthoc(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]
        b = 1

        for a in metrics:
  
            df_filtered = self.df[a]


            # Nemenyi-Test
            nemenyi = sp.posthoc_nemenyi_friedman(df_filtered)

            print(nemenyi)

            filename = f"nemenyi_{b}.xlsx"
            nemenyi.to_excel(filename)

            b = b + 1

    def get_median(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                data = self.df[b]

                median = np.median(data)

                print(f"Median von {b}: {median}")

    def symmertry(self):

        metrics = [self.flesch, self.gunning, self.dale_chall]

        for a in metrics:

            for b in a:

                attribute = (b.rpartition("_")[0]) + "_original"

                data = self.df[attribute] - self.df[b]

                sk = skew(data)

                print(f'{b}: {sk}')

                if (b.rpartition("_")[0]) == 'flesch':
                    title1 = 'Flesch'
                elif (b.rpartition("_")[0]) == 'gunning':
                    title1 = 'Gunning'
                else:
                    title1 = 'Dale Chall'

                if b.rsplit("_", 1)[1] == "tagged":
                    title2 = 'Tagging'
                elif b.rsplit("_", 1)[1] == "supressed":
                    title2 = 'Suppression'
                elif b.rsplit("_", 1)[1] == "randomized":
                    title2 = 'Randomisierung'
                else:
                    title2 = 'Generalisierung'

                # Histogramm
                fig, ax = plt.subplots(figsize=(8, 5))

                ax.hist(data, bins=30, edgecolor='black', color="#FC8D62")

                ax.set_xlabel("Werte", fontsize=12)
                ax.set_ylabel("Häufigkeit", fontsize=12)

                plt.tight_layout()

                plt.savefig(f"hist_{b.rsplit('_', 1)[1]}_{b.rpartition('_')[0]}_diff.png", dpi=300)

                plt.show()

    def showOriginal(self):

        df_original = self.df
     
        a = "gunning_original"
        b = "dale_chall_original"
        c = "flesch_original"

        print(a, df_original[a].mean())
        print(b, df_original[b].mean())
        print(c, df_original[c].mean())
    
        colors = {
            'Gunning': '#4682B4',   
            'Dale Chall': '#66C2A5',           
            'Flesch': '#FFD92F'       
        }

        markers = {
            'Gunning': 'o',
            'Dale Chall': 's',
            'Flesch': '^'
        }

        plt.figure(figsize=(10, 6))

        def scatter_sorted(col, label):
            y = df_original[col]  
            x = df_original.index  
            plt.scatter(
                x, y,
                label=label,
                color=colors[label],
                marker=markers[label],
                alpha=0.8,
                edgecolor='black',
                s=30 
            )

        scatter_sorted(a, 'Gunning')
        scatter_sorted(b, 'Dale Chall')
        scatter_sorted(c, 'Flesch')

        plt.legend(frameon=True, loc='best')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel(f'Lesbarkeit des Originals nach verschiedenen Metriken', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
                

        