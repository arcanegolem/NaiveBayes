from bayes import NaiveBayes

import pandas as pd

table = pd.read_excel("bayes.xlsx")

nb = NaiveBayes(initial=table)
print(nb.classify(['> 20000', 'Нет', 'Да', 'Да']))