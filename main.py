from bayes import NaiveBayes

import pandas as pd

table = pd.read_excel("bayes.xlsx")

nb = NaiveBayes(initial=table)
bayes_result = nb.classify(['< 10000', 'Да', 'Да', 'Да'])

print(bayes_result)

for class_v in bayes_result:
    print(f"{class_v} : {int(round(bayes_result[class_v], 2) * 100)}%")