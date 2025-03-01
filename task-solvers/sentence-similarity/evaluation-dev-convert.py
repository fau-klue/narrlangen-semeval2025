from collections import defaultdict
from pandas import DataFrame

lang = None
segmentation = None
model = None

results = defaultdict(dict)
with open("task-solvers/sentence-similarity/evaluation-dev-results.txt", "rt") as f:
    for line in f:
        if line.startswith("="):
            lang = line.split(" ")[1]
        if line.startswith("--"):
            model = line.split(" ")[1][:-1]
            segmentation = line.split(" ")[2].rstrip()
        if line.startswith("F1@coarse"):
            results[(lang, model, segmentation)]['f1 coarse'] = float(line.split(" ")[1])
            results[(lang, model, segmentation)]['sd coarse'] = float(line.split("(")[-1].rstrip().rstrip(")"))
        if line.startswith("F1@fine"):
            results[(lang, model, segmentation)]['f1 fine'] = float(line.split(" ")[1])
            results[(lang, model, segmentation)]['sd fine'] = float(line.split("(")[-1].rstrip().rstrip(")"))


df_results = DataFrame(results).T
df_results.to_csv("task-solvers/sentence-similarity/evaluation-dev-results.tsv", sep="\t")
df_results.to_latex("task-solvers/sentence-similarity/evaluation-dev-results.tex", float_format="%.2f")
