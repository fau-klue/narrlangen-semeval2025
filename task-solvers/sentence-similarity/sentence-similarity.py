import os
from argparse import ArgumentParser
from glob import glob
from pandas import concat, DataFrame, read_csv
from sentence_transformers import SentenceTransformer, util

# script to compare segments of documents to segments of subnarrative descriptions
# returns values for pairwise comparisons
# actual prediction via R script


if __name__ == '__main__':

    # ARGUMENTS
    parser = ArgumentParser()
    parser.add_argument("--glob_documents", default="task_data/train/*/raw-documents/*.txt")
    parser.add_argument("--glob_narratives", default="Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv")
    parser.add_argument("--path_out", default="task-solvers/sentence-similarity/results/paragraph-paragraph-distiluse-train.tsv.gz")
    parser.add_argument("--model", default="distiluse-base-multilingual-cased-v1")
    args = parser.parse_args()

    glob_documents = args.glob_documents
    glob_narratives = args.glob_narratives
    path_out = args.path_out
    model = args.model

    # MODEL
    print("loading model")
    sbert_model = SentenceTransformer(model)

    # NARRATIVES
    print("narratives")
    print("... reading")
    paths_narratives = glob(glob_narratives)
    dataframes = list()
    for path_narratives in paths_narratives:
        df = read_csv(path_narratives, sep="\t")
        df['sentence_nr'] = df.index
        df['sentence'] = df['Description']
        dataframes.append(df)
    narratives = concat(dataframes)
    print(f"... generating embeddings for {len(narratives)} sentences")
    narratives['embedding'] = narratives['sentence'].apply(sbert_model.encode)

    # DOCUMENTS
    print("documents")
    print("... reading")
    paths_docs = glob(glob_documents)
    dataframes = list()
    for path_doc in paths_docs:
        filename = os.path.basename(path_doc)
        with open(path_doc, "rt") as f:
            sentences = f.read().split("\n\n")
        df = DataFrame(data={'filename': filename, 'sentence': sentences})
        df['sentence_nr'] = df.index
        dataframes.append(df)
    docs = concat(dataframes)
    print(f"... generating embeddings for {len(docs)} sentences")
    docs['embedding'] = docs['sentence'].apply(sbert_model.encode)

    # COMPARISON
    print("comparing")
    sim = util.cos_sim(docs['embedding'].to_list(), narratives['embedding'].to_list())
    df = DataFrame(sim)
    df.index = docs['filename'].astype(str) + "_" + docs['sentence_nr'].astype(str)
    df.columns = narratives['Subnarrative'].astype(str) + "_" + narratives['sentence_nr'].astype(str)

    # WRITE
    print('writing')
    df.to_csv(path_out, sep="\t", compression="gzip")
