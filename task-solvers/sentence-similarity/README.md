# sentence similarity

## similarity scores
- similarity scores are calculated with `sentence-similarity.py`
- this script compares all document sentences to all description sentences
- scores are calculated separatedly for train / dev / test
- two parameters:
  - segmentation: we compare either paragraphs from documents with paragraphs from descriptions (paragraph-paragraph) or paragraphs from documents with sentences from descriptions (paragraph-sentence)
  - model: "distiluse-base-multilingual-cased-v1" vs. "paraphrase-multilingual-mpnet-base-v2"
- bash script for running all experiments: `sentence-similarity-experiments.sh` 

## actual prediction
- actual binary predictions per narrative are computed in `sentence-similarity.R`
- we use the maximum over all pairwise comparisons as a final score for each document
- for prediction on dev set, we determine optimal cut-off on train set, for prediction on test set, we use train and dev set
- we provide results per language in the squashed format needed for submissions

## evaluation
- we run the script provided by the organisers for all languages and splits (`evaluation-dev.sh`)
- output is saved as `evaluation-dev-results.txt`
- `evaluation-dev-convert.py` converts to .tsv and .tex
