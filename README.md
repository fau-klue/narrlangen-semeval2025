# Narrlangen @ SemEval2025 Task 10 (Subtask 2)

## Comparing (mostly) simple multilingual approaches to narrative classification

This repository accompanies our contribution to SemEval 2025 Task 10, SubTask 2: Multlingual characterisation and extraction from online news.

Our [task solvers](task-solvers/) comprise simple (bag-of-words) machine learning [baselines](task-solvers/ml_baseline/), prompt engineering of LLMs (only for English), a zero-shot approach based on [sentence similarity](task-solvers/sentence-similarity/), direct classification of fine-grained labels using SetFit, fine-tuning encoder models on fine-grained labels, and [hierarchical classification](task-solvers/multi_label_hierarchical_model/) using encoder models with two different classification head.

The manually crafted narrative descriptions -- about [climate change](narrative-descriptions/Narrative_Description_ClimateChange-sentences.tsv) and the [War in Ukraine](narrative-descriptions/Narrative_Description_War_in_Ukraine-sentences.tsv) -- are solely used in the sentence similarity approach.

For more details, refer to our upcoming publication:

```bibtex
@inproceedings{BlombachETC-SemEval2025,
   title={Narrlangen at SemEval-2025 Task 10: Comparing (mostly) simple multilingual approaches to narrative classification},
   author={Blombach, Andreas and Doan Dang, Bao Minh and Evert, Stephanie and Fuchs, Tamara and Heinrich, Philipp and Kalashnikova, Olena and Unjum, Naveed},
   booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation},
   series = {SemEval 2025},
   year = {2025},
   address = {Vienna, Austria},
   month = {July},
   pages = {}, 
   doi= {} 
}
```
