# Finegrained Narratives model

A Masked LLM(XLM-RoBERTa) was trained on the whole set of languages.
## Training
Most of the training was done using Jupyter Notebooks. In the [main](main.ipynb) notebook we take the combined dataset, and train the model for 100 epochs.

[format_results](format_results.ipynb) formats the results from one-hot encoded dataset to the test file format. 