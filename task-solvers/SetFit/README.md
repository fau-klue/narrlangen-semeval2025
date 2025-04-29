# SetFit

## Target matrices
- training and development files provided by the task organisers first need to be converted to matrices
- the provided R Script (`create_target_matrices.R`) can be used for this
- paths in the script need to be changed, however (sorry)

# Training
- run `setfit_simple_multiling.py` to train a SetFit model on the training data (adapted to our needs in the previous step)
- adapt the file paths in the file (`path_dataset1` etc.) as needed

## Actual prediction
- actual binary predictions per narrative are computed in `setfit_get_predictions_multiling.py`
- `path_data` and `path_model` in the file need to be changed according to the proper locations on your system
- we provide results per language in the squashed format needed for submissions (but files in matrix format and predicted probabilities of fine-grained labels will also be written to separate files)

## Evaluation
- we run the script provided by the organisers for all languages and splits (`evaluation-dev.sh`)
- output is saved as `evaluation-dev-results.txt`
- `evaluation-dev-convert.py` converts to .tsv and .tex
