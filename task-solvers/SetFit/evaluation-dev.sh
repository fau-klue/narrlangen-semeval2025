echo "============ EN ============"
echo "-- setfit: multilingual"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/setfit_simple_multiling_EN_fine-grained_dev_cats_target.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "============ BG ============"
echo "-- setfit: multilingual"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/setfit_simple_multiling_BG_fine-grained_dev_cats_target.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "============ PT ============"
echo "-- setfit: multilingual"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/setfit_simple_multiling_PT_fine-grained_dev_cats_target.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "============ RU ============"
echo "-- setfit: multilingual"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/setfit_simple_multiling_RU_fine-grained_dev_cats_target.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "============ HI ============"
echo "-- setfit: multilingual"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/setfit_simple_multiling_HI_fine-grained_dev_cats_target.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
