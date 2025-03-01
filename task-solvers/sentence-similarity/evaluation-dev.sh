echo "============ EN ============"
echo "-- distiluse: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-paragraph-distiluse-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- distiluse: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-sentence-distiluse-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-paragraph-paraphrase-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-sentence-paraphrase-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-paragraph-mini-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-sentence-mini-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-paragraph-xlm-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/EN-paragraph-sentence-xlm-dev.txt --gold_file ../../task_data/dev/EN/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt

echo "============ PT ============"
echo "-- distiluse: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-paragraph-distiluse-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- distiluse: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-sentence-distiluse-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-paragraph-paraphrase-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-sentence-paraphrase-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-paragraph-mini-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-sentence-mini-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-paragraph-xlm-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/PT-paragraph-sentence-xlm-dev.txt --gold_file ../../task_data/dev/PT/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt

echo "============ BG ============"
echo "-- distiluse: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-paragraph-distiluse-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- distiluse: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-sentence-distiluse-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-paragraph-paraphrase-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-sentence-paraphrase-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-paragraph-mini-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-sentence-mini-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-paragraph-xlm-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/BG-paragraph-sentence-xlm-dev.txt --gold_file ../../task_data/dev/BG/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt

echo "============ RU ============"
echo "-- distiluse: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-paragraph-distiluse-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- distiluse: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-sentence-distiluse-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-paragraph-paraphrase-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-sentence-paraphrase-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-paragraph-mini-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-sentence-mini-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-paragraph-xlm-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/RU-paragraph-sentence-xlm-dev.txt --gold_file ../../task_data/dev/RU/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt

echo "============ HI ============"
echo "-- distiluse: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-paragraph-distiluse-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- distiluse: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-sentence-distiluse-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-paragraph-paraphrase-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- paraphrase: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-sentence-paraphrase-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-paragraph-mini-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- mini: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-sentence-mini-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-paragraph"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-paragraph-xlm-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
echo "-- xlm: paragraph-sentence"
python3 ../../scorers_baselines/subtask2_scorer.py --prediction_file results/HI-paragraph-sentence-xlm-dev.txt --gold_file ../../task_data/dev/HI/subtask-2-annotations.txt --classes_file_fine ../../scorers_baselines/subtask2_subnarratives.txt --classes_file_coarse ../../scorers_baselines/subtask2_narratives.txt
