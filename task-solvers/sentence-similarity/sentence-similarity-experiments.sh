## "paraphrase-multilingual-MiniLM-L12-v2"
echo ""
echo "paraphrase-multilingual-MiniLM-L12-v2"
### paragraph-paragraph
echo "- paragraph-paragraph"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-mini-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-mini-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-mini-test.tsv.gz
### paragraph-sentence
echo "- paragraph-sentence"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-mini-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-mini-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-MiniLM-L12-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-mini-test.tsv.gz

## "paraphrase-xlm-r-multilingual-v1"
echo ""
echo "paraphrase-xlm-r-multilingual-v1"
### paragraph-paragraph
echo "- paragraph-paragraph"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-xlm-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-xlm-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-xlm-test.tsv.gz
### paragraph-sentence
echo "- paragraph-sentence"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-xlm-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-xlm-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-xlm-r-multilingual-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-xlm-test.tsv.gz

## "distiluse-base-multilingual-cased-v1"
echo "distiluse-base-multilingual-cased-v1"
### paragraph-paragraph
echo "- paragraph-paragraph"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-distiluse-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-distiluse-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-distiluse-test.tsv.gz
### paragraph-sentence
echo "- paragraph-sentence"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-distiluse-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-distiluse-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "distiluse-base-multilingual-cased-v1" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-distiluse-test.tsv.gz

## "paraphrase-multilingual-mpnet-base-v2"
echo ""
echo "paraphrase-multilingual-mpnet-base-v2"
### paragraph-paragraph
echo "- paragraph-paragraph"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-paraphrase-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-paraphrase-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-paragraph-paraphrase-test.tsv.gz
### paragraph-sentence
echo "- paragraph-sentence"
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/train/*/raw-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-paraphrase-train.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/dev/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-paraphrase-dev.tsv.gz
python3 task-solvers/sentence-similarity/sentence-similarity.py --glob_narratives "Narrative_Descriptions/Narrative_Description_*-sentences.tsv" --glob_documents "task_data/test/*/subtask-2-documents/*.txt" --model "paraphrase-multilingual-mpnet-base-v2" --path_out task-solvers/sentence-similarity/results/paragraph-sentence-paraphrase-test.tsv.gz
