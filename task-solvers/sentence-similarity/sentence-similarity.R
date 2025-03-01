library(tidyverse)
library(yardstick)
library(cutpointr)


# narratives #####
path_narratives <- Sys.glob("../../Narrative_Descriptions/Narrative_Description_*-paragraphs.tsv")

narratives <- read_tsv(path_narratives, id = "file_name", show_col_types = F) |> 
  mutate(topic = if_else(str_detect(file_name, "ClimateChange"), "CC", "URW")) |> 
  mutate(subnarrative = str_c(topic, Narrative, Subnarrative, sep = ": ")) 

# directly predicted the "Subnarrative", but unique
length(narratives$Subnarrative) == length(unique(narratives$Subnarrative))


# documents ####
path_train_documents <- Sys.glob("../../task_data/train/*/raw-documents/*.txt")
path_test_documents <- Sys.glob("../../task_data/test/*/subtask-2-documents/*.txt")
path_dev_documents <- Sys.glob("../../task_data/dev/*/subtask-2-documents/*.txt")

documents <- tibble(path = c(path_train_documents, path_test_documents, path_dev_documents)) |> 
  separate_wider_delim(path, "/", names_sep = "_") |> 
  mutate(doc_name = str_remove_all(path_7, ".txt")) |> 
  rename(lang = path_5, split = path_4) |> 
  select(doc_name, lang, split)

# splits and languages
documents |>
  group_by(lang, split) |> 
  summarise(n = n())

# documents are not unique
length(documents$doc_name) == length(unique(documents$doc_name))
documents |> mutate(dup = duplicated(doc_name)) |> filter(dup)
# RU-URW-1195 in train and test data
documents |> filter(doc_name == "RU-URW-1195")
# we remove it from the train data
documents <- documents |> anti_join(documents |> filter(doc_name == "RU-URW-1195", split == "train"))
documents |> filter(doc_name == "RU-URW-1195")

# macro topic from file-name
documents <- documents |> mutate(topic_file = case_when(str_detect(doc_name, "CC") ~ "CC",
                                                        str_detect(doc_name, "URW") ~ "URW",
                                                        str_detect(doc_name, "UA") ~ "URW",
                                                        str_detect(doc_name, "RU") ~ "URW"))


# gold #####
paths_gold <- Sys.glob("../../task_data/*/*/subtask-2-annotations.txt")

gold <- read_tsv(paths_gold, col_names = F, id = "file_name", show_col_types = F) |> 
  rename(doc_name = X1) |> 
  rename(subnarrative = X3) |> 
  separate_longer_delim(subnarrative, delim = ";") |> 
  mutate(doc_name = str_remove_all(doc_name, ".txt")) |> 
  select(doc_name, subnarrative) |> 
  left_join(documents |> select(doc_name, lang, split))

# plenty of subnarrative annotations
gold |> group_by(subnarrative, lang) |> summarise(n = n())

# macro topic from gold annotation
topics <- gold |> mutate(CC = str_detect(subnarrative, "CC: "), URW = str_detect(subnarrative, "URW: ")) |> 
  group_by(doc_name) |> summarise(CC = max(CC), URW = max(URW))

# there's one document that has both annotations, we just set it to unknown
topics |> group_by(CC, URW) |> summarise(n = n())
topics |> filter(CC == 1, URW == 1) |> pull(doc_name)

gold.topics <- topics |> 
  filter(doc_name != topics |> filter(CC == 1, URW == 1) |> pull(doc_name)) |> 
  mutate(topic = case_when(URW == 1 ~ "URW",
                           CC == 1 ~ "CC",
                           .default = "Other")) |> 
  mutate(topic_gold = as.factor(topic)) |> 
  select(doc_name, topic_gold)

documents <- documents |> left_join(gold.topics)

documents |> group_by(split, topic_file, topic_gold) |> summarise(n = n())

# similarity scores ####

pp <- read_tsv(Sys.glob("results/paragraph-paragraph-*.tsv.gz"), id = "experiment") |> 
  rename(doc_sentence = '...1') |> 
  mutate(experiment = str_remove_all(experiment, ".*/"),
         experiment = str_remove_all(experiment, ".tsv.gz")) |> 
  pivot_longer(cols = - c(experiment, doc_sentence), names_to = "subnarrative", values_to = "score")

ps <- read_tsv(Sys.glob("results/paragraph-sentence-*.tsv.gz"), id = "experiment") |> 
  rename(doc_sentence = '...1') |> 
  mutate(experiment = str_remove_all(experiment, ".*/"),
         experiment = str_remove_all(experiment, ".tsv.gz")) |> 
  pivot_longer(cols = - c(experiment, doc_sentence), names_to = "subnarrative", values_to = "score")

similarities <- rbind(pp, ps) |> 
  separate_wider_delim(experiment, delim = "-", names = c("segmentation_document", "segmentation_narrative", "model", "split")) |> 
  separate_wider_delim(doc_sentence, delim = ".txt_", names = c("doc_name", "doc_sentence")) |> 
  separate_wider_delim(subnarrative, delim = "_", names = c("subnarrative", "subnarrative_sentence"))

sentence_similarity <- similarities |> 
  mutate(segmentation = str_c(segmentation_document, segmentation_narrative, sep = "-")) |> 
  select(doc_name, doc_sentence, subnarrative, subnarrative_sentence, model, segmentation, score)

sentence_similarity <- sentence_similarity |>
  rename(Subnarrative = subnarrative) |> 
  left_join(narratives) |> 
  select(- c(Subnarrative, file_name, Narrative, Description))

# methodology: simple maximisation ####

# we use the maximum score of pairwise comparisons for prediction
prediction.scores <- sentence_similarity |> 
  group_by(model, segmentation, doc_name, subnarrative) |> 
  summarise(score = max(score),
            nr_comparisons = n()) |> 
  arrange(doc_name, desc(score)) |> 
  # join doc info
  left_join(documents)

## evaluation ####
prediction.eval <- prediction.scores |> left_join(
  gold |> select(doc_name, subnarrative) |> mutate(gold = T),
  by = c("doc_name", "subnarrative")
) |> mutate(truth = as.factor(replace_na(gold, F))) |> 
  select(lang, split, model, segmentation, doc_name, subnarrative, score, truth) |> 
  filter(split != "test") |> 
  filter(! is.na(subnarrative))

### distribution of scores ####
#### EN, paragraph-sentence ####
prediction.eval |> 
  filter(lang == "EN", segmentation == "paragraph-sentence") |> 
  ggplot(aes(x = score, y = model, col = truth)) +
  geom_boxplot() +
  facet_wrap(~ subnarrative)

# note that there's some subnarratives that only appear once for given language (either in train or in dev)
subnarratives.void <- prediction.eval |> 
  filter(model == "distiluse", segmentation == "paragraph-paragraph") |> 
  group_by(subnarrative, lang, truth) |> 
  summarise(n = n()) |> 
  group_by(subnarrative, lang) |> 
  summarise(n = n()) |> 
  filter(n != 2)

subnarratives.void |> print(n = 200)

### ROC ####

#### EN, paragraph-paragraph ####
prediction.eval |> 
  filter(lang == "EN", segmentation == "paragraph-paragraph") |> 
  anti_join(subnarratives.void) |>
  group_by(subnarrative, model) |> 
  roc_curve(truth, score, event_level = "second") |> 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, linetype = 2) +
  # theme(legend.position = "none") +
  facet_wrap(~ subnarrative)

# both models seem to work reasonably well for EN

#### ROC-AUC ####
prediction.eval.roc <- prediction.eval |> 
  anti_join(subnarratives.void) |>
  group_by(subnarrative, lang, model, segmentation) |> 
  roc_auc(truth, score, event_level = "second") |> 
  arrange(desc(.estimate))

prediction.eval.roc |> 
  group_by(lang, model, segmentation) |> 
  summarise(avg_roc_auc = mean(.estimate)) |> 
  arrange(lang, desc(avg_roc_auc)) |> 
  pivot_wider(names_from = segmentation, values_from = avg_roc_auc) |> 
  mutate(better_paragraph = `paragraph-paragraph` > `paragraph-sentence`)

# paragraph-paragraph is better in terms of ROC-AUC (except BG mini, by .001)

prediction.eval.roc |> 
  group_by(lang, model, segmentation) |> 
  summarise(avg_roc_auc = mean(.estimate)) |> 
  arrange(lang, desc(avg_roc_auc)) |> 
  pivot_wider(names_from = model, values_from = avg_roc_auc)

# paraphrase model is best in terms of ROC-AUC

### PR ####

#### EN, paragraph-paragraph ####

prediction.eval |> 
  filter(lang == "EN", segmentation == "paragraph-paragraph") |> 
  anti_join(subnarratives.void) |>
  group_by(subnarrative, model) |> 
  pr_curve(truth, score, event_level = "second") |> 
  ggplot(aes(x = recall, y = precision, col = model)) +
  geom_path() +
  coord_equal() +
  theme(legend.position = "none") +
  facet_wrap(~ subnarrative)

#### PR-AUC ####
prediction.eval.pr <- prediction.eval |> 
  anti_join(subnarratives.void) |>
  group_by(subnarrative, lang, model, segmentation) |> 
  pr_auc(truth, score, event_level = "second") |> 
  arrange(desc(.estimate))

prediction.eval.pr |> 
  group_by(lang, model, segmentation) |> 
  summarise(avg_roc_auc = mean(.estimate)) |> 
  arrange(lang, desc(avg_roc_auc)) |> 
  pivot_wider(names_from = segmentation, values_from = avg_roc_auc) |> 
  mutate(better_paragraph = `paragraph-paragraph` > `paragraph-sentence`)

# paragraph-paragraph better (except for RU paraphrase model, by .002)

prediction.eval.pr |> 
  group_by(lang, model, segmentation) |> 
  summarise(avg_roc_auc = mean(.estimate)) |> 
  arrange(lang, desc(avg_roc_auc)) |> 
  pivot_wider(names_from = model, values_from = avg_roc_auc)

# paraphrase model best, especially for HI and BG (except for RU paragraph-paragraph segmentation)

### prediction for dev ####

#### optimal F1 on train ####

# subnarratives that are not observed on train set
subnarratives.void.train <- prediction.eval |> 
  filter(model == "distiluse", segmentation == "paragraph-paragraph") |> 
  filter(split == "train") |> 
  group_by(subnarrative, lang, truth) |> 
  summarise(n = n()) |> 
  group_by(subnarrative, lang) |> 
  summarise(n = n()) |> 
  filter(n != 2)

cutoff.train <- prediction.eval |> 
  filter(split == "train") |>
  anti_join(subnarratives.void.train) |> 
  group_by(subnarrative, lang, model, segmentation) |> 
  group_modify(
    ~ cutpointr(.x, score, truth, method = maximize_metric, metric = F1_score, pos_class = TRUE, direction = ">=")
  ) |> 
  select(subnarrative, lang, model, segmentation, optimal_cutpoint)


# data.eval <- prediction.binary |> 
#   mutate(truth = as.logical(truth)) |> 
#   mutate(cat = case_when(
#     truth & prediction ~ "TP",
#     truth & !prediction ~ "FN",
#     !truth & prediction ~ "FP",
#     !truth & !prediction ~ "TN"
#   ))
# 
# eval.micro <- data.eval |> 
#   group_by(cat, lang, model, segmentation) |> 
#   summarise(freq = n()) |>  
#   mutate(subnarrative = "micro.avg")
# 
# eval.macro <- data.eval |> 
#   group_by(subnarrative, cat, lang, model, segmentation) |> 
#   summarise(freq = n())
# 
# eval.final <- rbind(eval.micro, eval.macro) |> 
#   pivot_wider(names_from = cat, values_from = freq)
# 
# 
# |> 
#   mutate(
#     across(everything(), ~replace_na(.x, 0)),
#     F1_score = 2 * TP / (2 * TP + FP + FN) 
#   ) |> 
#   select(subnarrative, lang, model, segmentation, F1_score) |> 
#   left_join(gold |> group_by(subnarrative, lang) |> summarise(support = n())) |> 
#   arrange(desc(support), desc(F1_score))
# 
# eval.final |> 
#   filter(subnarrative == "micro.avg") |> 
#   arrange(lang, desc(F1_score)) |> 
#   select(lang, model, segmentation, F1_score)

#### prediction for dev ####

# use cut-off on train set for determining actual prediction on dev set
prediction.dev <- prediction.eval |> 
  filter(split == "dev") |>
  left_join(cutoff.train) |> 
  mutate(prediction = (score >= optimal_cutpoint)) |> 
  select(doc_name, subnarrative, lang, model, segmentation, truth, prediction) |> 
  mutate(prediction = replace_na(prediction, F))

# we filter out if we know the topic
prediction.dev <- prediction.dev |>
  left_join(documents |> select(doc_name, split, topic_file)) |> 
  mutate(topic_match = str_detect(subnarrative, topic_file),
         topic_match = replace_na(topic_match, T),
         prediction = if_else(topic_match, prediction, F))

# we predict "Other" if the document has zero predictions
zero.pred <- prediction.dev |> 
  group_by(lang, model, segmentation, doc_name) |>
  summarise(n = n(), nr_pred = sum(prediction)) |> 
  ungroup() |> 
  filter(nr_pred == 0) |> 
  mutate(subnarrative = "Other") |> 
  select(doc_name, subnarrative, lang, model)

# rbind(zero.pred, prediction.dev)

prediction.dev |> 
  group_by(lang, model, segmentation, split) |> 
   mutate(truth = as.logical(truth)) |>
   mutate(cat = case_when(
      truth & prediction ~ "TP",
      truth & !prediction ~ "FN",
      !truth & prediction ~ "FP",
      !truth & !prediction ~ "TN"
  )) |> 
  group_by(cat, lang, model, segmentation, subnarrative) |>
  summarise(freq = n()) |>
  pivot_wider(names_from = cat, values_from = freq) |>
  mutate(
    across(everything(), ~replace_na(.x, 0)),
    nr_comparisons = TP + FP + TN + FN,
    F1_score = 2 * TP / (2 * TP + FP + FN)
  ) |> 
  ungroup() |>
  group_by(lang, model, segmentation) |> 
  summarise(F1_mean = mean(F1_score, na.rm = T),
            F1_sd = sd(F1_score, na.rm = T)) |> 
  print(n = 1000)
  ggplot(aes(y = model, x = F1_mean, col = segmentation)) + 
    geom_point() + 
    facet_wrap(~ lang, ncol = 1)

pred <- prediction.dev |>
  filter(prediction) |> 
  left_join(narratives) |> 
  mutate(narrative = str_c(topic, Narrative, sep = ": ")) |> 
  select(doc_name, lang, model, segmentation, narrative, subnarrative) |> 
  group_by(doc_name, lang, model, segmentation, narrative) |> 
  summarise(subnarrative = str_c(subnarrative, collapse = ";")) |> 
  group_by(doc_name, lang, model, segmentation) |> 
  summarise(subnarrative = str_c(subnarrative, collapse = ";"),
            narrative = str_c(narrative, collapse = ";")) |> 
  ungroup()

# write one file per language / model / segmentation
for (lang_ in (unique(pred$lang))){
  
  docs.dev <- documents |> filter(lang == lang_, split == "dev")
  
  for (model_ in (unique(pred$model))){
    for (segmentation_ in (unique(pred$segmentation))){

      p <- pred |> 
        filter(lang == lang_, model == model_, segmentation == segmentation_) |>
        select(doc_name, narrative, subnarrative) |> 
        right_join(docs.dev) |> 
        mutate(narrative = replace_na(narrative, "Other"),
               subnarrative = replace_na(subnarrative, "Other")) |> 
        arrange(doc_name) |> 
        mutate(doc_name = str_c(doc_name, ".txt"))
      
      p |> select(doc_name, narrative, subnarrative) |> 
        write_tsv(str_c("results/", lang_, "-", segmentation_, "-", model_, "-dev.txt"), col_names = F)
      
    }
  }
}


### prediction for test ####

subnarratives.void.train.dev <- prediction.eval |> 
  filter(model == "distiluse", segmentation == "paragraph-paragraph") |> 
  group_by(subnarrative, lang, truth) |> 
  summarise(n = n()) |> 
  group_by(subnarrative, lang) |> 
  summarise(n = n()) |> 
  filter(n != 2)

cutoff.train.dev <- prediction.eval |> 
  anti_join(subnarratives.void.train.dev) |> 
  group_by(subnarrative, lang, model, segmentation) |> 
  group_modify(
    ~ cutpointr(.x, score, truth, method = maximize_metric, metric = F1_score, pos_class = TRUE, direction = ">=")
  ) |> 
  select(subnarrative, lang, model, segmentation, optimal_cutpoint)

prediction.binary.test <- prediction |> 
  filter(split == "test") |> 
  left_join(cutoff.train.dev) |> 
  mutate(prediction = (score >= optimal_cutpoint)) 

pred.test <- prediction.binary.test |>
  filter(prediction) |> 
  left_join(narratives) |> 
  mutate(narrative = str_c(topic, Narrative, sep = ": ")) |> 
  select(doc_name, lang, model, segmentation, narrative, subnarrative) |> 
  group_by(doc_name, lang, model, segmentation, narrative) |> 
  summarise(subnarrative = str_c(subnarrative, collapse = ";")) |> 
  group_by(doc_name, lang, model, segmentation) |> 
  summarise(subnarrative = str_c(subnarrative, collapse = ";"),
            narrative = str_c(narrative, collapse = ";")) |> 
  ungroup()

# write one file per language / model / segmentation
for (lang_ in (unique(pred$lang))){
  
  docs.dev <- documents |> filter(lang == lang_, split == "test")
  
  for (model_ in (unique(pred$model))){
    for (segmentation_ in (unique(pred$segmentation))){
      p <- pred.test |> 
        filter(lang == lang_, model == model_, segmentation == segmentation_) |>
        select(doc_name, narrative, subnarrative) |> 
        right_join(docs.dev) |> 
        mutate(narrative = replace_na(narrative, "Other"),
               subnarrative = replace_na(subnarrative, "Other")) |> 
        arrange(doc_name) |> 
        mutate(doc_name = str_c(doc_name, ".txt"))
      
      p |> write_tsv(str_c("results/", lang_, "-", segmentation_, "-", model_, "-test.txt"), col_names = F)
      
    }
  }
}



# prediction via GLM ####
feature_vectors_long <- sentence_similarity |>
  group_by(doc_name, subnarrative, subnarrative_sentence, model, segmentation) |> 
  summarise(nr_comparisons = n(), score = max(score)) |> 
  ungroup()

feature_vectors_long |> filter(is.na(subnarrative))
# TODO: run again with new subnarrative names
feature_vectors_long <- feature_vectors_long |> filter(!is.na(subnarrative))


# PER SUBNARRATIVE, ACROSS LANGUAGES
languages <- documents |> pull(lang) |> unique()

# we only predict subnarratives that we observe in train and dev
subnarratives.train.dev <- gold |>
  group_by(subnarrative, split) |>
  summarise(n = n()) |> 
  group_by(subnarrative) |> 
  summarise(n = n()) |> 
  filter(n == 2) |> 
  pull(subnarrative) |> 
  unique()

# we need to select possible features: only those that have been observed on train set
gold.train <- gold |> filter(split == "train")
subnarratives.train <- gold.train |> pull(subnarrative) |> unique()

y.eval.total <- tibble()
for (subnarrative_ in subnarratives.train.dev){

  feature_vectors <- feature_vectors_long |>
    filter(# model == "paraphrase",
           # segmentation ==  "paragraph-paragraph",
           subnarrative %in% subnarratives.train) |> 
    mutate(feature = str_c(subnarrative, "_", subnarrative_sentence, "_", model, "_", segmentation)) |> 
    select(doc_name, feature, score) |> 
    pivot_wider(id_cols = doc_name, names_from = feature, values_from = score) |> 
    left_join(documents |> select(doc_name, lang))
  
  Xy.train <- feature_vectors |> 
    left_join(documents) |> 
    filter(lang == lang_, split == "train") |> 
    select(-c(lang, split)) |> 
    left_join(gold |> filter(subnarrative == subnarrative_)) |>
    mutate(gold = as.factor(!is.na(subnarrative))) |> 
    select(-c(lang, split, subnarrative))
  
  model.train <- Xy.train |> select(- doc_name) |>
    glm(formula = gold ~ ., family = binomial)
  
  X.dev <- feature_vectors |> 
    left_join(documents) |> 
    filter(split == "dev") |> 
    select(-c(lang, split))
  
  y.pred <- feature_vectors |> 
    left_join(documents) |> 
    filter(split == "dev") |> 
    select(doc_name) |> 
    mutate(score = model.train |> predict.glm(X.dev |> select(- doc_name)))
  
  y.eval <- y.pred |> left_join(gold |> filter(subnarrative == subnarrative_)) |> 
    mutate(gold = as.factor(!is.na(subnarrative))) |> 
    select(doc_name, score, gold) |> 
    mutate(subnarrative = subnarrative_)
  
  # y.eval |> 
  #   ggplot(aes(x = score, y = gold)) +
  #   geom_boxplot() +
  #   geom_jitter()
  
  y.eval.total <- rbind(y.eval.total, y.eval)

}

y.eval.total |> 
  left_join(documents) |> 
  filter(lang == "EN") |> 
  ggplot(aes(x = score, y = gold)) +
  geom_boxplot() +
  facet_wrap(~ subnarrative, scales = "free_x") +
  theme(legend.position = "none")

y.eval.total |> 
  left_join(documents) |> 
  group_by(subnarrative, lang) |> 
  roc_curve(gold, score, event_level = "second") |> 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = lang)) +
  geom_path() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, linetype = 2) +
  # theme(legend.position = "none") +
  facet_wrap(~ subnarrative)


# TOPIC FILTER
d <- read_tsv(Sys.glob("results/*.txt"), id = "file_name", col_names = F)

Xy <- feature_vectors |> 
  left_join(documents) |> 
  filter(doc_name != (topics |> filter(CC == 1, URW == 1) |> pull(doc_name))) |> 
  left_join(gold.topics) |> 
  filter(lang == "EN")

model.train <- Xy.train |> 
  filter(split == "train") |> 
  select(- c(doc_name, lang, split)) |>
  glm(formula = topic ~ ., family = binomial)

X.dev <- Xy |> 
  filter(split == "dev") |> 
  select(- lang, split)

score = model.train |> predict.glm(
    X.dev |> select(- doc_name)
  )

y.pred.dev <- X.dev |> 
  left_join(documents) |> 
  filter(split == "dev") |> 
  select(doc_name) |> 
  mutate(score = score) |> 
  left_join(gold.topics)


y.topic
