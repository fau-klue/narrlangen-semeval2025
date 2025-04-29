library(tidyverse)
library(tidytext)

languages <- c("BG", "EN", "HI", "PT", "RU")
narratives_coarse <- read_csv("task_data/subtask2_all_coarse-grained_narratives.csv",
                              show_col_types = FALSE) |>
  pull(narrative)
narratives_fine <- read_csv("task_data/subtask2_all_fine-grained_narratives.csv",
                            show_col_types = FALSE) |>
  pull(narrative)

for (lang in languages) {
  for (set in c("dev", "train")) {
    print(str_glue("Processing {set} data for {lang} ..."))
    # read in the data
    anno_file <- str_glue("task_data/{set}/{lang}/subtask-2-annotations.txt")
    if (set == "train") {
      path_to_docs <- str_glue("task_data/{set}/{lang}/raw-documents/")
    } else {
      path_to_docs <- str_glue("task_data/{set}/{lang}/subtask-2-documents/")
    }
    labels <- read_tsv(anno_file, col_names = c("file", "coarsegrained", "finegrained"),
                       show_col_types = FALSE)
    texts <- tibble(file = character(),
                    text = character())
    for (path in list.files(path_to_docs, full.names = TRUE)) {
      file <- str_remove(path, path_to_docs)
      text <- read_file(path)
      texts <- texts |>
        bind_rows(tibble(file = file, text = text))
    }
    
    # split texts at semicolons to get individual labels
    labels_long <- labels |>
      separate_longer_delim(coarsegrained:finegrained, ";") |>
      mutate(coarsegrained = factor(coarsegrained, levels = narratives_coarse),
             finegrained = factor(finegrained, levels = narratives_fine))
    
    # create a barplot of fine-grained category frequencies
    if (!dir.exists("eda_plots")) {dir.create("eda_plots")}
    bplot <- labels_long |>
      mutate(topic = ifelse(str_detect(coarsegrained, "^CC"),
                            "Climate change",
                            ifelse(str_detect(coarsegrained, "URW"),
                                   "Russo-Ukrainian War",
                                   "Other")) |>
               factor(levels = c("Climate change", "Russo-Ukrainian War", "Other")),
             finegrained = finegrained |>
               fct_relabel(\(x) str_remove(x, "CC: |URW: "))) |>
      ggplot(aes(x = finegrained, fill = topic)) +
      geom_bar() +
      scale_fill_discrete(drop = FALSE) +
      scale_x_discrete(drop = FALSE) +
      theme(axis.text.x = element_text(angle = -45, hjust = 0),
            plot.margin = margin(t = 5.5, r = 50, b = 5.5, l = 5.5))
    ggsave(str_glue("eda_plots/task2_{lang}_{set}_categories.svg"),
           bplot,
           scale = 2.5,
           width = 148, height = 105, units = "mm")

    # prepare data for matrix conversion
    labels_coarse <- labels_long |>
      select(file, label = coarsegrained) |>
      count(file, label) |>
      mutate(n = 1) # we only need to know if a course-grained label exists
    
    # make sure the last possible narrative level is included in the data;
    # otherwise, cast_dfm() will throw an error
    if (!last(narratives_coarse) %in% labels_coarse$label) {
      labels_coarse <- labels_coarse |>
        bind_rows(tibble(file = labels_coarse$file[1],
                         label = last(narratives_coarse) |>
                           factor(levels = narratives_coarse),
                         n = 0))
    }
    
    # create target matrices to use in ML models
    labels_coarse <- labels_coarse |>
      cast_dfm(file, label, n) |>
      quanteda::convert(to = "data.frame", docid_field = "file") |>
      as_tibble()
    
    # same stuff for fine-grained labels
    labels_fine <- labels_long |>
      select(file, label = finegrained) |>
      count(file, label)
    
    if (!last(narratives_fine) %in% labels_fine$label) {
      labels_fine <- labels_fine |>
        bind_rows(tibble(file = labels_fine$file[1],
                         label = last(narratives_fine) |>
                           factor(levels = narratives_fine),
                         n = 0))
    }
    
    labels_fine <- labels_fine |>
      cast_dfm(file, label, n) |>
      quanteda::convert(to = "data.frame", docid_field = "file") |>
      as_tibble()
    
    # combine these with texts
    labels_coarse <- labels_coarse |>
      left_join(texts, by = "file") |>
      select(file, text, everything())
    
    labels_fine <- labels_fine |>
      left_join(texts, by = "file") |>
      select(file, text, everything())
    
    # write files
    output_dir_coarse <- str_glue("task_data/subtask2_hf/{lang}_coarse-grained/")
    output_dir_fine <- str_glue("task_data/subtask2_hf/{lang}_fine-grained/")
    if (!dir.exists(output_dir_coarse)) {dir.create(output_dir_coarse)}
    if (!dir.exists(output_dir_fine)) {dir.create(output_dir_fine)}
    
    labels_coarse |>
      write_csv(str_glue("{output_dir_coarse}{set}.csv"))
    
    labels_fine |>
      write_csv(str_glue("{output_dir_fine}{set}.csv"))
  }
}

