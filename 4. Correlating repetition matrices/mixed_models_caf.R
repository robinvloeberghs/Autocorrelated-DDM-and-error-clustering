library(lme4)
library(lmerTest)
library(car)
library(emmeans)
library(rstudioapi)
library(dplyr)
library(ggplot2)


curdir <- dirname(getSourceEditorContext()$path)
setwd(curdir)

# Load in data ------------------------------------------------------------
df <- read.csv("C:/Users/u0141056/OneDrive - KU Leuven/PhD/PROJECTS/DDM and clustered errors/Analysis/correlation_empirical_predicted_caf.csv")

# Change order such that we have contrast variability - vanilla later in mixed models (always most complex model first)
df$model <- factor(df$model, levels = c("autocorrelated", "variability", "vanilla"))

# Mixed model over all datasets -------------------------------------------
m_total <- lmer(data=df, correlation ~ model +
                                (1 | dataset/subject),
          control=lmerControl(optimizer='bobyqa',optCtrl = list(maxfun=100000)))
summary(m_total)
anova(m_total)
eta_squared(m_total, partial = TRUE) # partial eta squared


# Contrasts and p-values
m_contrasts_total <- emmeans(m_total, ~ model)
df_contrasts_total <- data.frame(pairs(m_contrasts_total))
df_contrasts_total

# Confidence intervals
df_ci_total <- data.frame(confint(pairs(m_contrasts_total)))
df_ci_total

# Effect sizes
df_eff_size_total <- data.frame(eff_size(m_contrasts_total, sigma = sigma(m_total), edf = df.residual(m_total), method = "pairwise"))
df_eff_size_total

df_contrasts_total$lower_CI <- df_ci_total$lower.CL
df_contrasts_total$upper_CI <- df_ci_total$upper.CL
df_contrasts_total$eff_size <- df_eff_size_total$effect.size
df_contrasts_total$dataset <- "Total"
df_contrasts_total

# Winning model per subject
counts_winners <- df %>%
  group_by(dataset_subject) %>%
  slice_max(order_by = correlation, n = 1, with_ties = FALSE) %>%
  count(model)

table(counts_winners$model)


## Plotting ----------------------------------------------------------------

# Create unique subjects
df$subject_transformed <- paste(df$dataset, df$subject, sep = "_")
df_summary <- df %>%
  group_by(model) %>%
  summarise(
    mean_cor = mean(correlation, na.rm = TRUE),
    se_cor = sd(correlation, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

ggplot(df, aes(x = model, y = correlation)) +
  
  geom_line(
    aes(group = subject_transformed),
    colour = "black",
    alpha = .075
  ) +

  geom_line(
    data = df_summary,
    aes(x = model, y = mean_cor, group = 1),
    linewidth = 1,
    colour = "#36848C",
  ) +
  
  geom_point(
    data = df_summary,
    aes(x = model, y = mean_cor),
    size = 2.5,
    colour = "#36848C",
  ) +

  geom_errorbar(
    data = df_summary,
    aes(
      x = model,
      ymin = mean_cor - se_cor,
      ymax = mean_cor + se_cor
    ),
    width = .05,
    inherit.aes = FALSE,
    colour = "#36848C") +  
  labs(
    x = "Model",
    y = "Correlation",
    #title ="Correlation empirical and predicted repetition matrix"
  ) +
  scale_x_discrete(expand = expansion(add = c(.1, .1))) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.title.y = element_text(size = 14),
    axis.line = element_line(linewidth = .6)
  )

ggsave(
  "caf_correlation_over_datasets.png",
  width = 4,
  height = 4,
  units = "in",
  dpi = 600
)



# Mixed model for individual datasets -------------------------------------

"Adler_2018_Expt1_taskA"
"Adler_2018_Expt1_taskB"
"Adler_2018_Expt3"
"Desender_2022_exp2B"
"Law_unpub"
"Maniscalco_2017_expt2"
"Shekhar_2021"
"Zang_2026"

m_individual_dataset <- lmer(data=df[df$dataset=="Zang_2026",], 
                              correlation ~ model +
                              (1 | subject),
                              control=lmerControl(optimizer='bobyqa',optCtrl = list(maxfun=100000)))
summary(m_individual_dataset)
Anova(m_individual_dataset)
m_contrasts <- emmeans(m_individual_dataset, ~ model)
pairs(m_contrasts)



# Save the coefficients ---------------------------------------------------


datasets <- c("Adler_2018_Expt1_taskA","Adler_2018_Expt1_taskB",
              "Adler_2018_Expt3","Desender_2022_exp2B","Law_unpub",
              "Maniscalco_2017_expt2","Shekhar_2021","Zang_2026")

all_coefs <- data.frame()

for(dataset in datasets){
  
  m_individual_dataset <- lmer(data=df[df$dataset==dataset,], 
                               correlation ~ model +
                                 (1 | subject),
                               control=lmerControl(optimizer='bobyqa',optCtrl = list(maxfun=100000)))
  
  m_contrasts_individual_dataset <- emmeans(m_individual_dataset, ~ model)
  
  # Contrasts and p-values
  df_contrasts <- data.frame(pairs(m_contrasts_individual_dataset))
  
  # Confidence intervals
  df_ci <- data.frame(confint(pairs(m_contrasts_individual_dataset)))
  
  # Effect sizes
  df_eff_size <- data.frame(eff_size(m_contrasts_individual_dataset, sigma = sigma(m_individual_dataset), edf = df.residual(m_individual_dataset), method = "pairwise"))
  
  df_contrasts$lower_CI <- df_ci$lower.CL
  df_contrasts$upper_CI <- df_ci$upper.CL
  df_contrasts$eff_size <- df_eff_size$effect.size
  df_contrasts$dataset <- dataset
  
  all_coefs <- rbind(all_coefs, df_contrasts)
  
}

# Add the results from mixed models over all datasets
all_coefs <- rbind(all_coefs, df_contrasts_total)

# Add results from mixed model over all datasets

all_coefs$p_significance <- ifelse(all_coefs$p.value < 0.001, "***", ifelse(all_coefs$p.value < 0.01, "**", ifelse(all_coefs$p.value < 0.05, "*", " ")))

# Save in .csv file
#write.csv(all_coefs, '_all_mixed_models_coefficients.csv')




# Change names
levels(all_coefs$dataset) <- rev(c(
  'Adler & Ma (2018, experiment 1A)',
  'Adler & Ma (2018, experiment 1B)',
  'Adler & Ma (2018, experiment 3)',
  'Desender et al. (2022, experiment 2B)',
  'Law & Lee (unpublished)',
  'Maniscalco et al. (2017, experiment 2)',
  'Shekhar & Rahnev (2021, experiment 3)',
  'Zang et al. (2026)',
  'Total'
))


dataset_labels <- c(
  Adler_2018_Expt1_taskA = "Adler & Ma (2018, exp. 1A)",
  Adler_2018_Expt1_taskB = "Adler & Ma (2018, exp. 1B)",
  Adler_2018_Expt3  = "Adler & Ma (2018, exp. 3)",
  Desender_2022_exp2B = 'Desender et al. (2022, exp. 2B)',
  Law_unpub = "Law & Lee (unpublished)",
  Maniscalco_2017_expt2 = "Maniscalco et al. (2017, exp. 2)",
  Shekhar_2021 = "Shekhar & Rahnev (2021, exp. 3)",
  Zang_2026 = 'Zang et al. (2026)',
  Total = "Total"
)

# Sort alphabetically with Total as last
desired_order <- rev(c(sort(unique(all_coefs[all_coefs$dataset != 'Total',]$dataset), decreasing = FALSE),'Total'))


all_coefs$dataset <- factor(
  dataset_labels[all_coefs$dataset],
  levels = dataset_labels[desired_order]
)


size_multiplier <- 2

ggplot(all_coefs[all_coefs$contrast=="autocorrelated - vanilla",], aes(y = dataset, x = estimate, col = dataset)) +
  geom_vline(xintercept = 0, color = '#CCCCCC', linewidth = 1 * size_multiplier) +
  geom_point(size = 2.5 * size_multiplier) +  # Adjust width for wider spacing , color = "#21918C"
  geom_errorbarh(aes(xmin = lower_CI, xmax = upper_CI), height = 0.2, linewidth = 0.4 * size_multiplier) +
  labs(x = expression(r[autocorrelated - vanilla]),
       y = "Dataset",
       title = "Autocorrelated vs. Vanilla") +
  scale_color_viridis_d(option="C", end=.8) +
  geom_text(aes(label = paste(p_significance)), vjust = -0.1, position = position_dodge(width = 0.2), size = 3.5 * size_multiplier) +
  theme_classic() +
  theme(
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 15), 
    axis.title = element_text(size = 25), 
    title = element_text(size = 14),  
    legend.position = "none"
  )

ggsave(
  "contrast_autocorrelated_vanilla.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600
)



ggplot(all_coefs[all_coefs$contrast=="autocorrelated - variability",], aes(y = dataset, x = estimate, col = dataset)) +
  geom_vline(xintercept = 0, color = '#CCCCCC', linewidth = 1 * size_multiplier) +
  geom_point(size = 2.5 * size_multiplier) +  # Adjust width for wider spacing , color = "#21918C"
  geom_errorbarh(aes(xmin = lower_CI, xmax = upper_CI), height = 0.2, linewidth = 0.4 * size_multiplier) +
  labs(x = expression(r["autocorrelated - across trial variability"]),
       y = "Dataset",
       title = "Autocorrelated vs. Across-trial variability") +
  scale_color_viridis_d(option="C", end=.8) +
  geom_text(aes(label = paste(p_significance)), vjust = -0.1, position = position_dodge(width = 0.2), size = 3.5 * size_multiplier) +
  theme_classic() +
  theme(
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 15), 
    axis.title = element_text(size = 25), 
    title = element_text(size = 14),  
    legend.position = "none"
  )

ggsave(
  "contrast_autocorrelated_variability.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600
)



ggplot(all_coefs[all_coefs$contrast=="variability - vanilla",], aes(y = dataset, x = estimate, col = dataset)) +
  geom_vline(xintercept = 0, color = '#CCCCCC', linewidth = 1 * size_multiplier) +
  geom_point(size = 2.5 * size_multiplier) +  # Adjust width for wider spacing , color = "#21918C"
  geom_errorbarh(aes(xmin = lower_CI, xmax = upper_CI), height = 0.2, linewidth = 0.4 * size_multiplier) +
  labs(x = expression(r["across trial variability - vanilla"]),
       y = "Dataset",
       title = "Across-trial variability vs. Vanilla") +
  scale_color_viridis_d(option="C", end=.8) +
  geom_text(aes(label = paste(p_significance)), vjust = -0.55, position = position_dodge(width = 0.2), size = 3.5 * size_multiplier) +
  theme_classic() +
  theme(
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 15), 
    axis.title = element_text(size = 25), 
    title = element_text(size = 14),  
    legend.position = "none"
  )


ggsave(
  "contrast_vanilla_variability.png",
  width = 7,
  height = 5,
  units = "in",
  dpi = 600
)

