library(tidyverse)
library(ggplot2)

setwd("~/fakultet/heuristicke-metode-optimizacije/bayes-opt")

bayes_opt_results <- read.csv("bayes_opt_results.csv", header = F)
bayes_opt_results <- data.frame(bayes_opt_results)

bayes_opt_results <- 
    bayes_opt_results %>% 
    rename(start_samples = V1, surrogate_samples = V2, convergence_rate = V3) 


bayes_opt_results[which.max(bayes_opt_results$convergence_rate),]

    

bayes_plot <- ggplot(bayes_opt_results, aes(start_samples, surrogate_samples)) +
    geom_point(aes(color=-convergence_rate)) +
    scale_color_gradient(name = "convergence rate", trans = "log10", 
                          breaks = my_breaks, labels = my_breaks, 
                          low = "#ff5500", high = "#062161")+
    theme_dark()

bayes_plot

branin_hoo_results <- read.csv("branin_optimization.csv", header = F)
branin_hoo_results <- data.frame(branin_hoo_results)

branin_hoo_results <- 
    branin_hoo_results %>% 
    rename(x = V1, y = V2, value = V3)
branin_hoo_results$value <- -1 * branin_hoo_results$value  
my_breaks = c(2, 10, 40, 160, 320)

branin_plot <- ggplot(branin_hoo_results, aes(x, y)) + 
    geom_point(aes(color=value)) +
    scale_color_gradient(name = "value", trans = "log", breaks = my_breaks, labels = my_breaks, low = "#f27e3a", high = "#062161")+
    theme_dark()
branin_plot




summary(branin_hoo_results)
