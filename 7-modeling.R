library(dplyr)

# Pseudo R2: pscl::pR2(m)['McFadden']

d <- read.csv('../research-data/processed/lak23wlsample.csv')

py_to_r_bool <- function(v) {
v <- as.character(v)
v[v==""] <- NA
v <- v == "True" # NA == TRUE -> NA
return(v)
}

d$student_dropped_program <- d$student_dropped_program %>%
    py_to_r_bool()

d$graduated_on_time <- d$graduated_on_time %>%
    py_to_r_bool()

# Scale variables for modeling
d_model <- d

for (variable in c('sem_load_credit_hours', 'sem_load_predicted', 'n_courses', 'ratio_courses_stem_match', 'X.SEMESTER_GPA'))
    d_model[variable] <- d_model[variable] %>% scale()

# Model predicts that larger semester load in credit hours is negatively associated with dropout
m <- glm(student_dropped_program ~ sem_load_credit_hours, d_model, family='binomial')
summary(m)

# Analytical load not associated with dropout alone
m <- glm(student_dropped_program ~ sem_load_predicted, d_model, family='binomial')
summary(m)

# Together they have significant additive effect, where analytical load makes dropout more likely given a particular CH load
m <- glm(student_dropped_program ~ sem_load_credit_hours + sem_load_predicted, d_model, family='binomial')
summary(m)

# Here is the prediction space
m <- glm(student_dropped_program ~ sem_load_credit_hours * sem_load_predicted, d, family='binomial')
summary(m)

# ANOVA
m0 <- glm(student_dropped_program ~ sem_load_credit_hours, d_model, family='binomial')
m1 <- glm(student_dropped_program ~ sem_load_credit_hours + sem_load_predicted, d_model, family='binomial')
m2 <- glm(student_dropped_program ~ sem_load_credit_hours * sem_load_predicted, d, family='binomial')
anova(m0, m1, m2, test='Chisq')

# Repeat with on-time graduation

# Model predicts that larger semester load in credit hours is positively associated with on-time graduation
m <- glm(graduated_on_time ~ sem_load_credit_hours, d_model, family='binomial')
summary(m)

# Model predicts that larger semester load in predicted is positively associated with on-time graduation
m <- glm(graduated_on_time ~ sem_load_predicted, d_model, family='binomial')
summary(m)

# Model finds a significant interaction
m <- glm(graduated_on_time ~ sem_load_credit_hours + sem_load_predicted, d, family='binomial')
summary(m)

m <- glm(graduated_on_time ~ sem_load_credit_hours * sem_load_predicted, d, family='binomial')
summary(m)

# ANOVA
m0 <- glm(graduated_on_time ~ sem_load_credit_hours, d_model, family='binomial')
m1 <- glm(graduated_on_time ~ sem_load_credit_hours + sem_load_predicted, d_model, family='binomial')
m2 <- glm(graduated_on_time ~ sem_load_credit_hours * sem_load_predicted, d_model, family='binomial')
anova(m0, m1, m2, test='Chisq')
