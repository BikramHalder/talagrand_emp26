library("MASS") # for logistic regression
library("class") # for kNN
library("e1071") # for SVM
library("randomForest") # for randomForest
library("dplyr") # for data manipulation
library("ggplot2") # for plotting graphs

# conditional expectation of Y given X=x: needed for estimating Bayes risk
eta_function <- function(x1, alpha)  0.5 + 0.5 * sign(x1) * abs(x1)^alpha

bayes_classifier <- function(X) ifelse(2*eta_function(X)>1,1,-1)
bayes_risk <- function(test_data) {
  bayes_pred <- bayes_classifier(test_data$X)
  mean(bayes_pred != test_data$Y)
}

generate_data <- function(n, d, alpha) {
  X <- matrix(runif(n * d, -1, 1), nrow = n)
  eta <- 0.5 + 0.5 * sign(X[,1]) * abs(X[,1])^alpha
  Y <- 2*rbinom(n, size = 1, prob = eta)-1
  data.frame(Y = factor(Y), X)
}

generate_test_set <- function(N_test, d, alpha) {
  X <- matrix(runif(N_test * d, -1, 1), nrow = N_test)
  eta <- eta_function(X[,1], alpha)
  Y <- 2*rbinom(N_test, 1, eta)-1
  list(X = X, Y = Y)
}

classification_risk <- function(pred, truth) mean(pred != truth)

# logistic regression (essentially a classifier here)
run_logistic <- function(train_data, test_data) {
  train_data$Y <- as.factor((as.numeric(train_data$Y)+1)/2)
  model <- glm(Y ~ ., data = train_data, family = binomial())
  probs <- predict(model, newdata = data.frame(test_data$X),type = "response")
  preds <- ifelse(probs > 0.5, 1, -1)
  classification_risk(preds, test_data$Y)
}

# kNN experiment
run_knn <- function(train_data, test_data, k = 15) {
  train_x <- as.matrix(train_data[,-1])
  test_x  <- test_data$X
  pred <- knn(train = train_x,
              test = test_x,
              cl = train_data$Y,
              k = k)
  classification_risk(as.numeric(pred),test_data$Y)
}

# SVM experiment
run_svm <- function(train_data, test_data) {
  model <- svm(Y ~ ., data = train_data, kernel = "radial")
  pred <- predict(model,newdata = data.frame(test_data$X))
  classification_risk(as.numeric(pred),test_data$Y)
}

# random forest
run_rf <- function(train_data,test_data) {
  model <- randomForest(Y ~ .,data = train_data,ntree = 200 )
  pred <- predict(model,newdata = data.frame(test_data$X))
  classification_risk(as.numeric(pred),test_data$Y)
}

## Novel extension: heavy tailed covariates
# Extension A: Replacing Uniform predictors with heavy-tailed t-distribution.

generate_heavy_tail_data <- function(n,d,alpha,df_t = 2) {
  X <- matrix(rt(n * d, df = df_t),nrow = n)
  X <- X / max(abs(X))
  eta <- eta_function(X[,1], alpha)
  Y <- 2*rbinom(n, 1, eta)-1
  data.frame(Y = factor(Y), X)
}

# generating test data
generate_heavy_tail_test <- function(N_test,d,alpha,df_t = 2) {
  X <- matrix(rt(N_test * d, df = df_t), nrow = N_test)
  X <- X / max(abs(X))
  eta <- eta_function(X[,1], alpha)
  Y <- 2*rbinom(N_test, 1, eta)-1
  list(X = X, Y = Y)
}

# Extension B: Correlated Gaussian Design
# Sigma(i,j)=rho^|i-j|
# we consider positive correlation
generate_correlated_data <- function(n,d,alpha,rho = 0.8) {
  Sigma <- outer(1:d,1:d, function(i,j) rho^abs(i-j))
  X <- mvrnorm(n = n, mu = rep(0, d), Sigma = Sigma)
  X <- X / max(abs(X))#to bring it to the range of [-1,1]
  eta <- eta_function(X[,1], alpha)
  Y <- 2*rbinom(n, 1, eta)-1
  data.frame(Y = factor(Y), X)
}

generate_correlated_test <- function(N_test,d,alpha,rho = 0.7) {
  Sigma <- outer(1:d,1:d, function(i,j) rho^abs(i-j))
  X <- mvrnorm(n = N_test,mu = rep(0, d), Sigma = Sigma)
  X <- X / max(abs(X))
  eta <- eta_function(X[,1], alpha)
  Y <- 2*rbinom(N_test, 1, eta)-1
  list(X = X, Y = Y)
}

# Extension C: Nonlinear Bayes Boundary
# Consider the Bayes decision rule x_1+x_2^2=0 based on (x_1,x_2,...,x_d)
generate_nonlinear_boundary_data <- function(n,d,alpha) {
  X <- matrix(runif(n*d, -1, 1),nrow = n)
  boundary_signal <-  X[,1] + X[,2]^2
  eta <- 0.5 + 0.5 *sign(boundary_signal) * abs(boundary_signal)^alpha
  eta <- pmin(pmax(eta, 0), 1)
  Y <- 2*rbinom(n, 1, eta)-1
  data.frame(Y = factor(Y), X)
}

generate_nonlinear_test <- function(N_test, d,alpha) {
  X <- matrix(runif(N_test*d, -1, 1),nrow = N_test)
  boundary_signal <- X[,1] + X[,2]^2
  eta <- 0.5 + 0.5 * sign(boundary_signal) *  abs(boundary_signal)^alpha
  eta <- pmin(pmax(eta, 0), 1)
  Y <- 2*rbinom(N_test, 1, eta)-1
  list(X = X, Y = Y)
}

# modified empirical bayes risk function depending on type of decision boundary
estimate_bayes_risk <- function(test_data, boundary_type = "linear") {
  if(boundary_type == "linear")  bayes_pred <- ifelse(test_data$X[,1] > 0,1,-1)
  else   bayes_pred <- ifelse(test_data$X[,1] + test_data$X[,2]^2 > 0,1,-1)
  mean(bayes_pred != test_data$Y)
}

# main simulation engine
run_simulation <- function(generator, test_generator, boundary_type = "linear",
  alpha_grid = c(0.5,1,2,4), n_grid = c(100,250,500), d = 5,  repetitions = 50,
  extension_name = "Unknown") {
  results <- data.frame()
  N_test <- 50000
  for(alpha in alpha_grid) {
    test_data <- test_generator(N_test, d, alpha)
    #empirical risk under bayes classifier
    bayes_err <- estimate_bayes_risk(test_data, boundary_type)
    for(n in n_grid) {
      cat("Running:", extension_name, "alpha=", alpha, "n=", n,"\n")
      for(rep in 1:repetitions) {
        train_data <- generator(n,d,alpha)
        logit_err <- run_logistic(train_data, test_data)
        knn_err <- run_knn(train_data, test_data)
        svm_err <- run_svm(train_data, test_data)
        rf_err <- run_rf( train_data, test_data)
        temp <- data.frame(alpha = alpha,n = n,rep = rep,
                           method = c("Logistic", "kNN", "SVM", "RF"),
                           risk = c(logit_err, knn_err, svm_err, rf_err),
                           excess_risk = c(logit_err,knn_err, svm_err,rf_err) - bayes_err,
                           extension =  extension_name)
        results <- rbind(results, temp)
      }
    }
  }
  results
}

# excess risk for uniform predictor
results_uniform <- run_simulation(
  generator = generate_data,
  test_generator = generate_test_set,
  extension_name = "Uniform")

# extension to heavy tailed data
results_heavy <- run_simulation(
  generator =  generate_heavy_tail_data,
  test_generator = generate_heavy_tail_test,
  extension_name = "HeavyTail")

# extension to correlated gaussian
results_corr <- run_simulation(
  generator = generate_correlated_data,
  test_generator =  generate_correlated_test,
  extension_name = "Correlated Gaussian")

# extension to non-linear boundary
results_nonlinear <- run_simulation(
  generator = generate_nonlinear_boundary_data,
  test_generator =  generate_nonlinear_test,
  boundary_type = "nonlinear",
  extension_name =  "NonlinearBoundary")

all_results <- rbind(results_uniform, results_heavy, results_corr, results_nonlinear)

summary_results <-
  all_results |>
  group_by(extension,alpha,n,method) |>
  summarise(mean_excess = pmax(mean(excess_risk),1e-5))
# so that the logarithm is valid in case of non-positive empirical excess risk

plt <- ggplot(summary_results, aes(x = n, y = mean_excess, color = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  facet_grid(extension ~ alpha, scales = "free_y") +
  scale_x_log10() +
  scale_y_log10() +
  theme_bw() +
  labs(title = "Excess Risk under varying margins, geometric conditions, sample sizes",
    y = "Mean Empirical Excess Risk")+
  theme(plot.title = element_text(size = 10))

ggsave(filename = "./tex_files/plots/excess_risk_plots.pdf", plot = plt, width = 14, height = 10)
