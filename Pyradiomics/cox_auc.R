cat("\014"); rm(list = ls()); options(warn = -1); options(digits=3); 
# 1. 加载必要的库
library(survival)
library(pROC)   # 用于计算AUC和画ROC曲线
library(caret)  # 用于计算Confusion Matrix, ACC, F1

# 2. 读取数据 (请修改为你的实际文件路径)
# 务必使用验证集或原始数据集
file_path <- "C:/Users/Admin/Desktop/work/骨肉瘤/湘雅附2影像组学特征/radiomics_selected_features.csv"  
data <- read.csv(file_path)

# 3. 定义特征和权重 (来自你的 Lasso-Cox 结果)
feature_names <- c(
  "original_glrlm_LongRunLowGrayLevelEmphasis",
  "exponential_glszm_SizeZoneNonUniformityNormalized",
  "gradient_firstorder_Energy",
  "wavelet.LHH_glrlm_HighGrayLevelRunEmphasis",
  "wavelet.LHH_glszm_SizeZoneNonUniformity",
  "wavelet.HHH_glszm_ZonePercentage",
  "wavelet.LLL_firstorder_Skewness"
)

fixed_betas <- c(0.28440, -0.21280, 0.23297, 0.21061, -0.34356, 0.31515, -0.22907)

# 生存时间界限 (3年 = 36个月)
cutoff_time <- 1095 

# ================= 2. 数据准备 =================
data_model <- data[, c("OS", "event", feature_names)]

# ================= 3. 五折交叉验证流程 =================
set.seed(123) # 固定随机种子
folds <- createFolds(data_model$event, k = 5, list = TRUE, returnTrain = FALSE)

# 结果容器
results <- data.frame(
  Fold = integer(),
  AUC = numeric(),
  Accuracy = numeric(),
  F1 = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric()
)

for (i in 1:5) {
  # --- A. 划分数据 ---
  idx_val <- folds[[i]]
  train_set <- data_model[-idx_val, ] # 这里的"Train"仅用于计算均值和阈值
  val_set   <- data_model[idx_val, ]  # 验证集，用于出结果
  
  # --- B. 标准化 (重要：使用非验证集的数据来标准化验证集) ---
  train_x <- as.matrix(train_set[, feature_names])
  val_x   <- as.matrix(val_set[, feature_names])
  
  # 计算标准化参数 (均值/方差)
  preproc <- preProcess(train_x, method = c("center", "scale"))
  
  # 应用标准化
  train_x_scaled <- predict(preproc, train_x)
  val_x_scaled   <- predict(preproc, val_x)
  
  # --- C. 计算 Radscore (直接使用固定权重) ---
  # 不需要 coxph 建模，直接矩阵相乘
  train_scores <- as.numeric(train_x_scaled %*% fixed_betas)
  val_scores   <- as.numeric(val_x_scaled %*% fixed_betas)
  
  # --- D. 构建二分类标签 (3年生存) ---
  # 标签定义: 1 = 短生存(<3年, 高风险), 0 = 长生存(>=3年, 低风险)
  
  # 处理 Train 集 (为了找阈值)
  train_labels <- ifelse(train_set$OS < cutoff_time, 1,
                         ifelse(train_set$OS >= cutoff_time, 0, NA))
  
  # 处理 Val 集 (为了验证)
  val_labels   <- ifelse(val_set$OS < cutoff_time, 1,
                         ifelse(val_set$OS >= cutoff_time, 0, NA))
  
  # 移除 NA (删失且时间不够的样本)
  valid_train_idx <- !is.na(train_labels)
  valid_val_idx   <- !is.na(val_labels)
  
  train_clean_scores <- train_scores[valid_train_idx]
  train_clean_labels <- train_labels[valid_train_idx]
  
  val_clean_scores   <- val_scores[valid_val_idx]
  val_clean_labels   <- val_labels[valid_val_idx]
  
  # --- E. 确定最佳阈值 (在 Train 上找) ---
  # 即使权重不用训，Cutoff 也不能直接用验证集找，否则算作弊
  roc_train <- roc(train_clean_labels, train_clean_scores, quiet = TRUE)
  coords_best <- coords(roc_train, "best", ret = "threshold", transpose = TRUE)
  best_cutoff <- coords_best[1]
  
  # --- F. 在验证集上计算指标 ---
  
  # 1. AUC (不需要阈值)
  roc_val <- roc(val_clean_labels, val_clean_scores, quiet = TRUE)
  auc_val <- auc(roc_val)
  
  # 2. ACC, F1 (需要阈值)
  pred_class <- ifelse(val_clean_scores > best_cutoff, 1, 0)
  
  # 转为因子
  pred_factor <- factor(pred_class, levels = c(0, 1))
  actual_factor <- factor(val_clean_labels, levels = c(0, 1))
  
  # 计算混淆矩阵
  cm <- confusionMatrix(pred_factor, actual_factor, mode = "everything", positive = "1")
  
  # --- G. 记录结果 ---
  results[i, "Fold"] <- i
  results[i, "AUC"] <- as.numeric(auc_val)
  results[i, "Accuracy"] <- cm$overall["Accuracy"]
  results[i, "F1"] <- cm$byClass["F1"]
  results[i, "Sensitivity"] <- cm$byClass["Sensitivity"]
  results[i, "Specificity"] <- cm$byClass["Specificity"]
  
  # 打印每折的简报
  cat(sprintf("Fold %d: AUC=%.3f, ACC=%.3f, F1=%.3f\n", i, auc_val, cm$overall["Accuracy"], cm$byClass["F1"]))
}

# ================= 4. 输出最终统计 =================
cat("\n========================================\n")
cat("   Fixed-Weights 5-Fold Validation Report   \n")
cat("========================================\n")

# 打印详细表格
print(results)

cat("\n--- Summary (Mean ± SD) ---\n")
metrics <- c("AUC", "Accuracy", "F1", "Sensitivity", "Specificity")

for (m in metrics) {
  mean_val <- mean(results[[m]], na.rm = TRUE)
  sd_val <- sd(results[[m]], na.rm = TRUE)
  cat(sprintf("%-12s: %.3f ± %.3f\n", m, mean_val, sd_val))
}
