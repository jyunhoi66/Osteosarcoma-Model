library(car)
library(survival)
library(glmnet)
library(survC1)
library(MASS)

rm(list = ls())
best_aic <- Inf
best_model <- NULL

# 读取数据
data <- read.csv("C:/Users/Admin/Desktop/work/骨肉瘤/影像组学特征筛选/radiomics_features.csv")
# 需要移除的列
col_to_remove = c('patient_id','event','OS')
# 从原始数据中提取 'event_status' 和 'survival_time' 列，用于后续的拼接
event_status <- data$event
survival_time <- data$OS
# 去除相应的列
data_value <- data[, !(names(data) %in% col_to_remove)]

# Z-scores 标准化
data_value_scores <- scale(data_value)
# 查找全是NA的列
na_columns <- colSums(is.na(data_value_scores)) == nrow(data_value_scores)
# 去除全是NA的列
data_value_scores <- data_value_scores[, !na_columns]

data_matrix <- cbind(event_status,survival_time,data_value_scores)
# cox模型要求data必须是数据框
data_matrix <- as.data.frame(data_matrix)


####---------------------------------------单变量分析---------------------------------------####

# 筛选出p 值小于等于 0.05 的特征
selected_features <- c()
# 第1列是事件状态，第2列是生存时间
for (feature in colnames(data_matrix)[-c(1,2)]) { #[-c(1,2)]
  formula <- as.formula(paste("Surv(survival_time, event_status) ~", feature))
  model_univ <- coxph(formula, data = data_matrix)

  summary_model <- summary(model_univ)

  p_value <- summary_model$coefficients[1, 5]  # p 值

  if (p_value <= 0.05) {
    selected_features <- c(selected_features, feature)
  }
}
# 输出筛选后剩下的特征数量
print(length(selected_features))

# 筛选后的数据,     p值都大于0.05
data_matrix_filtered <- data_matrix[, c('survival_time', 'event_status', selected_features)]



n = 0
# 运行30次
for (i in 1:30) {
  n = n + 1
  print(n)
  
  ####-----------------------------lasso-cox feature selection-----------------------------####
  
  # 转换为矩阵格式
  x <- as.matrix(data_matrix_filtered[, selected_features])
  y <- with(data_matrix_filtered, Surv(survival_time, event_status))
  
  fit <- glmnet(x, y, family = "cox", alpha=1)

  cv_fit <- cv.glmnet(x, y, family = "cox", alpha=1,nfolds=10)
  plot(cv_fit)
  lambda_min <- cv_fit$lambda.min
  important_features <- coef(cv_fit, s = "lambda.min")
  
  # 输出重要的特征
  # print(important_features)
  feature_index <- which(as.numeric(important_features) != 0)
  
  # 回归系数不为0的特征名和特征回归系数
  feature_coef <- as.numeric(important_features)[feature_index]
  feature_name <- rownames(important_features)[feature_index]
  len_feature_name <- paste("lasso-cox选择的特征数量是：", length(feature_name))
  print(length(feature_name))
  
  
  
  ####-------------------------------------AIC 逐步回归-------------------------------------####
  
  final_model <- coxph(Surv(survival_time, event_status) ~ ., 
                       data = data_matrix_filtered[, c('survival_time', 
                                                       'event_status', feature_name)])
  
  step_model <- stepAIC(final_model, direction = "both", trace = 0) 
  
  # 检查是否是迄今为止最好的AIC
  current_aic <- AIC(step_model)
  if (current_aic < best_aic) {
    best_aic <- current_aic
    best_model <- step_model
  }
}
  
  
# 输出选出的最重要的特征
  # print(summary(step_model))

# 输出AIC最小的模型的摘要，包括选出的最重要的特征
summary_best_model <- summary(best_model)
print(summary_best_model)
# 打印最优模型的AIC
print(best_aic)

# 需要：install.packages("survminer")
library(survminer)

# 用最优 Cox 模型得到线性预测值并分组
lp <- predict(best_model, type = "lp")  # 对 data_matrix_filtered 里行生效
grp <- ifelse(lp > median(lp, na.rm = TRUE), "High", "Low")

fit_km <- survfit(Surv(survival_time, event_status) ~ grp, data = data_matrix_filtered)

ggsurvplot(
  fit_km, data = data_matrix_filtered, pval = TRUE, conf.int = TRUE,
  legend.title = "Risk group", legend.labs = c("High", "Low"),
  xlab = "Time", ylab = "Survival probability", palette = c("red", "blue")
)


#####
lp  <- as.numeric(predict(best_model, type = "lp"))   # 在构建 best_model 时所用数据上打分
grp <- ifelse(lp > median(lp, na.rm = TRUE), "High Risk", "Low Risk")
grp <- factor(grp, levels = c("High Risk", "Low Risk"))

plot_df <- within(data_matrix_filtered, { risk_group <- grp })

# 2) KM 拟合与 HR（High vs Low）
fit_km <- survfit(Surv(survival_time, event_status) ~ risk_group, data = plot_df)
fit_hr <- coxph(Surv(survival_time, event_status) ~ risk_group, data = plot_df)
hr_sm  <- summary(fit_hr)
HR     <- unname(hr_sm$conf.int[1, "exp(coef)"])
HR_LCL <- unname(hr_sm$conf.int[1, "lower .95"])
HR_UCL <- unname(hr_sm$conf.int[1, "upper .95"])
HR_lbl <- sprintf("HR = %.2f (95%% CI: %.2f–%.2f)", HR, HR_LCL, HR_UCL)

# 为左下角注释准备一个相对位置
maxT <- max(plot_df$survival_time, na.rm = TRUE)
ann_x <- 0.08 * maxT   # 横坐标靠左
ann_y1 <- 0.14         # HR 行
ann_y2 <- 0.10         # p 值行（ggsurvplot 会自动计算 p 值；这里把位置也放左下角）

# 3) 颜色与主题（色盲友好：红/蓝）
pal <- c("#D55E00", "#56B4E9")  # High=红, Low=蓝
base_size <- 15

p <- ggsurvplot(
  fit_km, data = plot_df,
  conf.int = TRUE,                         # 置信带
  censor.shape = 124, censor.size = 2.5,   # 删失标记
  surv.median.line = "hv",                 # 中位生存时间虚线（水平+垂直）
  pval = TRUE,                             # Log-rank p 值
  pval.size = 5,
  pval.coord = c(ann_x, ann_y2),           # 左下角
  risk.table = "abs_pct",                  # 风险表：绝对数(百分比)
  risk.table.title = "Patients at Risk",
  risk.table.height = 0.26,
  risk.table.y.text.col = TRUE,
  legend.title = "Risk Group",
  legend.labs  = c("High Risk", "Low Risk"),
  palette = pal,
  size = 1.4,                              # 曲线粗细
  linetype = 1,
  xlab = "Survival Time (Months)",
  ylab = "Survival Probability",
  ggtheme = theme_classic(base_size = base_size)
)

# 4) 顶部横向图例、坐标轴加粗、标题置中；添加 HR 文本
p$plot <- p$plot +
  ggtitle("Kaplan-Meier Survival Curve by Risk Group") +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.title = element_text(size = base_size, face = "plain"),
    legend.text  = element_text(size = base_size - 1),
    axis.line  = element_line(linewidth = 0.8),
    axis.ticks = element_line(linewidth = 0.8),
    plot.title = element_text(hjust = 0.5, face = "plain")
  ) +
  annotate("text", x = ann_x, y = ann_y1, label = HR_lbl, hjust = 0, vjust = 1, size = 5)

# 风险表去网格、字号协调
p$table <- p$table +
  theme_classic(base_size = base_size - 1) +
  theme(
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )

# 5) 展示
print(p)

# 6) 大尺寸导出（与示例类似宽屏）：12x10 英寸，300/600 dpi

ggsave("/Users/jyunhoiwong/Desktop/work/chenshuting_CD8/KM_curve.png",  plot = p,
       width = 12, height = 10, units = "in", dpi = 300)


# 1) 使用 glmnet 生成系数路径图
# 基于训练数据的 LASSO 拟合模型 `fit` 和交叉验证模型 `cv_fit`
# 画 LASSO 路径图：系数如何随 lambda 的变化而变化

# 对 `fit`（LASSO Cox 模型）画路径图
plot(fit, xvar = "lambda", label = TRUE)

# 2) 对交叉验证的最优lambda标出路径
abline(v = log(cv_fit$lambda.min), col = "red", lwd = 2, lty = 2) # 红线表示最优lambda

# 3) 设置图形标题
title("LASSO Cox Coefficient Path")

# 如果你希望调整图形的大小或者进行保存
# ggsave("lasso_path.png", plot = last_plot(), width = 8, height = 6, dpi = 300)
