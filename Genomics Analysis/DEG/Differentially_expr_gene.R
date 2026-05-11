# ====================================================================
# 差异表达基因分析
# ====================================================================
# 只对limma包的操作进行了完善，DESeq2"和"edgeR"包的差异表达分析使用原始
# COUNT，后续再完善这部分代码

rm(list = ls())
# 1. 加载安装包
required_packages <- c("DESeq2", "edgeR", "limma", "dplyr", "ggplot2", 
                       "pheatmap", "RColorBrewer", "VennDiagram", "readr",
                       "tibble", "stringr", "scales", "ggrepel", "readxl")

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      if (pkg %in% c("DESeq2", "edgeR", "limma")) {
        BiocManager::install(pkg)
      } else {
        install.packages(pkg)
      }
    }
    library(pkg, character.only = TRUE)
  }
}

# 加载包
install_and_load(required_packages)

# 2. 数据读取和预处理

# 基因表达矩阵，行为基因，列为样本
read_expression_matrix <- function(file_path) {
  cat("读取基因表达矩阵...\n")
  
  # 尝试不同的分隔符
  if (grepl("\\.xlsx$", file_path)) {
    expr_data <- read_excel(file_path)
    expr_data <- as.data.frame(expr_data)
    rownames(expr_data) <- expr_data[,1]
    expr_data <- expr_data[,-1]
  } else if (grepl("\\.csv$", file_path)) {
    expr_data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE)
  } else {
    expr_data <- read.table(file_path, header = TRUE, sep = "\t", 
                            stringsAsFactors = FALSE)
  }
  
  
  # TCGA 截取所有列名（样本ID）的前12个字符
  #colnames(expr_data) <- iconv(colnames(expr_data), from = "", to = "UTF-8")
  
  colnames(expr_data) <- substr(colnames(expr_data), 1, 12)
  
  # 确保每一列数据为数值型
  # expr_data <- as.data.frame(lapply(expr_data, function(x) {
  #   if (is.character(x)) as.numeric(x) else x
  # }))
  
  # 移除含有NA行
  expr_data <- expr_data[complete.cases(expr_data), ]
  
  cat("基因表达矩阵维度：", dim(expr_data), "（基因数×样本数）\n")
  cat("样本名称：", head(colnames(expr_data), 10), "...\n")
  
  return(expr_data)
}

# 样本分组信息
read_sample_groups <- function(file_path) {
  cat("读取样本分组文件...\n")
  
  if (grepl("\\.xlsx$", file_path)) {
    group_data <- read_excel(file_path)   # 读取 Excel
  } else if (grepl("\\.csv$", file_path)) {
    group_data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE)
  } else {
    group_data <- read.table(file_path, header = TRUE, sep = "\t", 
                             stringsAsFactors = FALSE)
  }
  group_data <- as.data.frame(group_data)
  group_data <- group_data[, c("patient_id", "hazard_group")]
  
  # 截取patient_id前12个字符
  group_data$patient_id <- substr(group_data$patient_id, 1, 12)
  
  # 确保分组信息为因子，即字符串映射成整数编码
  group_data$hazard_group <- as.factor(group_data$hazard_group)
  
  cat("样本分组信息：\n")
  print(table(group_data$hazard_group))
  
  return(group_data)
}

# 3. 数据匹配，质控

# 匹配表达矩阵和分组信息
match_samples <- function(expr_data, group_data) {
  cat("匹配样本信息...\n")
  
  # 共同样本
  common_samples <- intersect(colnames(expr_data), group_data$patient_id)
  
  if (length(common_samples) == 0) {
    stop("错误：表达矩阵和分组文件中没有共同的样本ID！")
  }
  
  cat("共同样本数量：", length(common_samples), "\n")
  
  expr_matched <- expr_data[, common_samples]
  
  group_matched <- group_data[group_data$patient_id %in% common_samples, ]
  
  # 确保表达矩阵与分组信息顺序一致
  group_matched <- group_matched[match(colnames(expr_matched), group_matched$patient_id), ]
  
  if (!all(colnames(expr_matched) == group_matched$patient_id)) {
    stop("错误：样本顺序不一致！")
  }
  
  cat("最终分析样本数：", ncol(expr_matched), "\n")
  cat("各组样本数：\n")
  print(table(group_matched$hazard_group))
  
  return(list(expression = expr_matched, groups = group_matched))
}

# 基因过滤
filter_genes <- function(expr_data, min_count = 0.5, min_samples = 4) {
  cat("进行基因过滤...\n")
  cat("过滤前基因数：", nrow(expr_data), "\n")           # 55308
  
  # 移除低表达基因
  keep_genes <- rowSums(expr_data >= min_count) >= min_samples
  expr_filtered <- expr_data[keep_genes, ]
  
  cat("过滤后基因数：", nrow(expr_filtered), "\n")
  
  return(expr_filtered)
}


# 4. 差异表达分析                

# DESeq2差异分析
run_deseq2_analysis <- function(expr_data, group_data) {
  cat("DESeq2进行差异分析...\n")
  
  # 表达数据为整数（DESeq2要求）
  expr_data <- round(expr_data)
  
  dds <- DESeqDataSetFromMatrix(countData = expr_data,
                                colData = data.frame(group = group_data$hazard_group),
                                design = ~ group)
  
  # 差异分析
  dds <- DESeq(dds)
  
  res <- results(dds)

  res_df <- as.data.frame(res)
  res_df$gene_id <- rownames(res_df)
  
  # 重新排列列
  res_df <- res_df[, c("gene_id", "baseMean", "log2FoldChange", "lfcSE", 
                       "stat", "pvalue", "padj")]
  
  # 按调整后p值排序
  res_df <- res_df[order(res_df$padj), ]
  
  return(res_df)
}

# limma差异分析
run_limma_analysis <- function(expr_data, group_data) {
  cat("正在使用limma进行差异分析...\n")
  
  # log2转换
  # if (max(expr_data, na.rm = TRUE) > 50) {
    # expr_data <- log2(expr_data + 1)
  # }
  
  design <- model.matrix(~ group_data$hazard_group)
  colnames(design) <- c("Intercept", "Group1_vs_Group0")
  
  # 拟合线性模型
  # 为每个基因构建一个线性模型，Y_g = Xβ_g + ε_g，Y_g为基因g在所有样本中的表达量向量，X是设计矩阵（包含分组信息），
  # β_g是基因g的系数向量（如截距、组间差异等），ε_g是服从正态分布的误差项
  fit <- lmFit(expr_data, design)   # expr_data每一列样本排列顺序和design每一行样本排列顺序要相同
  fit <- eBayes(fit)
  
  # limma计算logFC=实验组平均值(死) - 对照组平均值(活)，在此之前必须做log2转换
  res <- topTable(fit, coef = 2, number = Inf, sort.by = "P")  # sort.by = "adj.P"
  
  res$gene_id <- rownames(res)
  res <- res[, c("gene_id", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
  
  return(res)
}

# edgeR差异分析
run_edger_analysis <- function(expr_data, group_data) {
  cat("正在使用edgeR进行差异分析...\n")
  
  # 表达数据为整数
  expr_data <- round(expr_data)
  
  y <- DGEList(counts = expr_data, group = group_data$hazard_group)
  
  # 归一化
  y <- calcNormFactors(y)
  
  # 离散度
  y <- estimateDisp(y)
  
  # 精确检验
  et <- exactTest(y)
  
  res <- topTags(et, n = Inf)$table
  res$gene_id <- rownames(res)
  res <- res[, c("gene_id", "logFC", "logCPM", "PValue", "FDR")]
  
  return(res)
}

# 5. 结果                  

# 筛选差异表达基因
filter_deg <- function(results, pval_cutoff = 0.05, fc_cutoff = 2) {
  cat("差异表达基因筛选条件：调整后p值 <", pval_cutoff, "，绝对倍数变化 >", fc_cutoff, "\n")
  
  # 根据不同方法的结果格式进行筛选
  if ("padj" %in% colnames(results)) {
    # DESeq2结果
    deg <- results[!is.na(results$padj) & 
                     results$padj < pval_cutoff & 
                     abs(results$log2FoldChange) > log2(fc_cutoff), ]
    
    # 添加方向标签
    deg$direction <- ifelse(deg$log2FoldChange > 0, "Up", "Down")
    
  } else if ("adj.P.Val" %in% colnames(results)) {
    # limma结果
    deg <- results[!is.na(results$adj.P.Val) & 
                     results$adj.P.Val < pval_cutoff & 
                     abs(results$logFC) > log2(fc_cutoff), ]
    
    # 添加方向标签
    # 上一步以logFC为1过滤，则logFC> 0的就是大于1
    deg$direction <- ifelse(deg$logFC > 0, "Up", "Down")
    
  } else if ("FDR" %in% colnames(results)) {
    # edgeR结果
    deg <- results[!is.na(results$FDR) & 
                     results$FDR < pval_cutoff & 
                     abs(results$logFC) > log2(fc_cutoff), ]
    
    # 添加方向标签
    deg$direction <- ifelse(deg$logFC > 0, "Up", "Down")
  }
  
  cat("差异表达基因数量：", nrow(deg), "\n")
  cat("上调基因：", sum(deg$direction == "Up"), "\n")
  cat("下调基因：", sum(deg$direction == "Down"), "\n")
  
  return(deg)
}

# 6. 可视化    

# 火山图
# 将差异表达基因（经padj和logFC筛选）可视化，并凸显上调和下调DEGs
plot_volcano <- function(results, title = "Volcano Plot") {
  if ("padj" %in% colnames(results)) {
    # DESeq2结果
    plot_data <- data.frame(
      gene_id = results$gene_id,
      logFC = results$log2FoldChange,
      pvalue = -log10(results$padj),
      stringsAsFactors = FALSE
    )
  } else if ("adj.P.Val" %in% colnames(results)) {
    # limma结果
    plot_data <- data.frame(
      gene_id = results$gene_id,
      logFC = results$logFC,
      pvalue = -log10(results$adj.P.Val),        # -log10(0.05) ≈ 1.3  
      stringsAsFactors = FALSE
    )
  } else if ("FDR" %in% colnames(results)) {
    # edgeR结果
    plot_data <- data.frame(
      gene_id = results$gene_id,
      logFC = results$logFC,
      pvalue = -log10(results$FDR),
      stringsAsFactors = FALSE
    )
  }
  plot_data <- plot_data[is.finite(plot_data$pvalue), ]
  
  plot_data$color <- "NS"
  plot_data$color[plot_data$logFC > log2(1.2) & plot_data$pvalue > -log10(0.05)] <- "Up"
  plot_data$color[plot_data$logFC < -log2(1.2) & plot_data$pvalue > -log10(0.05)] <- "Down"
  
  # 火山图
  p <- ggplot(plot_data, aes(x = logFC, y = pvalue, color = color)) +
    geom_point(alpha = 0.6, size = 0.8) +
    scale_color_manual(values = c("Up" = "red", "Down" = "blue", "NS" = "gray")) +
    geom_vline(xintercept = c(-log2(1.2), log2(1.2)), linetype = "dashed", alpha = 0.5) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", alpha = 0.5) +
    labs(x = "Log2 Fold Change", y = "-Log10 Adjusted P-value", title = title) +
    theme_minimal() +                                 # 换主题
    theme(legend.title = element_blank())       
  
  return(p)
}


# MA图
plot_ma <- function(results, title = "MA Plot") {
  if ("padj" %in% colnames(results)) {
    # DESeq2结果
    plot_data <- data.frame(
      baseMean = log10(results$baseMean + 1),
      logFC = results$log2FoldChange,
      significant = results$padj < 0.05 & !is.na(results$padj),
      stringsAsFactors = FALSE
    )
  } else if ("adj.P.Val" %in% colnames(results)) {
    # limma结果
    plot_data <- data.frame(
      baseMean = results$AveExpr,
      logFC = results$logFC,
      significant = results$adj.P.Val < 0.05 & !is.na(results$adj.P.Val) & abs(results$logFC) > log2(1.2),
      stringsAsFactors = FALSE
    )
  } else if ("FDR" %in% colnames(results)) {
    # edgeR结果
    plot_data <- data.frame(
      baseMean = results$logCPM,
      logFC = results$logFC,
      significant = results$FDR < 0.05 & !is.na(results$FDR),
      stringsAsFactors = FALSE
    )
  }
  
  # MA图
  p <- ggplot(plot_data, aes(x = baseMean, y = logFC, color = significant)) +
    geom_point(alpha = 0.6, size = 0.8) +
    scale_color_manual(values = c("FALSE" = "gray", "TRUE" = "red")) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    labs(x = "Mean Expression", y = "Log2 Fold Change", title = title) +
    theme_minimal() +
    theme(legend.title = element_blank())
  
  return(p)
}



# 主分析函数
run_deg_analysis <- function(expr_file, group_file, output_dir = "deg_results",
                             method = "limma", pval_cutoff = 0.05, fc_cutoff = 1.2) {

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  cat("=== 差异表达基因分析开始 ===\n")
  cat("方法：", method, "\n")
  cat("显著性阈值：", pval_cutoff, "\n")
  cat("倍数变化阈值：", fc_cutoff, "\n\n")
  
  expr_data <- read_expression_matrix(expr_file)
  group_data <- read_sample_groups(group_file)    # only 两列数据
  
  # 取共同样本，且样本顺序一致(expr_data顺序)，返回列表(expression, groups)
  matched_data <- match_samples(expr_data, group_data)
  
  # 过滤低表达基因
  filtered_expr <- filter_genes(matched_data$expression, min_count = 0.5, min_samples = 4)
  
  if (method == "DESeq2") {
    deg_results <- run_deseq2_analysis(filtered_expr, matched_data$groups)
  } else if (method == "limma") {
    deg_results <- run_limma_analysis(filtered_expr, matched_data$groups)
  } else if (method == "edgeR") {
    deg_results <- run_edger_analysis(filtered_expr, matched_data$groups)
  } else {
    stop("请选择：DESeq2, limma, 或 edgeR")
  }
  
  deg_filtered <- filter_deg(deg_results, pval_cutoff, fc_cutoff)
  # 所有结果
  write.csv(deg_results, file.path(output_dir, paste0(method, "_deganaly_all_results.csv")), row.names = FALSE)
  
  # 差异基因的统计量
  write.csv(deg_filtered, file.path(output_dir, paste0(method, "_degs_results.csv")), row.names = FALSE)
  
  # 差异基因ID列表
  write.table(deg_filtered$gene_id, file.path(output_dir, paste0(method, "_degs_ids.txt")), 
              row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # 火山图
  volcano_plot <- plot_volcano(deg_results, paste(method, "Volcano Plot"))
  ggsave(file.path(output_dir, paste0(method, "_volcano_plot.png")), 
         volcano_plot, width = 8, height = 6, dpi = 300)
  
  # MA图
  ma_plot <- plot_ma(deg_results, paste(method, "MA Plot"))
  ggsave(file.path(output_dir, paste0(method, "_ma_plot.png")), 
         ma_plot, width = 8, height = 6, dpi = 300)
  
  cat("生成分析报告...\n")
  
  report <- paste(
    "=== 差异表达基因分析报告 ===",
    paste("分析方法：", method),
    paste("分析时间：", Sys.time()),
    paste("输入基因数：", nrow(expr_data)),
    paste("过滤后基因数：", nrow(filtered_expr)),
    paste("基因表达数据中样本总数：", ncol(matched_data$expression)),
    paste("分组信息中样本数：", paste(table(matched_data$groups$hazard_group), collapse = ", ")),
    paste("调整后显著性阈值：", pval_cutoff),
    paste("logFC阈值：", log2(fc_cutoff)),
    paste("差异表达基因总数：", nrow(deg_filtered)),
    paste("上调基因数：", sum(deg_filtered$direction == "Up")),
    paste("下调基因数：", sum(deg_filtered$direction == "Down")),
    "",
    "=== 输出文件 ===",
    paste("所有基因差异分析结果：", paste0(method, "_deganaly_all_results.csv")),
    paste("差异表达基因结果：", paste0(method, "_degs_results.csv")),
    paste("差异基因ID：", paste0(method, "_degs_ids.txt")),
    paste("火山图：", paste0(method, "_volcano_plot.png")),
    paste("MA图：", paste0(method, "_ma_plot.png")),
    sep = "\n"
  )
  
  writeLines(report, file.path(output_dir, "analysis_report.txt"))
  
  cat("=== 分析完成 ===\n")
  cat("结果已保存到：", output_dir, "\n")
  
  return(list(
    all_results = deg_results,
    deg_list = deg_filtered,
    matched_data = matched_data,
    filtered_expression = filtered_expr
  ))
}





# 实例化
results <- run_deg_analysis(
  expr_file = "C:/Users/Admin/Desktop/work/骨肉瘤/all_patients_counts.xlsx",        # 基因表达矩阵文件
  group_file = "C:/Users/Admin/Desktop/work/骨肉瘤/all_data_prediction.xlsx",  # 样本分组文件
  output_dir = "C:/Users/Admin/Desktop/work/骨肉瘤/deg_results",                 # 输出目录
  method = "edgeR",                           # 分析方法：DESeq2, limma, edgeR
  pval_cutoff = 0.05,                         
  fc_cutoff = 1.2                             
)

ehead(results$deg_list)

# 比较多种方法
# deseq2_results <- run_deg_analysis(expr_file, group_file, "deseq2_results", "DESeq2")
# limma_results <- run_deg_analysis(expr_file, group_file, "limma_results", "limma")
# edger_results <- run_deg_analysis(expr_file, group_file, "edger_results", "edgeR")
