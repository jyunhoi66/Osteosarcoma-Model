# ------------------ CYT-TIL-IFN ------------------------------------
cat("\014"); rm(list = ls()); options(warn = -1); options(digits=3); 
library(GSVA); library(limma); library(GSEABase)

# 新增 Excel 读写包
library(readxl)
#library(writexl)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# -----------------------------------------
# 定义简易函数：读取 xlsx 并将第1列设为行名
read_xlsx_data <- function(path) {
  dat <- as.data.frame(read_excel(path))
  rownames(dat) <- dat[,1]  # 第1列设为行名
  dat <- dat[,-1]           # 删除第1列
  return(dat)
}

# 修改 1: 读取 xlsx 文件 (假设你的文件名也改成了 .xlsx)
FPKM = read_xlsx_data("C:/Users/Admin/Desktop/work/骨肉瘤/FPKM_result.xlsx")
FPKM[which(FPKM<0)] = 0

# ----------- log2 TPM------------------------------
fpkmToTpm <- function(fpkm){exp(log(fpkm+1) - log(sum(fpkm)+1) + log(1e6))}
TPM <- log2(apply(FPKM,2,fpkmToTpm)+1); min(TPM)


gmtFile = "28immune cell.gmt"
geneSet = getGmt(gmtFile, geneIdType=SymbolIdentifier())

#ssgsea
param <- ssgseaParam(exprData = as.matrix(TPM), 
                     geneSets = geneSet,
                     normalize = TRUE)
ssgseaScore <- gsva(param, verbose=FALSE)

#ssGSEA score
normalize=function(x){
  return((x-min(x))/(max(x)-min(x)))}

#ssGSEA score
ssgseaOut=normalize(ssgseaScore)

# 将列名(样本名)作为第一行拼接到矩阵中
ssgseaOut=rbind('p_ID'=colnames(ssgseaOut), ssgseaOut)

# 修改 2: 保存为 xlsx 文件
# 转置并转为数据框，write_xlsx 会自动保存列名(即这里的p_ID和通路名)

write.csv(t(ssgseaOut), file=paste0(gmtFile, ".csv"), row.names = F)
#write_xlsx(out_df, path = paste0(gmtFile, ".xlsx"))
