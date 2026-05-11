# -------------------------- GSVA 2023.6.13 --------------------------------------
cat("\014"); rm(list = ls()); options(stringsAsFactors = F); options(warn = -1)
library(limma);library(GSEABase);library(GSVA);library(pheatmap)
# 新增读取Excel的包
library(readxl) 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# 定义一个简易函数：读取xlsx并将第1列设为行名 (模拟 read.csv(row.names=1))
read_xlsx_to_mat <- function(path) {
  dat <- as.data.frame(read_excel(path))
  rownames(dat) <- dat[,1]  # 第1列设为行名
  dat <- dat[,-1]           # 删除第1列
  return(as.matrix(dat))
}

# 修改部分：使用自定义函数读取xlsx
# fpkm = read_xlsx_to_mat("Inter-tr-ComBat_TME_Hub_GGH.xlsx"); co=1
fpkm = read_xlsx_to_mat("C:/Users/Admin/Desktop/work/骨肉瘤/FPKM_result.xlsx"); co=1

# ----------- log2 TPM ------------------------------
fpkmToTpm <- function(fpkm){exp(log(fpkm+1) - log(sum(fpkm)+1) + log(1e6))}
tpm <- log2(apply(fpkm,2,fpkmToTpm) + 1)

Key.gene = read.table('C:/Users/Admin/Desktop/work/骨肉瘤/湘雅附2的基因分析/GSVA/total.txt')
# 确保 key gene 在 tpm 中存在，防止报错
valid_genes <- intersect(Key.gene$V1, rownames(tpm))
Key.tpm = tpm[valid_genes, ]


# ---------- GSVA ----------------
gsva_C2   = getGmt("C:/Users/Admin/Desktop/work/骨肉瘤/湘雅附2的基因分析/GSVA/c2.all.v7.5.1.symbols.gmt")
gsva_C5   = getGmt("C:/Users/Admin/Desktop/work/骨肉瘤/湘雅附2的基因分析/GSVA/c5.all.v7.5.1.symbols.gmt")
gsva_HALL = getGmt("C:/Users/Admin/Desktop/work/骨肉瘤/湘雅附2的基因分析/GSVA/h.all.v7.5.1.symbols.gmt")

# 注意：GSVA 新版本参数可能有变，如果你用的是新版 GSVA (1.50+)，parallel.sz 改为 kcdf="Gaussian" 等
param_C2   <- gsvaParam(exprData = tpm,geneSets = gsva_C2,minSize = 5,maxSize = 500,maxDiff = TRUE,kcdf = "Gaussian")
param_C5   <- gsvaParam(exprData = tpm,geneSets = gsva_C5,minSize = 5,maxSize = 500,maxDiff = TRUE,kcdf = "Gaussian")
param_HALL   <- gsvaParam(exprData = tpm,geneSets = gsva_HALL,minSize = 5,maxSize = 500,maxDiff = TRUE,kcdf = "Gaussian")
rgtGsvaC2 <- gsva(param_C2, verbose=FALSE)
rgtGsvaC5 <- gsva(param_C5, verbose=FALSE)
rgtGsvaHALL <- gsva(param_HALL, verbose=FALSE)


rgtGsvaAll = rbind(rgtGsvaC2, rgtGsvaC5, rgtGsvaHALL)
rgtGsvaAllPathway = data.frame(rgtGsvaAll)
print(paste0('Num:', dim(rgtGsvaAllPathway)[1]))
if(co==1){write.csv(rgtGsvaAllPathway, file = "path train.csv", row.names = T)}
if(co==2){write.csv(rgtGsvaAllPathway, file = "path test.csv", row.names = T)}
