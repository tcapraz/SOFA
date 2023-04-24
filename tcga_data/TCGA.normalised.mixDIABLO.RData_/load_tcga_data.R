library(IlluminaHumanMethylation450kanno.ilmn12.hg19)

setwd("/home/tuemay/supervised_matrix_factorization/tcga/TCGA.normalised.mixDIABLO.RData_")

load("/home/tuemay/supervised_matrix_factorization/tcga/TCGA.normalised.mixDIABLO.RData_/TCGA.normalised.mixDIABLO.RData")


ann450k <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
colnames(data.train$methylation) <- ann450k[colnames(data.train$methylation), "UCSC_RefGene_Name"]

write.csv(data.train$protein, "train_prot.csv")
write.csv(data.train$methylation, "train_meth.csv")
write.csv(data.train$mrna, "train_rna.csv")
write.csv(data.train$mirna, "train_mirna.csv")
write.csv(data.train$subtype, "train_y.csv")

colnames(data.test$methylation) <- ann450k[colnames(data.test$methylation), "UCSC_RefGene_Name"]
write.csv(data.test$protein, "test_prot.csv")
write.csv(data.test$methylation, "test_meth.csv")
write.csv(data.test$mrna, "test_rna.csv")
write.csv(data.test$mirna, "test_mirna.csv")
write.csv(data.test$subtype, "test_y.csv")
