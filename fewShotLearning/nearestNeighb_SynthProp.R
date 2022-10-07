library(FNN)


library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))

library(data.table)

train_prod_7 <- fread(file="PreprocessNoPosteriorData/SmallGeneticMassSpecSpectraOnly_TRAIN.csv", data.table = F)

nrow(train_prod_7)
head(train_prod_7)

# Separate out positive and negative records
positive <- train_prod_7[train_prod_7$`Beta-lactamPhenoFamily` == 1,]
negative <- train_prod_7[train_prod_7$`Beta-lactamPhenoFamily` == 0,]

dim(positive) #59 x 3410
dim(negative) #43 x 3410


KNN_Model = knn(negative[1:(length(negative)-1)],positive[1:(length(positive)-1)],negative$`Beta-lactamPhenoFamily`,k=5)
nn_idx <- attr(KNN_Model, "nn.index")


#index back to the negative subset
nearest_neighbors_neg <- negative[nn_idx, ]  #295 neighbours - the number is larger than sample size due to duplicate neighbours
dim(nearest_neighbors_neg)

#oversample positive, then vertically concatenate with negative neigbhours then oversample again
oversample_pos <- rbind(positive, positive, positive, positive, positive)  #59 x 5 = 295

pos_neg_overample <- rbind(oversample_pos, nearest_neighbors_neg)
dim(pos_neg_overample)  #590 x 3410

#no need for this, only useful for neural network which need large number of samples
#train_prod_7_oversample <- rbind(pos_neg_overample, pos_neg_overample, pos_neg_overample, pos_neg_overample, pos_neg_overample)
#dim(train_prod_7_oversample)

#Synthprop
library("synthpop")
source("syn_strata2.r")
source("syn2.r")
library(stringr)
source("functions.syn2.r")
source("padMis.syn.r")
source("padModel.syn.r")
source("sampler.syn2.r")
library(randomForest)

my.seed <- 7

#Rename class as y since this is required for formula in randomForest y ~ .
colnames(pos_neg_overample)[colnames(pos_neg_overample) == 'Beta-lactamPhenoFamily'] <- 'y'

#issue is with the columns' names starting with numbers in dataset
#rf.fit <- randomForest(y ~ ., data = pos_neg_overample, ntree = 10)

response_col <- which(colnames(pos_neg_overample) == "y")
# Fixing the issue by adding a character to the column names except response
colnames(pos_neg_overample)[-response_col] <- paste0( "V", colnames(pos_neg_overample)[-response_col])

#remember that it generates samples of same size as N samples size
sds.default <- syn.strata2(pos_neg_overample, method='rf', seed = my.seed, minstratumsize=200, strata=pos_neg_overample$y)

#takes approx. 90 min to perform for positive set 
#combine with original dataset once the negative samples have been sampled in the same manner

#to simplify things, we should initially use only the positive and the positive-like negative simulated samples 
#to augment existing dataset and evaluate the neural network performance

saveRDS(sds.default, "synthProp_seed_7.rds")

synthProp_seed_7 <- readRDS("synthProp_seed_7.rds")

#save synthetic data and labels

synthPropRF_seed_7<- cbind(synthProp_seed_7$syn, synthProp_seed_7$strata.syn)

write.csv(synthPropRF_seed_7, file="model_input/synthPropRF_seed_7.csv", row.names = F)

