
# coding: utf-8

# In[1]:

import pandas as pd
import os
import numpy as np
from collections import defaultdict
import rpy2.robjects as robjects
import time
from rpy2.robjects.packages import importr
import sys



#new
importr('plyr')
robjects.r('''
library("plyr")
# ------------------------------------------------------------------------------------
# Description: AZ-Sanger Challenge scoring functions
# Authors: Michael P Menden, Julio Saez-Rodriguez, Mike Mason, Thomas Yu
# ------------------------------------------------------------------------------------
# Get observation format for Subchallenge 1
getObs_ch1 <- function(ls) {
  return(data.frame(CELL_LINE=as.character(ls$CELL_LINE),
                    COMBINATION_ID=as.character(ls$COMBINATION_ID),
                    OBSERVATION=ls$SYNERGY_SCORE))
}

# Get the drug combinations score of Subchallenge 1
getDrugCombiScore_ch1 <- function(obs, pred, confidence=NA, topX=10) {
  R <- c()
  obs <- read.csv(obs,stringsAsFactors = F)
  obs <- getObs_ch1(obs)
  pred <- read.csv(pred,stringsAsFactors=F)
  pred <- pred[match(paste(obs$CELL_LINE,obs$COMBINATION_ID),paste(pred$CELL_LINE,pred$COMBINATION_ID)),]

  pred$COMBINATION_ID <- gsub(" ", "", pred$COMBINATION_ID)
  for (i in as.character(unique(obs$COMBINATION_ID))) {
      R <- c(R, cor(obs[obs$COMBINATION_ID == i, 'OBSERVATION'],
                    pred[pred$COMBINATION_ID == i, 'PREDICTION']))
  }
  #Make NA's in R = 0
  R[is.na(R)] = 0
  names(R) <- as.character(unique(obs$COMBINATION_ID))

  if (!file.exists(confidence))
    return(round(c(mean=mean(R),
             ste=sd(R),
             n=sum(!is.na(R))),2))

  confidence <- read.csv(confidence,stringsAsFactors=F)
  confidence <- confidence[match(unique(obs$COMBINATION_ID),confidence$COMBINATION_ID),]

  nStep <- 1000
  nVal <- round(topX * (length(R) / 100))
  boot_R <- rep(0, nVal)
  for (i in 1:nStep) {
    idx <- order(confidence$CONFIDENCE, sample(length(R)), decreasing = T)[1:nVal]
    boot_R <- boot_R + R[idx]
  }

  return(round(c(mean=mean(boot_R/nStep),
                 ste=sd(boot_R/nStep),
                 n=sum(!is.na(boot_R/nStep))),2))
}

# ------------------------------------------------------------------------------------
# Get the global score of Subchallenge 1
# ------------------------------------------------------------------------------------
getGlobalScore_ch1 <- function(obs, pred) {
  obs <- read.csv(obs, stringsAsFactors=F)
  obs <- getObs_ch1(obs)
  pred <- read.csv(pred,stringsAsFactors=F)
  pred <- pred[match(paste(obs$CELL_LINE,obs$COMBINATION_ID),paste(pred$CELL_LINE,pred$COMBINATION_ID)),]

  x = obs$OBSERVATION
  y = pred$PREDICTION

  agg <- aggregate(OBSERVATION ~ CELL_LINE, obs, median)
  z0 <- agg$OBSERVATION[match(obs$CELL_LINE, agg$CELL_LINE)]

  agg <- aggregate(OBSERVATION ~ COMBINATION_ID, obs, median)
  z1 <- agg$OBSERVATION[match(obs$COMBINATION_ID, agg$COMBINATION_ID)]

  parCor <- function(u,v,w) {
    numerator = cor(u,v) - cor(u,w) * cor(w,v)
    denumerator = sqrt(1-cor(u,w)^2) * sqrt(1-cor(w,v)^2)
    return(numerator/denumerator)
  }

  numerator=parCor(x,y,z1) - parCor(x,z0,z1) * parCor(z0,y,z1)
  denumerator= sqrt(1-parCor(x,z0,z1)^2) * sqrt(1-parCor(z0,y,z1)^2)
  # ----------------------------------------
  # sub challenge 1 FINAL SCORING metrics
  # ----------------------------------------
  temp   <- data.frame(OBSERVATION = x, PREDICTION = y, COMBINATION_ID = obs$COMBINATION_ID)
  R      <- ddply(temp, "COMBINATION_ID", function(x){if(length(x$OBSERVATION) > 1){return(cor(x$OBSERVATION,x$PREDICTION))}else{return(0)}})
  R[is.na(R[,2]),2] <- 0;
  R$N    <- table(obs$COMBINATION_ID)
  obsMax <- ddply(obs, "COMBINATION_ID", function(x){max(x$OBSERVATION)})

  primaryScoreCH1     <- sum(R$V1*sqrt(R$N-1))/sum(sqrt(R$N-1))                       # primary metric
  gt20Inds             <- obsMax[,2] >= 20
  tieBreakingScoreCH1  <- sum((R$V1*sqrt(R$N-1))[gt20Inds])/sum(sqrt(R$N-1)[gt20Inds]) # tie-breaking metric
  # ----------------------------------------

  # partial out the mean of synergy across cell lines and combinationations
  return(c(score=numerator/denumerator,
           final= primaryScoreCH1,
           tiebreak= tieBreakingScoreCH1))
}

''')


# filePath = "C:/Users/Jang/Desktop/deleteFeature_except/" #10000 result
filePath = "J4condor/result/" #10000 result
# basePath = "C:/Users/Jang/Downloads/baseline_exclude.csv"
basePath = "J3condor/result/baseline.csv"

baseline = []

df = pd.read_csv(basePath)
for baseValue in df['baseline'] :
    baseline.append(baseValue)

result = {}
for i in range(10):
    result[i] = []


for root, dirs, files in os.walk(filePath):
    for fname in files:

        cvFileName = os.path.join(root, fname)
        cvFileName_only_filename = cvFileName.split('/')[-1]

        zz = cvFileName_only_filename.split("svm_result")[1].split(".csv")[0]


        index = zz.split("_")[1]
        deleteIndex = zz.split("_")[0]


        testsetpath = "/home/ubuntu/data/answers/ch1_newtestset_wtest_"
        trainDir = testsetpath + str(index)+".csv"#answerSewt
        value = robjects.r['getGlobalScore_ch1'](trainDir,cvFileName)[1]

        sunKyu_Value = value - baseline[int(index)]

        l = result[int(index)]
        l.append(str(deleteIndex)+","+str(sunKyu_Value))
        result[int(index)] = l


round_num = sys.argv[1]
filePath = "data/" + round_num + "/J5condor/result/"
os.makedirs(filePath)
for i in range(10):
    f = open(filePath+"result_"+str(i)+".csv",'w')
    for line in result[i]:
        f.write(line+"\n")
        f.flush()
    f.close()
