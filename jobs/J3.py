
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import rpy2.robjects as robjects
import time
from rpy2.robjects.packages import importr
import sys
import numpy as np
import glob

# input
# resultFilePath: absolute path of the folder that result file exists
# testDirList: list. for each list index, the element must be the absolute path of
#               test set(csv file) with the exact same index
# paramPath: parameter will be written in the path
# baselinePath: baseline will be written in the path

round_num = sys.argv[1]
def findBestParam(resultFilePath, testDirList, paramPath, baselinePath):
    if not os.path.isdir("data/" + round_num + "/J3condor"):
        os.makedirs("data/" + round_num + "/J3condor")
    if not os.path.isdir("data/" + round_num + "/J3condor/result"):
        os.makedirs("data/" + round_num + "/J3condor/result")

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


    valuedict = {}
    indexvaluedict = {}

    for fname in glob.glob(resultFilePath+"*.csv"):
        print fname
        only_filename = fname.replace('\\','/').split('/')[-1]
        # print fname
        parampostfx = only_filename.split("svm_result")[1].split(".csv")[0]

        Index = parampostfx.split("_")[-1]
        C = parampostfx.split("_")[0]
        Gamma = parampostfx.split("_")[1]

        trainDir = testDirList[int(Index)]

        value = robjects.r['getGlobalScore_ch1'](trainDir,fname)[1]
        if C+"_"+Gamma not in valuedict:
            valuedict[C+"_"+Gamma] = []
        l = valuedict[C+"_"+Gamma]
        l.append(value)
        valuedict[C+"_"+Gamma] = l
        indexvaluedict[C+"_"+Gamma+"_"+Index] = value

    maxparam = "0.0_0.0"
    maxvalue = -1

    for key in valuedict:
        meanvalue = np.mean(valuedict[key])
        if maxvalue < meanvalue:
            maxparam = key
            maxvalue = meanvalue
    finalC = maxparam.split("_")[0]
    finalGamma = maxparam.split("_")[1]
    valuelist = []
    for i in range(10):
        valuelist.append(indexvaluedict[maxparam+"_"+str(i)])

    f = open(paramPath,'w')
    f.write("C,Gamma\n"+finalC+","+finalGamma+"\n")
    f.flush()
    f.close()
    f = open(baselinePath, "w")
    f.write("index,baseline\n")
    for i in range(10):
        f.write(str(i)+","+str(valuelist[i])+'\n')
        f.flush()
    f.close()

# output : both float values. and one list of each index's finalscore Those will be used as C and Gamma params in SVR learning

# This is an example of inputs and outputs
if __name__ == '__main__':

    testsetpath = "/pizza/data/answer/ch1_new_test_set_"
    testsetList = []
    for i in range(10):
        testsetList.append(testsetpath+str(i)+".csv")
    findBestParam("J2condor/result/", testsetList, "data/" + round_num + "/J3condor/result/parameter.csv", "data/" + round_num + "/J3condor/result/baseline.csv")
