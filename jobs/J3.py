
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
problemNum = sys.argv[2]

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

    library(ROCR)
    # ------------------------------------------------------------------------------------
    # Description: AZ-Sanger Challenge scoring functions
    # Authors: Michael P Menden, Julio Saez-Rodriguez, Mike Mason, Thomas Yu
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # Get observation format for Subchallenge 2
    # ------------------------------------------------------------------------------------
    getObs_ch2 <- function(ls,threshold = F) {
      combi <- unique(ls$COMBINATION_ID)
      cell <- unique(ls$CELL_LINE)
      mat <- matrix(NA, nrow=length(combi), ncol=length(cell),
                    dimnames=list(combi, cell))
      for (i in 1:nrow(ls))
        mat[as.character(ls$COMBINATION_ID[i]),
            as.character(ls$CELL_LINE[i])] <- ls$SYNERGY_SCORE[i]

      if (is.numeric(threshold)) {
        mat <- signif(mat,3)
        mat[mat < threshold] = 0
        mat[mat >= threshold ] = 1
      }
      return(mat)
    }
    # ------------------------------------------------------------------------------------
    # Get prediction format for Subchallenge 2
    # ------------------------------------------------------------------------------------
    getPred_ch2 <- function(pred) {
      if (all(row.names(pred) == c(1:nrow(pred)))) {
        row.names(pred) = pred[,1]
        pred = pred[,-1]
      }
      pred <- as.matrix(pred)
      return(pred)
    }
    # ------------------------------------------------------------------------------------
    # Get unsigned score from one dimensional ANOVA
    # ------------------------------------------------------------------------------------
    getNegLog10pVal_ch2 <- function(fit, obs) {
      s <- 0
      if (!is.na(fit$coefficients[2]) & sum(!is.na(obs)) > 2)
          s <- -log10(anova(fit)['pred','Pr(>F)'])
      return(s)
    }
    # -----------------------------------------------------------------------
    # Scores from confusion Matrix
    # -----------------------------------------------------------------------
    getPrecision_ch2 <- function(obs_path, pred, threshold=30) {
      obs <- read.csv(obs_path)
      obs <- getObs_ch2(obs,threshold)

      pred <- read.csv(pred,stringsAsFactors=F,check.names = F)
      pred <- getPred_ch2(pred)
      pred <- pred[match(row.names(obs),row.names(pred)),]
      pred <- pred[,match(colnames(obs),colnames(pred))]
      #Remove all NA's
      pred <- as.numeric(pred)[!is.na(obs)]
      obs <- as.numeric(obs)[!is.na(obs)]
      preds <- prediction(pred,obs)
      prec <- performance(preds,"prec") #precision (Acc + )
      sens <- performance(preds,"sens") #True positive rate (Sensitivity) (Cov +)
      npv <- performance(preds,"npv") #Negative predictive value (Acc - )
      spec <- performance(preds,"spec") #True negative rate(specificity) (Cov -)
      auc <- performance(preds,"auc") #Area under curve (AUC)
      phi <- performance(preds,"phi") #phi correlation coefficient, (matthews)
      aupr <- performance(preds, "prec", "rec") #Area under precision recall (AUPR)

      prec_val <- unlist(prec@y.values)[2]
      sens_val <- unlist(sens@y.values)[2]
      npv_val <- unlist(npv@y.values)[2]
      spec_val <- unlist(spec@y.values)[2]
      auc_val <- unlist(auc@y.values)
      phi_val <- unlist(phi@y.values)[2]
      BAC <- (sens_val + spec_val)/2
      F1 <- 2*preds@tp[[1]][2]/(2*preds@tp[[1]][2] + preds@fn[[1]][2] + preds@fp[[1]][2])
      aupr_val <- unlist(aupr@y.values)[2]

      #If predictions are 0, ROCR treats 0 as the positive value when it is actually the negative
      if (all(pred == 0)) {
        prec_val <- unlist(prec@y.values)[1]
        sens_val <- unlist(sens@y.values)[1]
        npv_val <- unlist(npv@y.values)[1]
        spec_val <- unlist(spec@y.values)[1]
        auc_val <- unlist(auc@y.values)
        phi_val <- unlist(phi@y.values)[1]
        BAC <- (sens_val + spec_val)/2
        F1 <- 2*preds@tp[[1]][1]/(2*preds@tp[[1]][1] + preds@fn[[1]][1] + preds@fp[[1]][1])
        aupr_val <- unlist(aupr@y.values)[1]
      }

      return(round(c(prec=prec_val,
                     sens = sens_val,
                     npv = npv_val,
                     spec=spec_val,
                     auc=auc_val,
                     phi=phi_val,
                     BAC=BAC, # Tie breaking Metric
                     F1=F1,
                     aupr=aupr_val),2))
    }
    # ------------------------------------------------------------------------------------
    # Get the drug combinations score of Subchallenge 2
    # ------------------------------------------------------------------------------------
    getOneDimScore_ch2 <- function(obs_path, pred, confidence="none", topX=10, rows=T) {
      obs <- read.csv(obs_path)
      obs <- getObs_ch2(obs)

      pred <- read.csv(pred,stringsAsFactors=F,check.names = F)
      pred <- getPred_ch2(pred)
      pred <- pred[match(row.names(obs),row.names(pred)),]
      pred <- pred[,match(colnames(obs),colnames(pred))]
      n <- ncol(obs)
      if (rows)
        n <- nrow(obs)

      s <- c()
      for (i in 1:n) {
        sign <- 1
        if (rows) {
          fit <- aov(obs[i,] ~ pred[i,])
          nlp <- getNegLog10pVal_ch2(fit,obs[i,])
            if (nlp!=0 & (mean(obs[i, pred[i,]==1], na.rm=T) < mean(obs[i, pred[i,]==0], na.rm=T)))
              sign <- -1
        } else {
          fit <- aov(obs[,i] ~ pred[,i])
          nlp <- getNegLog10pVal_ch2(fit,obs[,i])
            if (nlp!=0 & (mean(obs[pred[,i]==1, i], na.rm=T) < mean(obs[pred[,i]==0, i], na.rm=T)))
              sign <- -1
        }
        s <- c(s, sign * nlp)
      }

      if (!file.exists(confidence))
        return(round(c(mean=mean(s),
                 ste=sd(s)),2))

      confidence <- read.csv(confidence,stringsAsFactors=F,check.names = F)
      confidence <- getPred_ch2(confidence)
      confidence <- confidence[match(row.names(obs),row.names(confidence)),]
      confidence <- confidence[,match(colnames(obs),colnames(confidence))]

      if (rows) {
        nVal <- round(topX * (nrow(confidence) / 100))
      } else {
        nVal <- round(topX * (ncol(confidence) / 100))
      }

      nStep <- 1000
      boot_score <- rep(0, nVal)
      for (i in 1:nStep) {
        if (rows) {
          avgConf <- sapply(1:nrow(confidence), function(x) mean(confidence[x, !is.na(obs[x,])]))
        } else {
          avgConf <- sapply(1:ncol(confidence), function(x) mean(confidence[!is.na(obs[,x]), x]))
        }
        idx <- order(avgConf, sample(length(avgConf)), decreasing = T)[1:nVal]
        boot_score <- boot_score + s[idx]
      }

      return(round(c(mean=mean(boot_score/nStep),
                     ste=sd(boot_score/nStep)),2))
    }
    # ------------------------------------------------------------------------------------
    # Get the performance score of Subchallenge 2
    # ------------------------------------------------------------------------------------
    getGlobalScore_ch2 <- function(obs_path, pred) {
      obs <- read.csv(obs_path)
      obs <- getObs_ch2(obs)

      pred <- read.csv(pred,stringsAsFactors=F,check.names = F)
      pred <- getPred_ch2(pred)
      pred <- pred[match(row.names(obs),row.names(pred)),]
      pred <- pred[,match(colnames(obs),colnames(pred))]
      # regress out combination bias
      cov <- rep(rownames(obs), ncol(obs))

      c0 <- rep(rownames(obs), ncol(obs))
      c1 <- as.vector(matrix(colnames(obs), ncol=ncol(obs), nrow=nrow(obs), byrow=T))

      obs <- as.vector(obs)
      pred <- as.vector(pred)

      c0 <- c0[!is.na(obs)]
      c1 <- c1[!is.na(obs)]
      pred <- pred[!is.na(obs)]
      obs <- obs[!is.na(obs)]

      if(all(pred==0) | all(pred==1))
        return(0)
      # run anove with combination label as covariate
      fit <- aov(obs ~ c0 + c1 + pred)
      pVal <- -log10(anova(fit)['pred','Pr(>F)'])

      sign <- 1
      if (sum(!is.na(obs[pred==1])) >0  && sum(!is.na(obs[pred==0]))>0)
        if (mean(obs[pred==1], na.rm=T) < mean(obs[pred==0], na.rm=T))
          sign <- -1

      return(round(sign * pVal,2)) # Final Metric
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

        if problemNum == "2" :
            value = robjects.r['getGlobalScore_ch2'](trainDir, fname)[0]

        else :
            value = robjects.r['getGlobalScore_ch1'](trainDir, fname)[1]

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
    testsetpath = "/home/ubuntu/data/answers/ch1_newtestset_wtest_"
    testsetList = []
    for i in range(10):
        testsetList.append(testsetpath+str(i)+".csv")
    findBestParam("J2condor/result/", testsetList, "data/" + round_num + "/J3condor/result/parameter.csv", "data/" + round_num + "/J3condor/result/baseline.csv")
