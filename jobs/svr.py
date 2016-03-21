import sys
import os
import pandas as pd


import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from sklearn import svm
from sklearn.linear_model import MultiTaskLasso
from glob import glob
import shutil
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.ensemble as ensemble
from sklearn.isotonic import isotonic_regression
import sklearn.linear_model as linear_model
import sklearn.gaussian_process as gaussian
import sklearn.neighbors as neigbors

# filePath = "/pizza/"
# filePath = "C:/Users/Jang/Desktop/pizza/"

def get_result_value(resultFilePath, testDirList, problem_num):

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
    # Get the performance score//aa of Subchallenge 2
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

    if problem_num.startswith('2'):
        value = robjects.r['getGlobalScore_ch2'](testDirList, resultFilePath)[0]
    else :
        value = robjects.r['getGlobalScore_ch1'](testDirList, resultFilePath)[1]

    return value


def find_max_index(train_libfm, test_libfm):
    index_list = []
    with open(train_libfm,'r') as r:
        for line in r:
            line_list = line.strip().split(" ")
            for index in range(1, len(line_list)):
                if line_list[index].split(":")[0] is not '':
                    index_list.append(int(line_list[index].split(":")[0]))

    with open(test_libfm,'r') as r:
        for line in r:
            line_list = line.strip().split(" ")
            for index in range(1, len(line_list)):
                if line_list[index].split(":")[0] is not '':
                    index_list.append(int(line_list[index].split(":")[0]))

    return max(index_list)

def convertToSKlearnInput_withMaxIndex(libfmFile,maxIndex):
    with open(libfmFile,'r') as r:
        synergy_list=[]
        feature_list=[]
        line_list_list=[]

        for line in r:
            line_list = line.strip().split(" ")
            line_list_list.append(line_list)
            synergy_list.append(float(line_list[0]))


        max_index = maxIndex

        for line_list in line_list_list:
            temp_feature_list = [0]*(max_index+1)
            for index in range(1,len(line_list)):
                if line_list[index].split(":")[0] is not '':
                    feature_index=int(line_list[index].split(":")[0])
                if line_list[index].split(":")[0] is not '':
                    feature_value=float(line_list[index].split(":")[1])
                temp_feature_list[feature_index]= feature_value
            feature_list.append(temp_feature_list)

        return feature_list, synergy_list

def convertToSKlearnInput(libfmFile):
    with open(libfmFile,'r') as r:
        synergy_list=[]
        feature_list=[]

        index_list=[]
        line_list_list=[]

        for line in r:
            line_list = line.strip().split(" ")
            line_list_list.append(line_list)
            synergy_list.append(float(line_list[0]))
            #print line_list
            for index in range(1, len(line_list)):
                if line_list[index].split(":")[0] is not '':
                    index_list.append(int(line_list[index].split(":")[0]))
        max_index = max(index_list)


        for line_list in line_list_list:
            temp_feature_list = [0]*(max_index+1)
            for index in range(1,len(line_list)):
                if line_list[index].split(":")[0] is not '':
                    feature_index=int(line_list[index].split(":")[0])
                if line_list[index].split(":")[0] is not '':
                    feature_value=float(line_list[index].split(":")[1])
                temp_feature_list[feature_index]= feature_value
            feature_list.append(temp_feature_list)

        return feature_list, synergy_list

def get_excluded_list():
    file_path = naming + '.txt'
    f = open(file_path)

    excluded_list = list()
    for v in f:
        excluded_list.append( int(v) )

    return excluded_list

def run_sklearn(train_hdf, test_hdf ,testPredCSV,testCSV,
        input_c, input_gamma, excluded_list, problem_num,
        n_est=1000, lr=0.07, depth=7, maxindexBool=False, threshold=20):
    print train_hdf, test_hdf, testCSV

    train_feature_list = pd.read_hdf(train_hdf, 'traindf')
    test_feature_list = pd.read_hdf(test_hdf, 'testdf')

    train_synergy_list = train_feature_list['synergy_score']
    train_feature_list = train_feature_list.drop('synergy_score', axis=1)

    train_feature_list = train_feature_list.drop(excluded_list, axis=1)
    test_feature_list = test_feature_list.drop(excluded_list, axis=1)

    #regr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=0, loss='ls')
    regr = svm.SVR(kernel='rbf', C=input_c, gamma=input_gamma, epsilon=5)
    #regr = ensemble.RandomForestRegressor(n_estimators=value1,max_features=value2, max_depth=value3,n_jobs=4, verbose=1)
    #regr = KernelRidge(kernel='rbf',alpha=value1, gamma=value2)
    #regr = MultiTaskLasso(alpha=1.0)
    regr.fit(train_feature_list , train_synergy_list)
    pred = regr.predict(test_feature_list)
    testDF = pd.read_csv(testCSV).loc[:,["CELL_LINE","COMBINATION_ID","SYNERGY_SCORE"]]
    testDF["SYNERGY_SCORE"]=pred
    testDF.columns = ["CELL_LINE","COMBINATION_ID","PREDICTION"]

    target_path = testPredCSV + '/svm_result' + str(input_c) + '_0.061_eval.csv'

    if problem_num.startswith('2'):
        pivoted_df = testDF.pivot(index='COMBINATION_ID',columns='CELL_LINE',values='PREDICTION')
        filledpivoted_df = pivoted_df.fillna(0.0)
        filledpivoted_df[filledpivoted_df<threshold] = 0
        filledpivoted_df[filledpivoted_df>=threshold] = 1
        filledpivoted_df.to_csv(target_path)
    else :
        testDF.to_csv(target_path,index=False)

    # final testing
    value = get_result_value(target_path, testCSV, problem_num)
    f = open(testPredCSV + '/final_result.txt', 'w')
    f.write(str(value))
    f.close()
    #rearrange_results(testPredCSV)

"""
def rearrange_results(result_file_path):
    content = []
    for file_path in glob('J3/*'):
        file_name = os.path.basename(file_path)

        f = open(file_path)
        content.append("### " + file_name)
        content.append(f.read())
        content.append("\n\n")
        f.close()

        os.rename(file_path, '/'.join([result_file_path, file_name]))

    f = open( '/'.join([result_file_path, 'all_result.txt']), 'a')
    f.write( '\n'.join(content))
    f.close()
"""


# remove excluded index in dataframe
"""
naming = sys.argv[1]
execute_time = sys.argv[2]
problem_num = sys.argv[3]
input_c = float(sys.argv[4])
input_gamma = 0.061
set_idx = int(sys.argv[5])

data_path = sys.argv[6]
answer_path = sys.argv[7]
"""

def run(data_path, answer_path, result_path, C, gamma, excluded_list, problem_num):
    run_sklearn('/'.join([ data_path, 'compact_train.hdf']),
                '/'.join([ data_path, 'compact_test.hdf']),
                result_path,
                answer_path,
                C, gamma,
                excluded_list,
                problem_num,
                maxindexBool=True)
