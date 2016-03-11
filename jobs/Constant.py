
import pandas as pd
import rpy2.robjects as robjects


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

  DrugCombiScore_ch1_Wsq <- sum(R$V1*sqrt(R$N-1))/sum(sqrt(R$N-1)) # primary metric
  DrugCombiScore_ch1_GT20  <- mean(R[obsMax[,2] >= 20,2])          # tie-breaking metric
  # ----------------------------------------

  # partial out the mean of synergy across cell lines and combinationations
  return(c(score=numerator/denumerator,
           final=DrugCombiScore_ch1_Wsq,
           tiebreak=DrugCombiScore_ch1_GT20))
}
''')
#
# robjects.r('''
# getObs_ch1 <- function(ls) {
#   return(data.frame(CELL_LINE=as.character(ls$CELL_LINE),
#                     COMBINATION_ID=as.character(ls$COMBINATION_ID),
#                     OBSERVATION=ls$SYNERGY_SCORE))
# }
#
# # Get the drug combinations score of Subchallenge 1
# getDrugCombiScore_ch1 <- function(obs, pred, confidence=NA, topX=10) {
#   R <- c()
#   obs <- read.csv(obs,stringsAsFactors = F)
#   obs <- getObs_ch1(obs)
#   pred <- read.csv(pred,stringsAsFactors=F)
#   pred <- pred[match(paste(obs$CELL_LINE,obs$COMBINATION_ID),paste(pred$CELL_LINE,pred$COMBINATION_ID)),]
#
#   pred$COMBINATION_ID <- gsub(" ", "", pred$COMBINATION_ID)
#   for (i in as.character(unique(obs$COMBINATION_ID))) {
#       R <- c(R, cor(obs[obs$COMBINATION_ID == i, 'OBSERVATION'],
#                     pred[pred$COMBINATION_ID == i, 'PREDICTION']))
#   }
#   #Make NA's in R = 0
#   R[is.na(R)] = 0
#   names(R) <- as.character(unique(obs$COMBINATION_ID))
#
#   if (!file.exists(confidence))
#     return(round(c(mean=mean(R),
#              ste=sd(R),
#              n=sum(!is.na(R))),2))
#
#   confidence <- read.csv(confidence,stringsAsFactors=F)
#   confidence <- confidence[match(unique(obs$COMBINATION_ID),confidence$COMBINATION_ID),]
#
#   nStep <- 1000
#   nVal <- round(topX * (length(R) / 100))
#   boot_R <- rep(0, nVal)
#   for (i in 1:nStep) {
#     idx <- order(confidence$CONFIDENCE, sample(length(R)), decreasing = T)[1:nVal]
#     boot_R <- boot_R + R[idx]
#   }
#
#   return(round(c(mean=mean(boot_R/nStep),
#                  ste=sd(boot_R/nStep),
#                  n=sum(!is.na(boot_R/nStep))),2))
# }
#
# # ------------------------------------------------------------------------------------
# # Get the global score of Subchallenge 1
# # ------------------------------------------------------------------------------------
# getGlobalScore_ch1 <- function(obs, pred) {
#   obs <- read.csv(obs, stringsAsFactors=F)
#   obs <- getObs_ch1(obs)
#   pred <- read.csv(pred,stringsAsFactors=F)
#   pred <- pred[match(paste(obs$CELL_LINE,obs$COMBINATION_ID),paste(pred$CELL_LINE,pred$COMBINATION_ID)),]
#
#   x = obs$OBSERVATION
#   y = pred$PREDICTION
#
#   agg <- aggregate(OBSERVATION ~ CELL_LINE, obs, median)
#   z0 <- agg$OBSERVATION[match(obs$CELL_LINE, agg$CELL_LINE)]
#
#   agg <- aggregate(OBSERVATION ~ COMBINATION_ID, obs, median)
#   z1 <- agg$OBSERVATION[match(obs$COMBINATION_ID, agg$COMBINATION_ID)]
#
#   parCor <- function(u,v,w) {
#     numerator = cor(u,v) - cor(u,w) * cor(w,v)
#     denumerator = sqrt(1-cor(u,w)^2) * sqrt(1-cor(w,v)^2)
#     return(numerator/denumerator)
#   }
#
#   numerator=parCor(x,y,z1) - parCor(x,z0,z1) * parCor(z0,y,z1)
#   denumerator= sqrt(1-parCor(x,z0,z1)^2) * sqrt(1-parCor(z0,y,z1)^2)
#
#   # partial out the mean of synergy across cell lines and combinationations
#   return(c(score=numerator/denumerator))
# }
# ''')




# # _instance = None
# filePath  = ""
# libfmFIlePath = ""
# lassoAnalysisPath = "../ResultFiles/Lasso analysis"
# doubleCrossvalidationPath = "C:/Users/Sunkyu/Development/DREAM/InputFiles/doubleCrossvalidation/"
# leaderboardPath = "C:/Users/Sunkyu/Development/DREAM/InputFiles/leaderboardSubmit/"
# javaFilePath = "C:/Users/Sunkyu/Development/Practice/resources/"
# libfmPath = "C:/Users/Sunkyu/Development/DREAM/Codes/LibFM/"
# nasFeaturePath = "Q:/DreamChallenge/AstraZeneca-Sanger/Challenge Resources/chall1/"

def libfmFileToDF(path):

    returnDF = pd.DataFrame()
    synergyScoreList = []
    maxLibfmId = 0
    with open(path) as f:
        lines = f.readlines()
        indexNum = 0
        for l in lines:
            lsplit = l.split()
            dataList = [0] * (20000)
            synergyScoreList.append(lsplit[0])
            for temp in lsplit[1:]:
                temp = temp.split(":")
                try:
                    libfmId = int(temp[0])
                    if libfmId > maxLibfmId:
                        maxLibfmId = libfmId
                    data = float(temp[1].strip())
                    dataList[libfmId] = data
                except:
                    print temp
            if len(returnDF) == 0:
                returnDF = pd.DataFrame(index=range(0, len(dataList)))
            returnDF[str(indexNum)] = dataList
            indexNum += 1

    returnDF = returnDF[0:(maxLibfmId + 1)]
    tp = returnDF.transpose()
    tp["Synergy score"] = synergyScoreList
    return tp

def makeLibfmFileWithDF(originaldf, df, libfmFileDir):
    print "Making new Train libfmFile.."
    synScore = df['Synergy score']
    features = df.drop("Synergy score", axis=1)
    originalfeatures = originaldf.drop("Synergy score", axis=1)
    lines = []
    for score, featureList in zip(synScore, features.itertuples()):

        line = str(score)
        idx = -1
        for i in features.columns:
            idx += 1
            if featureList[idx+1] == 0:
                continue
            #print i
            #print idx, featureList[idx]
            line += " " + str(i) + ":" + str(featureList[idx+1])
        lines.append(line)

    newTrainSetDir = libfmFileDir
    with open(newTrainSetDir, "w") as fw:
        for l in lines:
            fw.write(l + "\n")

    return len(featureList)

def write(predictedSynergy, answerSetFilePath, rootPath):
    fr = open(answerSetFilePath,'r')
    print "write"
    predFilePath = rootPath + "predFile.csv"
    fw = open(predFilePath,'w')
    lines = fr.readlines()
    i = 0
    fw.write("CELL_LINE,COMBINATION_ID,PREDICTION\n")
    for line in lines:
        splited = line.split(",")
        if i > 0:
            fw.write(splited[0]+","+splited[13].replace('\n', '')+","+str(predictedSynergy[i-1])+"\n")
        i+=1
    fw.close()
    fr.close()

    return predFilePath

def calculate(obsFile, predFile):
    [score, final, tiebreak] = robjects.r['getGlobalScore_ch1'](obsFile,predFile)
    # confidenceFile= Parameters.confidenceFilePath
    # [mean, ste, n] = robjects.r['getDrugCombiScore_ch1'](obsFile,predFile,confidenceFile,10)
    result = []
    # print score, final, tiebreak
    # result.append(score)
    # result.append(final)
    # result.append(tiebreak)
    # # print "10%"
    # print mean, ste, n,
    # result.append(mean)
    # result.append(ste)
    # result.append(n)
    # [mean, ste, n] = robjects.r['getDrugCombiScore_ch1'](obsFile,predFile,confidenceFile,20)
    # # print "20%"
    # print mean, ste, n,
    # result.append(mean)
    # result.append(ste)
    # result.append(n)
    # [mean, ste, n] = robjects.r['getDrugCombiScore_ch1'](obsFile,predFile,confidenceFile,30)
    # # print "30%"
    # print mean, ste, n
    # result.append(mean)
    # result.append(ste)
    # result.append(n)
    return score, final, tiebreak
