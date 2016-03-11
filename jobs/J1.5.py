
def makeCondorSubmitFile(pastRound_bestC, submitFilesDir):
    """
    :param pastRound_bestC: Centered on C parameter in the past round, it makes 12 candidate C parameter.
    :param submitFilesDir: Folder where submit files are saved.
    :return:
    """
    submitFormatPath = "condorSubmitFormat.txt"

    cList = range(pastRound_bestC - 20*6, pastRound_bestC + 20*6, 20)
    gammaList = [x/1000.0 for x in range(1,100,20)]

    submitFormatFile = open(submitFormatPath, "r")
    submitFormatFileLines = list(submitFormatFile.readlines())
    """
    Line index to change : 11("arguments = CrossValidDreamChallenge.py C_VALUE GAMMA_VALUE DATASETNUM_VALUE")
    """

    fileNumber = 0
    for cValue in cList:
        if cValue < 1:
            continue
        for gammaValue in gammaList:
           for datasetNum in range(0,10):
               with open(submitFilesDir + "condor.submit" + str(fileNumber), "w") as fw:
                   fileNumber += 1
                   for lineNum in range(len(submitFormatFileLines)):
                       tempLine = submitFormatFileLines[lineNum]
                       if lineNum == 11:
                           tempLine = tempLine.replace("C_VALUE", str(cValue))
                           tempLine = tempLine.replace("GAMMA_VALUE", str(gammaValue))
                           tempLine = tempLine.replace("DATASETNUM_VALUE", str(datasetNum))
                       fw.write(tempLine)

if __name__ == '__main__':
    makeCondorSubmitFile(300, "../Data/submitFiles/")