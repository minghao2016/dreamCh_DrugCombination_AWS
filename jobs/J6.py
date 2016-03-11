import pandas as pd
import numpy as np
import os

def makeExcludedIndex(resultDataDir, extractedColumnIndexPath, threshold=9, numberOfExcludedIndex=50):
    """
    :param resultDataDir: there must be 10 files named ["result_N.csv"(N : 0 to 9)]
    :param extractedColumnIndexPath: result file path including file name. 'txt' file type is recommended.
    :return: the number of excluded Indexes. It must be lower than 50(numberOfExcludedIndex).
    """
    DataDir = resultDataDir
    dfList = []
    for index in range(0,10):
        tempDF = pd.read_csv(DataDir + "/result_" + str(index) + ".csv", index_col=0, header=None)
        dfList.append(tempDF)

    result = pd.concat(dfList, axis=1)
    indexList = []
    meanList = []
    countList = []
    for tuple in result.itertuples():
        index = int(tuple[0])
        data = []
        for v in tuple[1:]:
            data.append(float(v))
        count = 0
        for d in data:
            if d > 0 : count += 1
        indexList.append(index)
        meanList.append(np.mean(data))
        countList.append(count)

    resultDF = pd.DataFrame({"Mean":meanList, "Count":countList}, index=indexList)
    resultDF_sorted = resultDF.sort_values(["Count", "Mean"], ascending=False)

    resultDF_overThreshold = resultDF_sorted[resultDF_sorted["Count"] >= threshold]

    extractedColumnIndexList = list(resultDF_overThreshold.index[0:numberOfExcludedIndex])
    paths = extractedColumnIndexPath.split('/')
    target_dir = '/'.join(paths[:-1])
    os.makedirs(target_dir)
    with open(extractedColumnIndexPath, "w") as fw:
        for v in extractedColumnIndexList:
            fw.write(str(v) + "\n")

    return len(extractedColumnIndexList)

if __name__ == '__main__':
    import sys
    round_num = int(sys.argv[1])
    makeExcludedIndex("J5condor/result/", "data/" + str(round_num + 1) + "/J6condor/result/excludedIndexes.txt")
