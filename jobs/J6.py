import pandas as pd
import numpy as np
import os, sys
import smtplib
from email.mime.text import MIMEText
from email.header    import Header
import shutil

#def makeExcludedIndex(resultDataDir, extractedColumnIndexPath, threshold=8, numberOfExcludedIndex=50):
def makeExcludedIndex(round_num, resultDataDir, extractedColumnIndexPath, threshold=6, numberOfExcludedIndex=50):
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
            v_float = float(v)
            if np.isnan(v_float):
                continue
            data.append(v_float)
        count = 0
        for d in data:
            if d > 0 : count += 1
        indexList.append(index)
        meanList.append(np.sum(data) / 10.0)
        countList.append(count)

    resultDF = pd.DataFrame({"Mean":meanList, "Count":countList}, index=indexList)
    resultDF_sorted = resultDF.sort_values(["Count", "Mean"], ascending=False)

    contents = "Top 20% in Round #" + round_num + "\n"
    mainContentNumber = int(len(resultDF_sorted.index) * 0.2)

    main_count = 0
    for t in resultDF_sorted.itertuples():
        for v in t:
            contents += str(v) + " "
        contents += "\n"
        if main_count == mainContentNumber:
            break
        main_count += 1

    contents += "\n"
    contents += "Bottom 20%\n"

    main_count = 0
    resultDF_sorted_ascending = resultDF.sort_values(["Count", "Mean"], ascending=True)
    for t in resultDF_sorted_ascending.itertuples():
        for v in t:
            contents += str(v) + " "
        contents += "\n"
        if main_count == mainContentNumber:
            break
        main_count += 1

    smtp_host = 'smtp.gmail.com'
    login, password = 'dmis.dreamchallenge@gmail.com', 'dmisinfos#1'

    recipients_emails = [
            'minji.jeon1@gmail.com',
            'hyeockyoonjang@gmail.com',
            'sunkyu4276@gmail.com',
            'leeheewon78@gmail.com',
            'kangj@korea.ac.kr',
            'minhwan90@gmail.com']
    msg = MIMEText(contents, 'plain', 'utf-8')
    msg['Subject'] = Header('J6 Result', 'utf-8')
    msg['From'] = login
    msg['To'] = ", ".join(recipients_emails)
    s = smtplib.SMTP(smtp_host, 587, timeout=10)
    s.set_debuglevel(1)
    try:
        s.starttls()
        s.login(login, password)
        s.sendmail(msg['From'], recipients_emails, msg.as_string())
    finally:
        s.quit()

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
    round_num = sys.argv[1]
    makeExcludedIndex(round_num, "J5condor/result/", "data/" + str(int(round_num)+1) + "/J6condor/result/excludedIndexes.txt")
