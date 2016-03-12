# -* coding:utf-8 *-

from multiprocessing import Process
from subprocess import call
import sys
import pandas as pd
import os, shutil
import Constant

def execute(round_num):
    #input1:defaultTrainlibfm
    #input2:defaultTestlibfm

    defaultTestlibfm = "includeTest_single_1a.libfm"
    defaultTrainlibfm = "includeTrain_single_1a.libfm"

    # defaultTrainlibfm = sys.argv[0]
    # defaultTestlibfm = sys.argv[1]
    round_num_str = str(round_num)


    dataFilePath = "data/" + round_num_str + "/J1condor/includeTestSamples_1a/"
    j6FilePath = "data/" + round_num_str + "/J6condor/result/"

    #test
    J6FileNames = os.listdir(j6FilePath)
    droplinesSet = set()
    for J6FileName in J6FileNames:
        print j6FilePath+J6FileName
        f = open(j6FilePath+J6FileName,"r")
        lines = map(int, f.read().splitlines())
        droplinesSet = droplinesSet.union(set(lines))
        f.close()
    print len(droplinesSet)," features are removed"
    droplines = list(droplinesSet)

    for i in range(0,10):
        currfilePath = dataFilePath+"set"+str(i)+"/"
        basicTrainlibfm = currfilePath+defaultTrainlibfm
        basicTestlibfm = currfilePath+defaultTestlibfm

        print "loading..."
        originaltrainDF = Constant.libfmFileToDF(basicTrainlibfm)
        originaltestDF = Constant.libfmFileToDF(basicTestlibfm)

        newtrainDF = originaltrainDF.drop(originaltrainDF.columns[droplines], axis=1)
        newtestDF = originaltestDF.drop(originaltrainDF.columns[droplines], axis=1)

        Constant.makeLibfmFileWithDF(originaltrainDF, newtrainDF,currfilePath+"Train_single_new.libfm")
        Constant.makeLibfmFileWithDF(originaltestDF, newtestDF,currfilePath+"Test_single_new.libfm")


    #`call(["bash", "uploader.sh"])
