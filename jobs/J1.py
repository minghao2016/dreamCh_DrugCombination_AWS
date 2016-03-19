# -* coding:utf-8 *-
from multiprocessing import Process
from subprocess import call
import sys
import pandas as pd
import os, shutil
import Constant
import email_sender


#input1:defaultTrainlibfm
#input2:defaultTestlibfm

#defaultTestlibfm = "includeTest_single_2.libfm"
#defaultTrainlibfm = "includeTrain_single_2.libfm"
defaultTestlibfm = "tmpTest_single_1a_expanded.libfm"
defaultTrainlibfm = "tmpTrain_single_1a_expanded.libfm"
# defaultTrainlibfm = sys.argv[0]
# defaultTestlibfm = sys.argv[1]

def single_loop(i, dataFilePath, droplines):
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


def execute(round_num):
    round_num_str = str(round_num)

    dataFilePath = "data/" + round_num_str + "/J1condor/includeTestSamples_1a/"

    droplinesSet = set()
    for round_idx in range(round_num+1):
    	j6FilePath = "data/" + str(round_idx) + "/J6condor/result/"

	#test
	J6FileNames = os.listdir(j6FilePath)

        lines = list()
	for J6FileName in J6FileNames:
	    print j6FilePath+J6FileName
	    f = open(j6FilePath+J6FileName,"r")
            lines = map(int, f.read().splitlines())
            droplinesSet = droplinesSet.union(set(lines))
            f.close()

        if len(lines) == 0 and round_num != 0 and round_num == round_idx:
            email_sender.send("Finish!")
            sys.exit()

    print len(droplinesSet)," features are removed"
    droplines = list(droplinesSet)

    processes = list()
    for i in range(0,10):
	p = Process(target=single_loop, args=(i,dataFilePath, droplines))
        processes.append(p)
	p.start()

    for p in processes:
        p.join()

    # rewrite result6 file
    total_j6_file_path = "data/" + str(round_idx) + "/J6condor/result/excludedIndexes.txt"
    f = open(total_j6_file_path, 'w')
    f.write('\n'.join([ str(val) for val in droplines ]))
    f.close()

    return droplinesSet

    #`call(["bash", "uploader.sh"])

if __name__ == '__main__':
    execute(0)
