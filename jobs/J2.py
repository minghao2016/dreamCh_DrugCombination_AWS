import sys
import os
import pandas as pd
import csv

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

# deprecated
"""
round_num = int(sys.argv[1])
problemNum = sys.argv[2]
value1 = float(sys.argv[3])
value2 = float(sys.argv[4])
value4 = int(sys.argv[5])
"""
round_num   = int(sys.argv[1])
problemNum  = sys.argv[2]        # not used in including model
feature_candidate = sys.argv[3]
c_value     = float(sys.argv[4])
gamma_value = float(sys.argv[5])
set_num     = int(sys.argv[6])

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
    file_path = 'excludedIndexes.txt'
    f = open(file_path)

    excluded_list = list()
    for v in f:
        excluded_list.append( int(v) )
    return excluded_list

def get_default_feature_ids():
    file_path = 'default_ids.txt'
    fids = list()

    f = open(file_path)
    for v in f:
        fids.append( int(v) )

    # add test feature ids
    f = open('/features/exclude_' + feature_candidate + '.csv')
    csv_f = csv.reader(f)
    csv_f.next()
    for v in csv_f:
        fids.append( int(v[0]) )
    print fids
    return fids

def run_sklearn(train_hdf, test_hdf,testPredCSV,testCSV , n_est=1000, lr=0.07, depth=7, maxindexBool=False, threshold=20):
    """
    if maxindexBool==False:
        train_feature_list, train_synergy_list = convertToSKlearnInput(train_libfm)
        test_feature_list, test_synergy_list = convertToSKlearnInput(test_libfm)
    elif maxindexBool == True:
        max_index = find_max_index(train_libfm,test_libfm)
        train_feature_list, train_synergy_list = convertToSKlearnInput_withMaxIndex(train_libfm,max_index)
        test_feature_list, test_synergy_list = convertToSKlearnInput_withMaxIndex(test_libfm,max_index)
    """
    print train_hdf
    print test_hdf
    train_feature_list = pd.read_hdf(train_hdf, 'traindf')
    test_feature_list = pd.read_hdf(test_hdf, 'validationdf')

    train_synergy_list = train_feature_list['synergy_score']
    train_feature_list = train_feature_list.drop('synergy_score', axis=1)

    """
    # remove excluded_indices
    excluded_list = get_excluded_list()
    train_feature_list = train_feature_list.drop(excluded_list, axis=1)
    test_feature_list  = test_feature_list.drop(excluded_list, axis=1)
    """
    features_ids = get_default_feature_ids()
    train_feature_list = train_feature_list[features_ids]
    test_feature_list  = test_feature_list[features_ids]

    print features_ids
    print len(features_ids)

    #regr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=0, loss='ls')
    regr = svm.SVR(kernel='rbf', C=c_value, gamma=gamma_value, epsilon=5)
    #regr = ensemble.RandomForestRegressor(n_estimators=value1,max_features=value2, max_depth=value3,n_jobs=4, verbose=1)
    #regr = KernelRidge(kernel='rbf',alpha=value1, gamma=value2)
    #regr = MultiTaskLasso(alpha=1.0)
    regr.fit(train_feature_list , train_synergy_list)
    pred = regr.predict(test_feature_list)
    testDF = pd.read_csv(testCSV).loc[:,["CELL_LINE","COMBINATION_ID","SYNERGY_SCORE"]]
    testDF["SYNERGY_SCORE"]=pred
    testDF.columns = ["CELL_LINE","COMBINATION_ID","PREDICTION"]

    if problemNum=="2" :
        pivoted_df = testDF.pivot(index='COMBINATION_ID',columns='CELL_LINE',values='PREDICTION')
        filledpivoted_df = pivoted_df.fillna(0.0)
        filledpivoted_df[filledpivoted_df<threshold] = 0
        filledpivoted_df[filledpivoted_df>=threshold] = 1
        filledpivoted_df.to_csv(testPredCSV)

    else :
        testDF.to_csv(testPredCSV,index=False)




print "round_num: {0}\nproblem: {1}\nfeature: {2}\nc: {3}\ngamma: {4}, set: {5}".format(
        round_num, problemNum, feature_candidate, c_value, gamma_value, set_num)


root_dir = "/hdf/"
os.makedirs("data/" + str(round_num) + "/J2condor/result/" + feature_candidate)
run_sklearn(root_dir + str(set_num) + "compact_train.hdf", # single train set
            root_dir + str(set_num) + "compact_validation.hdf", # single test set
            "data/" + str(round_num) + "/J2condor/result/" + feature_candidate + "/svm_result_" +str(c_value)+"_"+str(gamma_value)+"_"+str(set_num)+".csv", # result file path
            root_dir + "/answer/ch1_new_test_set_excluded_"+str(set_num)+".csv", # answer set
            maxindexBool=True)


"""
run_sklearn("J1condor/set"+str(value4)+"/Train_single_new.libfm", # single train set
            "J1condor/set"+str(value4)+"/Test_single_new.libfm", # single test set
            "data/" + str(round_num) + "/J2condor/result/svm_result"+str(value1)+"_"+str(value2)+"_"+str(value4)+".csv", # result file path
            "/pizza/data/answer/ch1_new_test_set_"+str(value4)+".csv", # answer set
            maxindexBool=True)
"""
# results =  glob("Data/J2condor/result/svm_result"+str(value1)+"_"+str(value2)+"_"+str(value4)+".csv")
#
# for result in results:
#     f_name = os.path.basename(result)
#     shutil.move(result,"./Data/J2condor/" + f_name)
