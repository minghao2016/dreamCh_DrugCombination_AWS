import sys
import os
import pandas as pd


from sklearn import svm
from sklearn.linear_model import MultiTaskLasso
from glob import glob
import shutil
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.ensemble as ensemble
from sklearn.isotonic import isotonic_regression
from sklearn.kernel_ridge import KernelRidge
import sklearn.linear_model as linear_model
import sklearn.gaussian_process as gaussian
import sklearn.neighbors as neigbors

filePath = "/pizza/"
featureFolderPath = "J6condor/result/" # TBD
# filePath = "C:/Users/Jang/Desktop/pizza/"

round_num = sys.argv[1]
value1 = int(sys.argv[2])
value4 = int(sys.argv[3])

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

            #del temp_feature_list[value1]
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

def get_params():
    f = open('J3condor/result/parameter.csv')
    lines = f.read().split('\n')

    target_line = lines[1].strip()
    params = target_line.split(',')

    return (float(params[0]), float(params[1]))


def run_sklearn(train_libfm, test_libfm,testPredCSV,testCSV , n_est=1000, lr=0.07, depth=7, maxindexBool=True):

    if maxindexBool == True:
        max_index = find_max_index(train_libfm,test_libfm)
        train_feature_list, train_synergy_list = convertToSKlearnInput_withMaxIndex(train_libfm,max_index)
        test_feature_list, test_synergy_list = convertToSKlearnInput_withMaxIndex(test_libfm,max_index)


    train_feature_df = pd.DataFrame(train_feature_list)
    test_feature_df = pd.DataFrame(test_feature_list)

    train_feature_df = train_feature_df.drop(value1, axis=1)
    test_feature_df = test_feature_df.drop(value1, axis=1)

    #regr = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=0, loss='ls')
    (param_c, param_gamma) = get_params()
    print param_c
    print param_gamma
    regr = svm.SVR(kernel='rbf', C=param_c, gamma=param_gamma, epsilon=5)
    #regr = ensemble.RandomForestRegressor(n_estimators=value1,max_features=value2, max_depth=value3,n_jobs=4, verbose=1)
    #regr = KernelRidge(kernel='rbf',alpha=value1, gamma=value2)
    #regr = MultiTaskLasso(alpha=1.0)
    regr.fit(train_feature_df , train_synergy_list)
    pred = regr.predict(test_feature_df)
    testDF = pd.read_csv(testCSV).loc[:,["CELL_LINE","COMBINATION_ID","SYNERGY_SCORE"]]

    testDF["SYNERGY_SCORE"]=pred
    testDF.columns = ["CELL_LINE","COMBINATION_ID","PREDICTION"]
    testDF.to_csv(testPredCSV,index=False)


delfeatlist = []
for fn in os.listdir(featureFolderPath):
    f = open(fn, 'r')
    for l in f:
        delfeatlist.append((int)(l.strip()))
    f.close()
if value1 in delfeatlist:
    sys.exit(0)

os.makedirs("data/" + str(round_num) + "/J4condor/result/")
run_sklearn("/mina/data/" + round_num + "/J1condor/set"+str(value4)+"/Train_single_new.libfm", # single train set
            "/mina/data/" + round_num + "/J1condor/set"+str(value4)+"/Test_single_new.libfm", # single test set
            "data/" + str(round_num) + "/J4condor/result/svm_result"+str(value1)+"_"+str(value4)+".csv",
            "/pizza/data/answer/ch1_new_test_set_"+str(value4)+".csv",
            #dellist,
            1000, 0.07, 7, True)
"""
run_sklearn("J1condor/set"+str(value4)+"/Train_single_new.libfm", # single train set
            "J1condor/set"+str(value4)+"/Test_single_new.libfm", # single test set
            "data/" + str(round_num) + "/J4condor/result/svm_result"+str(value1)+"_"+str(value4)+".csv",
            "/pizza/data/answer/ch1_new_test_set_"+str(value4)+".csv",
            #dellist,
            1000, 0.07, 7, True)
"""
"""
os.makedirs("data/" + str(round_num) + "/J4condor/result/)
results =  glob(filePath+"data/prediction/svm_result"+str(value1)+"_"+str(value4)+".csv")
for result in results:
    f_name = os.path.basename(result)
    shutil.move(result,"./result/" + f_name)
"""
