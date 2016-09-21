import json
import os

"""
f = open('JobGenerator/excluded_candidate.txt')
candidate = [int(v) for v in f]
f.close()
"""

all_features = [
        file_name.split('/')[0].split('_',1)[1].split('.')[0]
        for file_name in os.listdir('data/features/')]

def params(j1_type, include_features):
    # test feature list
    include_features = set(include_features)
    test_features = [feature
            for feature in all_features
            if feature not in include_features]

    print test_features
    """
    if j1_type == 'a':
        values = [
                (p, q)
                # previous feature range
                #for p in range(294,1248) ## 294~1248
                #for p in range(303, 1294) ## 294~124a
                for p in candidate
                for q in range(0, 10)
                ]
    else:
        ### Problem 1.b
        values = [
                (p, q)
                for p in range(0,923) ## 294~881
                for q in range(0, 10)
                ]
    """
    values = [
            (feature, set_num)
            for feature in test_features
            for set_num in range(0, 10)
            ]

    return values
