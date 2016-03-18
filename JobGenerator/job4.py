def params(j1_type):
    if j1_type == 'a':
        values = [
                (p, q)
                # previous feature range
                #for p in range(294,1248) ## 294~1248
                for p in range(303, 1294) ## 294~124a
                for q in range(0, 10)
                ]
    else:
        values = [
                (p, q)
                for p in range(294,911) ## 294~881
                for q in range(0, 10)
                ]


    return values
