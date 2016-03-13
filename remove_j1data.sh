flag=$1
if [ $flag = a ]; then
    rm data/0/J1condor/includeTestSamples_1a/set*/include*1b.libfm
elif [ $flag = b ]; then
    rm data/0/J1condor/includeTestSamples_1a/set*/include*1a.libfm
fi
