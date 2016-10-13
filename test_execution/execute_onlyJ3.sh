i=0
end=44
while [ $i -le $end ]
do
    condor_submit 'submit/7/3.'$i'.submit'
    i=$(($i++1))
    echo $i
done
