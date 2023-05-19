OUTPUT_DIR=/clhome/MQICHU/ssd/cluster_results
QMAP=/data/xpcs8//partitionMapLibrary/2021-1/cssiTest202103_qmap_Aerogel_good_1.h5

for i in $(seq 4 6);
do 
  idx=$(printf "%03d" $i)
  echo " "
  echo $idx
   boost_corr \
       --gpu_id 0    \
       --raw /data/xpcs8/APSU_TestData_202106/APSU_TestData_$idx/APSU_TestData_$idx.h5\
       --qmap $QMAP \
       --output $OUTPUT_DIR \
       --verbose
done
