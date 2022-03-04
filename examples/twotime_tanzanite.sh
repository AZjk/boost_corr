#!/bin/bash
# perform aps xpcs twotime correlation on real data
boost_corr \
	-t 'Twotime' \
	-i -1 \
	-r /data/xpcs8/2022-1/leheny202202/A056_Ludox15_att00_L2M_quiescent_001/A056_Ludox15_att00_L2M_quiescent_001_001..h5 \
	-q /data/xpcs8/partitionMapLibrary/2022-1/leheny202202_qmap_2M_Test_S360_D60_A009.h5 \
	-o /data/xpcs8/2022-1/leheny202202/cluster_results_RealTime -s sqmap -v \
	-avg_frame 3 \
	-dq "1-60"

