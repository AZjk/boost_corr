#!/bin/bash

# perform aps xpcs multitau correlation on real data
DATA_DIR='/scratch'

# imm example
echo "*********************Lambda IMM example*********************************"
boost_corr \
    --gpu_id 0    \
    --raw ${DATA_DIR}/xpcs_data_raw/A005_Dragonite_25p_Quiescent_att0_Lq0_001/\
A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm \
    --qmap ${DATA_DIR}/xpcs_data_raw/qmap/harden201912_qmap_Dragonite_Lq0_S270_D54.h5 \
    --output ${DATA_DIR}/cluster_results \
    --save_G2 \
    --verbose

# rigaku example
# echo " "
echo "***********************Rigaku example***********************************"
boost_corr \
    -i 1    \
    -r ${DATA_DIR}/xpcs_data_raw/O018_Silica_D100_att0_Rq0_00001/O018_Silica_D100_att0_Rq0_00001.bin \
    -q ${DATA_DIR}/xpcs_data_raw/qmap/qzhang20191006_qmap_Rigaku_S270_D27_log.h5 \
    --output ${DATA_DIR}/cluster_results \
    --verbose

# rigaku example
echo " "
echo "***********************Rigaku example***********************************"
boost_corr \
    -i 0    \
    -r ${DATA_DIR}/xpcs_data_raw/I026_D100_2mmCap_250C10p_att01_Rq0_00001/I026_D100_2mmCap_250C10p_att01_Rq0_00001.bin\
    -q ${DATA_DIR}/xpcs_data_raw/qmap/babnigg202107_2_qmap_Rq0_S180_D18_Linear.h5 \
    --output ${DATA_DIR}/cluster_results \
    --verbose

# hdf example
echo " "
echo "***********************HDF example***********************************"
boost_corr \
    -i 0    \
    -r ${DATA_DIR}/xpcs_data_raw/A003_Cu3Au_att0_001/A003_Cu3Au_att0_001.imm \
    -q ${DATA_DIR}/xpcs_data_raw/qmap/ycao201910_qmap_E010_1.h5 \
    -o ${DATA_DIR}/cluster_results \
    --verbose

# imm with phi bins example
echo " "
echo "***********************IMM example***********************************"
boost_corr \
    -i 0    \
    -r ${DATA_DIR}/xpcs_data_raw/B981_2_10k_star_dynamic_0p1Hz_Strain1.01mm_Ampl0.005mm_att5_Lq0_001/B981_2_10k_star_dynamic_0p1Hz_Strain1.01mm_Ampl0.005mm_att5_Lq0_001_00001-00800.imm \
    -q ${DATA_DIR}/xpcs_data_raw/qmap/foster202110_qmap_RubberDonut_Lq0_S180_18_D18_18.h5 \
    -o ${DATA_DIR}/cluster_results \
    --verbose
