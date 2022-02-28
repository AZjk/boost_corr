# imm example
echo "*********************Lambda IMM example*********************************"
python gpu_corr.py \
    --batch_size 1024 \
    --gpu_id 2    \
    --raw ~/ssd/xpcs_data_raw/A005_Dragonite_25p_Quiescent_att0_Lq0_001/\
A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm \
    --qmap ~/ssd/xpcs_data_raw/qmap/harden201912_qmap_Dragonite_Lq0_S270_D54.h5 \
    --output /clhome/MQICHU/ssd/cluster_results \
    --verbose

# rigaku example
# echo " "
# echo "***********************Rigaku example***********************************"
# python gpu_corr.py \
#     --batch_size 256 \
#     -i 1    \
#     -r ~/ssd/xpcs_data_raw/O018_Silica_D100_att0_Rq0_00001/O018_Silica_D100_att0_Rq0_00001.bin \
#     -q ~/ssd/xpcs_data_raw/qmap/qzhang20191006_qmap_Rigaku_S270_D27_log.h5 \
#     --output /clhome/MQICHU/ssd/cluster_results \
#     --verbose

# rigaku example
echo " "
echo "***********************Rigaku example***********************************"
python gpu_corr.py \
    --batch_size 128 \
    -i 2    \
    -r /local/data_miaoqi/xpcs_data_raw/I026_D100_2mmCap_250C10p_att01_Rq0_00001/I026_D100_2mmCap_250C10p_att01_Rq0_00001.bin\
    -q /local/data_miaoqi/xpcs_data_raw/qmap/babnigg202107_2_qmap_Rq0_S180_D18_Linear.h5 \
    --output /clhome/MQICHU/ssd/cluster_results \
    --verbose

# hdf example
echo " "
echo "***********************HDF example***********************************"
python gpu_corr.py \
    --batch_size 2048 \
    -i 2    \
    -r ~/ssd/xpcs_data_raw/A003_Cu3Au_att0_001/A003_Cu3Au_att0_001.imm \
    -q ~/ssd/xpcs_data_raw/qmap/ycao201910_qmap_E010_1.h5 \
    -o /clhome/MQICHU/ssd/cluster_results \
    --verbose
