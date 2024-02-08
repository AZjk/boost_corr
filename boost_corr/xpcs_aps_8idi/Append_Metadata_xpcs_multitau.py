import sys
import h5py
import math
import os
import shutil
import numpy as np


def append_qmap(meta_type, *args, **kwargs):
    assert meta_type in ('nexus', 'legacy')

    if meta_type == 'legacy':
        append_qmap_legacy(*args, **kwargs)
    elif meta_type == 'nexus':
        append_qmap_nexus(*args, **kwargs)


def append_qmap_legacy(meta_fname, qmap_fname, output_fname, **kwargs):
    copy_metadata_legacy(meta_fname, output_fname)
    copy_qmap(output_fname, qmap_fname)
    copy_avg_stride(output_fname, kwargs['avg_frame'], kwargs['stride_frame'])
     

def append_qmap_nexus(meta_fname, qmap_fname, output_fname, **kwargs):
    shutil.copy(meta_fname, output_fname)
    copy_qmap(output_fname, qmap_fname)
    copy_avg_stride(output_fname, kwargs['avg_frame'], kwargs['stride_frame'])


def copy_qmap(output_fname, qmap_fname, entry='/xpcs'):
    qmap_file = h5py.File(qmap_fname, "r")
    output_file = h5py.File(output_fname, "r+")
    qmap_file.copy("/data/dphival", output_file, name=entry+"/dphilist")
    qmap_file.copy("/data/dphispan", output_file, name=entry+"/dphispan")
    qmap_file.copy("/data/dqval", output_file, name=entry+"/dqlist")
    qmap_file.copy("/data/dynamicMap", output_file, name=entry+"/dqmap")
    qmap_file.copy("/data/mask", output_file, name=entry+"/mask")
    qmap_file.copy("/data/dqspan", output_file, name=entry+"/dqspan")
    qmap_file.copy("/data/dnoq", output_file, name=entry+"/dnoq")
    qmap_file.copy("/data/dnophi", output_file, name=entry+"/dnophi")

    qmap_file.copy("/data/sphival", output_file, name=entry+"/sphilist")
    qmap_file.copy("/data/sphispan", output_file, name=entry+"/sphispan")
    qmap_file.copy("/data/sqval", output_file, name=entry+"/sqlist")
    qmap_file.copy("/data/staticMap", output_file, name=entry+"/sqmap")
    qmap_file.copy("/data/sqspan", output_file, name=entry+"/sqspan")
    qmap_file.copy("/data/snoq", output_file, name=entry+"/snoq")
    qmap_file.copy("/data/snophi", output_file, name=entry+"/snophi")

    output_file[entry+"/qmap_hdf5_filename"] = qmap_fname

    output_file.close()
    qmap_file.close()


def copy_avg_stride(output_fname, avg_frame=1, stride_frame=1,
                    avg_frame_burst=1, stride_frame_burst=1):
    # temp = output_file.create_dataset(
    #     entry+"/stride_frames", (1, 1), dtype='uint64')
    # temp[(0, 0)] = stride_frame
    # temp = output_file.create_dataset(
    #     entry+"/stride_frames_burst", (1, 1), dtype='uint64')
    # temp[(0, 0)] = 1

    # temp = output_file.create_dataset(
    #     entry+"/avg_frames", (1, 1), dtype='uint64')
    # temp[(0, 0)] = avg_frame
    # temp = output_file.create_dataset(
    #     entry+"/avg_frames_burst", (1, 1), dtype='uint64')
    # temp[(0, 0)] = 1

    with h5py.File(output_fname, 'r+') as f:
        f['/xpcs/avg_frames'] = avg_frame
        f['/xpcs/stride_frames'] = stride_frame
        f['/xpcs/avg_frame_burst'] = avg_frame_burst
        f['/xpcs/stride_frame_burst'] = stride_frame_burst


def copy_metadata_legacy(meta_fname, output_fname, 
                         entry='/xpcs', entry_out='/exchange',
                         analysis_type='Multitau'):

    # Open the three .h5 files
    meta_file = h5py.File(meta_fname, "r")
    output_file = h5py.File(output_fname, "w")

    # Copy /measurement from meta_file into outputfile /measurement
    meta_file.copy('/measurement', output_file)

    ############################################################
    # flatfield file for Lambda (only detector with flatfield right now)
    # if meta_file["/measurement/instrument/detector/manufacturer"][()] == "LAMBDA":
    #     flatfieldfile = h5py.File(
    #         "/home/beams/8IDIUSER/Python_HDF5_DataExchange/Flatfield_AsKa_Th5p5keV.hdf", "r")
    #     flatfieldfile.copy("/flatField_transpose", output_file,
    #                        name="/measurement/instrument/detector/flatfield")
    #     flatfieldfile.close()
    ############################################################

    ############################################################
    # this is a less elegant way to define strings, works but has some unknown
    # issues with HDF5
    # Filling in datasets for /xpcs in output file (alphabetical order)
    #dt = h5py.special_dtype(vlen=unicode)
    #temp = output_file.create_dataset(entry+"/analysis_type", (1,1),dtype=dt)
    #temp[(0,0)] = "DYNAMIC"
    ############################################################

    # better example to write strings instead of the above (as per Nick)
    # Version# history: Started with no field like that, then added 0.5 which
    # was when stride and sum frames were added
    # Increasing to 1.0 when adding Normalize frames by framesum, two time, etc
    output_file[entry+"/Version"] = "1.0"
    output_file[entry+"/analysis_type"] = analysis_type

    temp = output_file.create_dataset(entry+"/batches", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1

    meta_file.copy("/measurement/instrument/detector/blemish_enabled",
                   output_file, name=entry+"/blemish_enabled")
    meta_file.copy("/measurement/instrument/acquisition/compression",
                   output_file, name=entry+"/compression")
    meta_file.copy("/measurement/instrument/acquisition/dark_begin",
                   output_file, name=entry+"/dark_begin")
    meta_file.copy("/measurement/instrument/acquisition/dark_begin",
                   output_file, name=entry+"/dark_begin_todo")
    meta_file.copy("/measurement/instrument/acquisition/dark_end",
                   output_file, name=entry+"/dark_end")
    meta_file.copy("/measurement/instrument/acquisition/dark_end",
                   output_file, name=entry+"/dark_end_todo")
    meta_file.copy("/measurement/instrument/acquisition/data_begin",
                   output_file, name=entry+"/data_begin")
    meta_file.copy("/measurement/instrument/acquisition/data_begin",
                   output_file, name=entry+"/data_begin_todo")
    meta_file.copy("/measurement/instrument/acquisition/data_end",
                   output_file, name=entry+"/data_end")
    meta_file.copy("/measurement/instrument/acquisition/data_end",
                   output_file, name=entry+"/data_end_todo")

    temp = output_file.create_dataset(
        entry+"/delays_per_level", (1, 1), dtype='uint64')
    temp[(0, 0)] = 4  # default dpl for multitau
    temp = output_file.create_dataset(
        entry+"/delays_per_level_burst", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1
    # meta_file["/measurement/instrument/detector/burst/number_of_bursts"][()]
    # default dpl for multitau in the burst mode when applicable


    data_begin_todo = int(output_file[entry+"/data_begin_todo"][()])
    data_end_todo = int(output_file[entry+"/data_end_todo"][()])

    static_mean_window = max(math.floor(
        (data_end_todo - data_begin_todo + 1) / 10), 2)
    dynamic_mean_window = max(math.floor(
        (data_end_todo - data_begin_todo + 1) / 10), 2)

    temp = output_file.create_dataset(
        entry+"/dynamic_mean_window_size", (1, 1), dtype='uint64')
    temp[(0, 0)] = dynamic_mean_window

    meta_file.copy("/measurement/instrument/detector/flatfield_enabled",
                   output_file, name=entry+"/flatfield_enabled")
    meta_file.copy("/measurement/instrument/acquisition/datafilename",
                   output_file, name=entry+"/input_file_local")

    # build input_file_remote path
    parent = meta_file["/measurement/instrument/acquisition/parent_folder"][()]
    datafolder = meta_file["/measurement/instrument/acquisition/data_folder"][()]
    datafilename = meta_file["/measurement/instrument/acquisition/datafilename"][()]
    input_file_remote = os.path.join(parent, datafolder, datafilename)
    output_file[entry+"/input_file_remote"] = input_file_remote

    meta_file.copy("/measurement/instrument/detector/kinetics_enabled",
                   output_file, name=entry+"/kinetics")
    meta_file.copy("/measurement/instrument/detector/lld",
                   output_file, name=entry+"/lld")

    output_file[entry+"/normalization_method"] = "TRANSMITTED"
    output_file[entry+"/output_data"] = entry_out

    meta_file.copy("/measurement/instrument/acquisition/datafilename",
                   output_file, name=entry+"/output_file_local")

    output_file[entry+"/output_file_remote"] = "output/results"

    meta_file.copy("/measurement/instrument/detector/sigma",
                   output_file, name=entry+"/sigma")
    meta_file.copy("/measurement/instrument/acquisition/specfile",
                   output_file, name=entry+"/specfile")
    meta_file.copy("/measurement/instrument/acquisition/specscan_dark_number",
                   output_file, name=entry+"/specscan_dark_number")
    meta_file.copy("/measurement/instrument/acquisition/specscan_data_number",
                   output_file, name=entry+"/specscan_data_number")

    temp = output_file.create_dataset(
        entry+"/static_mean_window_size", (1, 1), dtype='uint64')
    temp[(0, 0)] = static_mean_window

   

    temp = output_file.create_dataset(entry+"/swbinX", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1

    temp = output_file.create_dataset(entry+"/swbinY", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1

    #######################
    temp = output_file.create_dataset(
        entry+"/normalize_by_framesum", (1, 1), dtype='uint64')
    temp[(0, 0)] = 0

    temp = output_file.create_dataset(
        entry+"/normalize_by_smoothed_img", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1

    output_file[entry+"/smoothing_method"] = "symmetric"
    output_file[entry+"/smoothing_filter"] = "None"

    temp = output_file.create_dataset(
        entry+"/num_g2partials", (1, 1), dtype='uint64')
    temp[(0, 0)] = 1

    temp = output_file.create_dataset(
        entry+"/twotime2onetime_window_size", (1, 1), dtype='uint64')
    temp[(0, 0)] = dynamic_mean_window
    ##temp[(0,0)] = 300

    # direct multitau or twotime analysis (for multitau, set max_bins=1,
    # set bin_stride as needed)
    max_bins = 1
    bin_stride = 1
    temp_bins = np.arange(1, max_bins+1, bin_stride)
    temp = output_file.create_dataset(
        entry+"/qphi_bin_to_process", (temp_bins.size, 1), dtype='uint64')
    temp[:, 0] = temp_bins

    #######################
    # Close all the files
    meta_file.close()
    output_file.close()
    #######################

    return


if __name__ == '__main__':
    append_qmap(sys.argv[0], sys.argv[1], sys.argv[2])
