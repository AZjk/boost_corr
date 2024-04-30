import sys
import h5py
import math
import os
import shutil
import numpy as np


def append_qmap(meta_type, meta_fname, qmap_fname, output_fname,
                avg_frame=1, stride_frame=1,
                analysis_type="Multitau"):
    assert meta_type in ('nexus', 'legacy')
    if meta_type == 'legacy':
        copy_metadata_legacy(meta_fname, output_fname,
                             analysis_type=analysis_type)
    elif meta_type == 'nexus':
        shutil.copy(meta_fname, output_fname)

    # copy_qmap(output_fname, qmap_fname)
    with h5py.File(qmap_fname, 'r') as fs, h5py.File(output_fname, 'r+') as fd:
        if '/entry/instrument' not in fd:
            fd.create_dataset('/entry/instrument/')
        fs.copy('/entry/instrument/mask/', fd['/entry/instrument/'])

    copy_additional_metadata(output_fname, avg_frame=avg_frame,
                             stride_frame=stride_frame,
                             analysis_type=analysis_type)


def copy_additional_metadata(output_fname, avg_frame=1, stride_frame=1,
                             avg_frame_burst=1, stride_frame_burst=1,
                             analysis_type="Multitau"):
    with h5py.File(output_fname, 'r+') as f:
        f['/xpcs/avg_frames'] = avg_frame
        f['/xpcs/stride_frames'] = stride_frame
        f['/xpcs/avg_frame_burst'] = avg_frame_burst
        f['/xpcs/stride_frame_burst'] = stride_frame_burst

        # legacy metadata file may already have analysis_type
        if "/xpcs/analysis_type" in f:
            del f["/xpcs/analysis_type"]
        f["/xpcs/analysis_type"] = analysis_type


def copy_metadata_legacy(meta_fname, output_fname, 
                         entry='/xpcs', entry_out='/exchange',
                         analysis_type='Multitau'):

    # Open the three .h5 files
    meta_file = h5py.File(meta_fname, "r")
    output_file = h5py.File(output_fname, "w")

    # Copy /measurement from meta_file into outputfile /measurement
    meta_file.copy('/measurement', output_file)

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
