import sys
import shutil
import h5py


def append_qmap(output_fname,
                meta_fname=None,
                qmap_fname=None,
                meta_type='nexus',
                avg_frame=1, stride_frame=1,
                analysis_type="Multitau"):
    assert meta_type in ('nexus')
    # copy the metadata file
    shutil.copy(meta_fname, output_fname)
    # copy_qmap(output_fname, qmap_fname)
    copy_qmap(source_file=qmap_fname, target_file=output_fname)
    copy_additional_metadata(output_fname, avg_frame=avg_frame,
                             stride_frame=stride_frame,
                             analysis_type=analysis_type)


def copy_qmap(source_file, target_file, source_group='/qmap', 
              target_group='/xpcs/qmap'):
    """
    Copies a group from one HDF5 file to another with the option to rename the group.

    Args:
        source_file (str): Path to the source HDF5 file.
        target_file (str): Path to the target HDF5 file.
        source_group (str): Name of the source group to copy.
        target_group (str): Name of the target group in the target file.
    
    Returns:
        None
    """
    with h5py.File(source_file, "r") as src, h5py.File(target_file, "a") as tgt:
        src.copy(source_group, tgt, name=target_group)


def copy_additional_metadata(output_fname, avg_frame=1, stride_frame=1,
                             avg_frame_burst=1, stride_frame_burst=1,
                             analysis_type="Multitau"):
    with h5py.File(output_fname, 'r+') as f:
        f['/xpcs/qmap/avg_frames'] = avg_frame
        f['/xpcs/qmap/stride_frames'] = stride_frame
        f['/xpcs/qmap/avg_frame_burst'] = avg_frame_burst
        f['/xpcs/qmap/stride_frame_burst'] = stride_frame_burst
        f["/xpcs/qmap/analysis_type"] = analysis_type


if __name__ == '__main__':
    append_qmap(sys.argv[0], sys.argv[1], sys.argv[2])
