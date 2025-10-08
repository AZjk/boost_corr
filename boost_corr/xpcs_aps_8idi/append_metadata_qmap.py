import os
import sys
import shutil
import h5py


def append_metadata_qmap(output_fname, meta_fname=None, qmap_fname=None,
                         meta_type='nexus'):
    """
    Appends metadata to an output file.
    Parameters
    ----------
    output_fname : str
        The name of the output file.
    meta_fname : str, optional
        The name of the metadata file.
    qmap_fname : str, optional
        The name of the qmap file.
    meta_type : str, optional
        The type of metadata. Default is 'nexus'.
    """
    assert meta_type in ('nexus')
    # copy the metadata file
    shutil.copyfile(meta_fname, output_fname)
    os.chmod(output_fname, 0o644)
    # copy_qmap(output_fname, qmap_fname)
    copy_qmap(source_file=qmap_fname, target_file=output_fname)


def copy_qmap(source_file, target_file, source_group='/qmap',
              target_group='/xpcs/qmap'):
    """
    Copies a group from one HDF5 file to another with the option to rename the group.

    Parameters:
    -----------
        source_file (str): Path to the source HDF5 file.
        target_file (str): Path to the target HDF5 file.
        source_group (str): Name of the source group to copy.
        target_group (str): Name of the target group in the target file.

    Returns:
    ----------
        None
    """
    with h5py.File(source_file, "r") as src, h5py.File(target_file, "a") as tgt:
        src.copy(source_group, tgt, name=target_group)



if __name__ == '__main__':
    append_metadata_qmap(sys.argv[0], sys.argv[1], sys.argv[2])
