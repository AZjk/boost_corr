=======================
Boost-Corr User Guide
=======================

:Author: Miaoqi Chu
:Email: mqichu@anl.gov
:Date: October 8, 2025

Overview
========

Boost-Corr is a high-performance tool for computing Multi-tau and Twotime correlation functions for APS-8IDI XPCS (X-ray Photon Correlation Spectroscopy) datasets. The tool supports both GPU and CPU processing and can handle various data formats including IMM, Rigaku, and HDF5 files.

Key Features
============

* **Multi-tau correlation analysis**: Fast computation of intensity correlation functions
* **Twotime correlation analysis**: Two-time correlation function computation with smoothing options
* **GPU acceleration**: CUDA support for high-performance processing
* **Multiple data formats**: Support for IMM, Rigaku (.bin), and HDF5 (.h5) files
* **Flexible frame processing**: Frame averaging, striding, and range selection
* **Configuration files**: JSON-based configuration for reproducible analysis
* **Batch processing**: Automated GPU scheduling for multiple jobs

Basic Usage
===========

The basic syntax for boost-corr is::

    boost_corr_bin -r <raw_data_file> -q <qmap_file> [options]

.. note::
   On APS machines, boost-corr is installed as user ``8idiuser`` and can be launched using the command ``boost_corr_bin``.

Required Arguments
==================

``-r, --raw RAW_FILENAME``
    Path to the raw data file. Supported formats:
    
    * ``.imm`` - Lambda detector IMM files
    * ``.bin`` or ``.bin.000`` - Rigaku detector (500k and 3M) binary files  
    * ``.h5`` - HDF5 files

``-q, --qmap QMAP_FILENAME``
    Path to the q-map file (HDF5 format) that defines the scattering vector partitions for analysis.

Analysis Types
==============

Boost-corr supports three types of correlation analysis:

Multi-tau Analysis (Default)
------------------

Multi-tau correlation computes the normalized intensity correlation function g₂(q,τ) as a function of delay time τ::

    boost_corr_bin -r data.imm -q qmap.h5 -t Multitau

Twotime Analysis  
----------------

Twotime correlation computes the two-time correlation function C(q,t₁,t₂)::

    boost_corr_bin -r data.h5 -q qmap.h5 -t Twotime

Note the results of twotime analysis scales as ``n^2``. Use this analysis only for datasets with reasonable number of frames.

Both Analysis
-------------

Performs both multi-tau and twotime analysis in a single run::

    boost_corr_bin -r data.imm -q qmap.h5 -t Both

Command Line Options
====================

Output and File Management
---------------------------

``-o, --output OUTPUT_DIR``
    Output directory for result files (default: ``cluster_results``)

``-w, --overwrite``
    Overwrite existing result files

``-p, --prefix PREFIX``
    Add prefix to result filename

``-u, --suffix SUFFIX``
    Add suffix to result filename

``-G, --save-G2``
    Save G2, IP (intensity), and IF (intensity fluctuation) data to file

GPU and Performance
-------------------

``-i, --gpu-id GPU_ID``
    GPU selection options:
    
    * ``-1``: Use CPU only
    * ``-2``: Auto-scheduling (automatically select available GPU)
    * ``≥0``: Use specific GPU ID (e.g., 0, 1, 2...)

Frame Processing
----------------

``-b, --begin-frame FRAME``
    Starting frame index (0-based, default: 0). Use negative values for Python-style slicing from the end.

``-e, --end-frame FRAME``
    Ending frame index (0-based, exclusive, default: -1 for all frames)

``-f, --stride-frame STRIDE``
    Frame stride for processing (default: 1)

``-a, --avg-frame COUNT``
    Number of frames to average before correlation (default: 1)

``-nf, --normalize-frame {0,1}``
    Enable (1) or disable (0) frame-based normalization (default: 1)

Analysis Parameters
-------------------

``-s, --smooth SMOOTH``
    Smoothing method for Twotime correlation (default: ``sqmap``)

``-d, --dq-selection DQ_SELECTION``
    DQ list selection. Examples:
    
    * ``"all"``: Use all dynamic q indices (default)
    * ``"1,2,5-7"``: Select specific indices [1,2,5,6,7]
    * ``"10-20"``: Select range [10,11,12,...,20]

Utility Options
---------------

``-v, --verbose``
    Enable verbose output for detailed logging

``-n, --dry-run``
    Show arguments without executing (useful for testing configurations)

``-c, --config CONFIG_JSON``
    Load configuration from JSON file (command line arguments override config values)

Configuration Files
===================

Boost-corr supports JSON configuration files for reproducible analysis. Create a configuration file with your preferred settings:

Example Configuration
---------------------

.. code-block:: json

    {
      "qmap": "/path/to/qmap.h5",
      "output": "/path/to/results",
      "smooth": "sqmap",
      "gpu_id": 0,
      "begin_frame": 1,
      "end_frame": -1,
      "stride_frame": 1,
      "avg_frame": 1,
      "type": "Multitau",
      "dq_selection": "all",
      "verbose": true,
      "overwrite": false,
      "save_G2": false
    }

Using Configuration Files
--------------------------

Load a configuration file with the ``-c`` option::

    boost_corr_bin -r data.imm -c config.json

Command line arguments will override configuration file values, allowing you to modify specific parameters while keeping the rest of your configuration intact.

Usage Examples
==============

Basic Multi-tau Analysis
-------------------------

Perform multi-tau correlation on an IMM file::

    boost_corr_bin \
        -r /data/sample_001.imm \
        -q /data/qmap.h5 \
        -o /results \
        -i 0 \
        --verbose

Twotime Analysis with Frame Selection
-------------------------------------

Analyze frames 100-500 with frame averaging::

    boost_corr_bin \
        -r /data/sample.h5 \
        -q /data/qmap.h5 \
        -t Twotime \
        -b 100 \
        -e 500 \
        -a 3 \
        -d "1-60" \
        --verbose

CPU-only Analysis
-----------------

Run analysis on CPU (useful when GPUs are unavailable)::

    boost_corr_bin \
        -r /data/sample.bin \
        -q /data/qmap.h5 \
        -i -1 \
        --verbose

Auto-scheduled GPU Analysis
---------------------------

Let boost-corr automatically select an available GPU::

    boost_corr_bin \
        -r /data/sample.imm \
        -q /data/qmap.h5 \
        -i -2 \
        --save-G2 \
        --verbose

Batch Processing with Configuration
-----------------------------------

Process multiple files using a configuration file::

    # Create config.json with common settings
    boost_corr_bin -r /data/sample1.imm -c config.json
    boost_corr_bin -r /data/sample2.imm -c config.json
    boost_corr_bin -r /data/sample3.imm -c config.json

Data Format Support
===================

IMM Files (Lambda Detector)
----------------------------

Lambda detector IMM files are directly supported::

    boost_corr_bin -r sample_00001-20000.imm -q qmap.h5

Rigaku Files
------------

Rigaku detector binary files (.bin)::

    boost_corr_bin -r sample.bin -q qmap_rigaku.h5

HDF5 Files
----------

HDF5 format files::

    boost_corr_bin -r sample.h5 -q qmap.h5

Output Files
============

Boost-corr generates HDF5 output files containing:

Multi-tau Results
-----------------

* **g2**: Normalized intensity correlation function g₂(q,τ)
* **tau**: Delay time values
* **qvalues**: Scattering vector magnitudes
* **scattering**: Static scattering intensity I(q)

Twotime Results
---------------

* **C2**: Two-time correlation function C(q,t₁,t₂)
* **scattering**: Static scattering intensity I(q)
* **qvalues**: Scattering vector magnitudes

Optional G2 Data (with --save-G2)
----------------------------------

* **IP**: Intensity values
* **IF**: Intensity fluctuation values

Performance Tips
================

GPU Selection
-------------

* Use ``-i -2`` for automatic GPU scheduling in multi-GPU systems
* Monitor GPU memory usage with ``nvidia-smi`` for optimal batch sizes
* Use CPU (``-i -1``) for small datasets or when GPUs are busy

Frame Processing
----------------

* Use frame averaging (``-a``) to reduce noise in low-count data
* Adjust ``begin_frame`` and ``end_frame`` to exclude bad frames
* Use frame striding (``-f``) to reduce computational load for preliminary analysis

Memory Management
-----------------

* For large datasets, consider processing in chunks using frame ranges
* Monitor system memory usage, especially for twotime analysis
* Use appropriate batch sizes based on available GPU memory

Troubleshooting
===============

Common Issues
-------------

**"CUDA out of memory" errors**
    * Reduce batch size or use CPU processing (``-i -1``)
    * Process data in smaller frame ranges
    * Close other GPU-using applications

**"No such file or directory" errors**
    * Verify file paths are correct and accessible
    * Check file permissions
    * Ensure qmap file format is compatible

**Slow performance**
    * Enable GPU acceleration (``-i 0`` or higher)
    * Use frame striding for preliminary analysis
    * Check if other processes are using the GPU

**Empty or incorrect results**
    * Verify qmap file matches your detector configuration
    * Check frame range settings (``begin_frame``, ``end_frame``)
    * Enable verbose output (``-v``) for debugging information

Verbose Output
--------------

Always use ``--verbose`` for detailed logging when troubleshooting::

    boost_corr_bin -r data.imm -q qmap.h5 --verbose

This provides information about:

* Dataset properties (frame count, detector size)
* GPU/CPU device selection
* Processing progress and timing
* Memory usage and performance metrics

Getting Help
============

For additional help and support:

* Use ``boost_corr_bin --help`` to see all available options
* Check the examples directory for sample scripts and configurations
* Enable verbose output for detailed processing information
* Verify your data formats and qmap files are compatible with APS-8IDI standards

Advanced Usage
==============

DQ Selection Patterns
---------------------

The ``--dq-selection`` option supports flexible q-bin selection:

* ``"all"``: All available q-bins
* ``"1,3,5"``: Specific q-bins 1, 3, and 5
* ``"1-10"``: Range from 1 to 10 (inclusive)
* ``"1-5,8,10-15"``: Mixed ranges and individual selections

Frame Indexing
--------------

Frame indices use 0-based indexing:

* ``--begin-frame 0``: Start from first frame
* ``--begin-frame -100``: Start 100 frames from the end
* ``--end-frame -1``: Process until the last frame
* ``--end-frame 1000``: Stop before frame 1000

This user guide covers the essential usage patterns and options for boost-corr. For the most up-to-date information, always refer to the built-in help system and example scripts provided with the software.
