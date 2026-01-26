# boost-corr

A high-performance correlation (multi-tau/two-time) package for X-ray Photon Correlation Spectroscopy (XPCS) running on GPU and CPU.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

## Features

- **High Performance**: GPU-accelerated correlation computation using PyTorch
- **Flexible**: Supports both multi-tau and two-time correlation analysis
- **Multiple Formats**: Handles IMM, Rigaku, HDF5, and Timepix4 data formats
- **Timepix4 Support**: Native support for Timepix4 detectors with configurable time binning
- **CPU Fallback**: Automatic fallback to CPU when GPU is unavailable
- **Command-line Interface**: Easy-to-use CLI for batch processing
- **Python API**: Programmatic access for custom workflows

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch (with CUDA support for GPU acceleration)

### Step 1: Create Virtual Environment

Create a new virtual environment using conda (recommended):

```bash
# Create environment
conda create -n boost_corr python=3.12

# Activate environment
conda activate boost_corr
```

Alternatively, use venv:

```bash
python -m venv boost_corr_env
source boost_corr_env/bin/activate  # On Linux/Mac
# or
boost_corr_env\Scripts\activate  # On Windows
```

### Step 2: Install boost-corr

#### From PyPI (Stable)

```bash
pip install boost-corr
```

#### From Source (Development)

```bash
git clone https://github.com/AdvancedPhotonSource/boost_corr.git
cd boost_corr
pip install -e .
```


## Usage

### Command-Line Interface

#### Multi-tau Correlation Example

Using GPU 0, with frame stride of 3 and averaging every 3 frames:

```bash
boost_corr -t Multitau -i 0 \
  -r /data/A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm \
  -q /data/qmap/harden201912_qmap_Dragonite_Lq0_S270_D54.h5 \
  -o /output \
  -f 3 -a 3 \
  -v
```

#### Two-time Correlation Example

Using CPU with sqmap smoothing, averaging every 3 frames:

```bash
boost_corr -t Twotime -i -1 \
  -r /data/A056_Ludox15_att00_L2M_quiescent_001_001.h5 \
  -q /data/qmap/leheny202202_qmap_2M_Test_S360_D60_A009.h5 \
  -o /output \
  -s sqmap \
  -a 3 \
  -d "1-60" \
  -v
```

#### Using Configuration File

```bash
boost_corr -c config.json
```

Example `config.json`:
```json
{
  "raw": "/data/sample_001.h5",
  "qmap": "/data/qmap.h5",
  "output": "/results",
  "type": "Multitau",
  "gpu_id": 0,
  "verbose": true
}
```

### Command-Line Options

```
usage: boost_corr [-h] -r RAW_FILENAME [-q QMAP_FILENAME] [-o OUTPUT_DIR]
                  [-s SMOOTH] [-i GPU_ID] [-nf {0,1}] [-b BEGIN_FRAME]
                  [-e END_FRAME] [-f STRIDE_FRAME] [-a AVG_FRAME] [-t TYPE]
                  [-d DQ_SELECTION] [-v] [-G] [-n] [-np NUM_PARTIAL_G2]
                  [--crop-ratio-threshold CROP_RATIO_THRESHOLD] [-p PREFIX]
                  [-u SUFFIX] [--bin-time-s BIN_TIME_S]
                  [--run-config-path RUN_CONFIG_PATH] [-w] [-c CONFIG_JSON]

Options:
  -h, --help            Show this help message and exit
  -r, --raw             Raw data file (imm/rigaku/hdf) [REQUIRED]
  -q, --qmap            Q-map file (h5/hdf)
  -o, --output          Output directory [default: cluster_results]
  -s, --smooth          Smoothing method for two-time correlation [default: sqmap]
  -i, --gpu-id          GPU selection: -1=CPU, -2=auto, >=0=specific GPU [default: -1]
  -nf, --normalize-frame  Frame normalization: 0=disable, 1=enable [default: 1]
  -b, --begin-frame     Starting frame index (0-based) [default: 0]
  -e, --end-frame       Ending frame index (-1=all frames) [default: -1]
  -f, --stride-frame    Frame stride for processing [default: 1]
  -a, --avg-frame       Number of frames to average [default: 1]
  -t, --type            Analysis type: Multitau, Twotime, or Both [default: Multitau]
  -d, --dq-selection    DQ selection (e.g., "1,2,5-7" or "all") [default: all]
  -v, --verbose         Enable verbose output
  -G, --save-G2         Save G2, IP, and IF to file
  -n, --dry-run         Show arguments without executing
  -np, --num-partial-g2 Number of partial G2 to compute [default: 0]
  --crop-ratio-threshold Threshold for valid pixel ratio to trigger cropping [default: 0.5]
  -p, --prefix          Prefix for result filename
  -u, --suffix          Suffix for result filename
  --bin-time-s          Time bin size in seconds for Timepix4 data [default: 1e-6]
  --run-config-path     Path to the run configuration file for Timepix4 data
  -w, --overwrite       Overwrite existing result files
  -c, --config          Configuration JSON file path
```

### Python API

#### Basic Multi-tau Correlation

```python
import torch
from boost_corr import MultitauCorrelator

# Check version
import boost_corr
print(f"boost-corr version: {boost_corr.__version__}")

# Setup
frame_num = 1024
det_size = (128, 128)
device = 'cuda:0'  # Use 'cpu' for CPU-only

# Create correlator
mc = MultitauCorrelator(frame_num=frame_num, det_size=det_size, device=device)

# Process frames
for n in range(frame_num):
    # Generate or load frame data
    frame = torch.rand(det_size, device=device).reshape(1, -1)
    mc.process(frame)

# Get results
mc.post_process()
result = mc.get_results()

print(f"Correlation shape: {result['g2'].shape}")
```

#### Two-time Correlation

```python
from boost_corr import TwotimeCorrelator

# Create two-time correlator
tc = TwotimeCorrelator(frame_num=frame_num, det_size=det_size, device=device)

# Process frames
for n in range(frame_num):
    frame = torch.rand(det_size, device=device).reshape(1, -1)
    tc.process(frame)

# Get results
tc.post_process()
result = tc.get_results()
```

#### Using with Real XPCS Data

```python
from boost_corr.xpcs_aps_8idi.gpu_corr_multitau import solve_multitau

result = solve_multitau(
    raw='/data/sample_001.h5',
    qmap='/data/qmap.h5',
    output='/results',
    gpu_id=0,
    begin_frame=0,
    end_frame=-1,
    stride_frame=1,
    avg_frame=1,
    verbose=True
)
```

## Timepix4 Detector Support

boost-corr provides native support for Timepix4 detectors with advanced features:

### Basic Timepix4 Usage

```bash
boost_corr -t Multitau -i 0 \
  -r /data/sample_001.tpx \
  -q /data/qmap.h5 \
  -o /output \
  -v
```

### Multi-chip Timepix4 Configuration

For multi-chip setups (e.g., `.tpx.000`, `.tpx.001`, `.tpx.002`), provide a run configuration file:

```bash
boost_corr -t Multitau -i 0 \
  -r /data/sample_001.tpx.000 \
  -q /data/qmap.h5 \
  -o /output \
  --run-config-path /data/run_config.json \
  -v
```

The run configuration file specifies chip layout and time binning parameters. See the [timepix_dataset](https://github.com/AZjk/timepix_dataset) package for configuration details.

### Key Features

- **Sparse Data Handling**: Efficient processing of photon-counting sparse data
- **Time Binning**: Configurable time binning (default: 1 μs)
- **Memory Optimization**: Automatic GPU/CPU memory management based on data size
- **bfloat16 Precision**: Optimized data type for GPU performance

## GPU Scheduling

For automatic GPU selection on multi-GPU systems:

```bash
boost_corr -i -2 -r data.h5 -q qmap.h5
```

This will automatically select an available GPU with sufficient memory.

## Performance Tips

1. **Use GPU**: GPU acceleration provides 10-100x speedup over CPU
2. **Batch Processing**: Use frame averaging (`-a`) to reduce memory usage
3. **Frame Stride**: Use stride (`-f`) to skip frames for faster processing
4. **Memory**: Monitor GPU memory usage for large datasets

## Supported Data Formats

- **HDF5**: Standard XPCS HDF5 format (`.h5`, `.hdf`, `.hdf5`)
- **IMM**: APS 8-ID-I IMM format (`.imm`)
- **Rigaku**: Rigaku detector format (`.bin`, `.bin.000`)
- **Timepix4**: Amsterdam Scientific Instruments Timepix4 detector (`.tpx`, `.tpx.000`, `.tpx.001`, `.tpx.002`)
  - Supports single and multi-chip configurations
  - Configurable time binning for photon-counting data
  - Automatic sparse-to-dense conversion with bfloat16 optimization

## Output Files

Results are saved in the specified output directory:
- `*_multitau.h5`: Multi-tau correlation results
- `*_twotime.h5`: Two-time correlation results



## Citation

If you use boost-corr in your research, please cite:

```bibtex
@software{boost_corr,
  author = {Chu, Miaoqi},
  title = {boost-corr: High-performance XPCS correlation analysis},
  url = {https://github.com/AdvancedPhotonSource/boost_corr},
  year = {2022}
}
```

## License

Copyright (c) 2026, UChicago Argonne, LLC. All rights reserved.

This software is distributed under a 3-clause BSD license. See [LICENSE](LICENSE) for details.

## Credits

This package was developed at Argonne National Laboratory for the Advanced Photon Source.

## Support

- **Issues**: [GitHub Issues](https://github.com/AdvancedPhotonSource/boost_corr/issues)
- **Documentation**: [Read the Docs](https://boost-corr.readthedocs.io) (coming soon)
- **Contact**: mqichu@anl.gov
