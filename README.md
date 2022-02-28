==========
boost-corr
==========






a high-performance correlation (multi-tau/two-time) package running on GPU and CPU

Installation (updated on 02/28/2022)
--------
* create new virtual environment use conda. Python version is specified here because the pytorch package only supports python 3.7-3.9. You can also use "python -m venv" or "virtualenv" to create a virutal enviroment.

        conda create -n YOUR_ENV_NAME python==3.9.7


* activate your new environment
        
        conda activate YOUR_ENV_NAME

* install the latest pytorch package, following the instructions on [pytorch.org](https://pytorch.org/get-started/locally/)

        pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

* clone and install boost-corr
        
        git clone git@github.com:AZjk/boost_corr.git
        cd boost_corr
        pip install .


Usage
--------
* call boost_corr in command line to compute XPCS correlation for the workflow of APS-8IDI.
* * Twotime correlation example, using CPU and sqmap smooth, average every 3 frames.

        boost_corr -t Twotime -i -1 -r /data/xpcs8/2022-1/leheny202202/A056_Ludox15_att00_L2M_quiescent_001/A056_Ludox15_att00_L2M_quiescent_001_001..h5 -q /data/xpcs8/partitionMapLibrary/2022-1/leheny202202_qmap_2M_Test_S360_D60_A009.h5 -o /scratch -s sqmap -v -avg_frame 3 -dq "1-60"

* * Multitau correlation example, using the first GPU, stride of 3 frames and average every 3 frames.
    
        boost_corr -t Multitau -i 0 -r /scratch/xpcs_data_raw/A005_Dragonite_25p_Quiescent_att0_Lq0_001/ A005_Dragonite_25p_Quiescent_att0_Lq0_001_00001-20000.imm --qmap /scratch/xpcs_data_raw/qmap/harden201912_qmap_Dragonite_Lq0_S270_D54.h5 --output /scratch --verbose -stride_frame 3 -avg_frame 7
    
* * Full list

```
 $ boost_corr --help
 usage: boost_corr [-h] -r RAW_FILENAME -q QMAP_FILENAME [-o OUTPUT_DIR] [-s SMOOTH] [-i GPU_ID] [-begin_frame BEGIN_FRAME] [-end_frame END_FRAME] [-stride_frame STRIDE_FRAME]
                 [-avg_frame AVG_FRAME] [-t TYPE] [-dq TYPE] [--verbose] [--dryrun] [--overwrite] [-c CONFIG.JSON]

 Compute Multi-tau/Twotime correlation for XPCS datasets on GPU/CPU

 optional arguments:
 -h, --help            show this help message and exit
 -r RAW_FILENAME, --raw RAW_FILENAME
                         the filename of the raw data file (imm/rigaku/hdf)
 -q QMAP_FILENAME, --qmap QMAP_FILENAME
                         the filename of the qmap file (h5/hdf)
 -o OUTPUT_DIR, --output OUTPUT_DIR
                         [default: cluster_results] the output directory for the result file. If not exit, the program will create this directory.
 -s SMOOTH, --smooth SMOOTH
                         [default: sqmap] smooth method to be used in Twotime correlation.
 -i GPU_ID, --gpu_id GPU_ID
                         [default: -1] choose which GPU to use. if the input is -1, then CPU is used
 -begin_frame BEGIN_FRAME
                         [default: 1] begin_frame specifies which frame to begin with for the correlation. This is useful to get rid of the bad frames in the beginning.
 -end_frame END_FRAME  [default: -1] end_frame specifies the last frame used for the correlation. This is useful to get rid of the bad frames in the end. If -1 is used, end_frame will be set to
                         the number of frames, i.e. the last frame
 -stride_frame STRIDE_FRAME
                         [default: 1] stride_frame defines the stride.
 -avg_frame AVG_FRAME  [default: 1] stride_frame defines the number of frames to be averaged before the correlation.
 -t TYPE, --type TYPE  [default: "Multitau"] Analysis type: ["Multitau", "Twotime", "Both"].
 -dq TYPE, --dq_selection TYPE
                         [default: "all"] dq_selection: a string that select the dq list, eg. '1, 2, 5-7' selects [1,2,5,6,7]. If 'all', all dynamic qindex will be used.
 --verbose, -v         verbose
 --dryrun, -dr         dryrun: only show the argument without execution.
 --overwrite, -ow      whether to overwrite the existing result file.
 -c CONFIG.JSON, --config CONFIG.JSON
                         configuration file to be used. if the same key is passed as an argument, the value in the configure file will be omitted.      
```


* call boost-corr as a Module

```python
import torch

# check version
import boost_corr
print(boost_corr.__version__)
# output is something like 0.1.3.dev0+g5204424.d20220228

# doing multitau correlation
from boost_corr import MultitauCorrelator
frame_num = 1024
det_size = (128, 128)

device = 'cpu'
# device = 'cuda:0'     # if you want to run it with GPU

mc = MultitauCorrelator(frame_num=frame_num, det_size=det_size)
for n in range(frame_num):
    # generate a random 2d array; it must be reshaped to be used in boost-corr
    x = torch.rand(det_size).reshape(1, -1)
    mc.process(x)

mc.post_process()
result = mc.get_results()
```


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage