# Installation

*The following is a brief summary of installation instructions. For more details, please see Caffe's original instructions in the appendix below*.

The installation instructions for Ubuntu 14.04 can be summarized as follows (the instructions for other Linux versions may be similar).  
1. Pre-requisites
 * It is recommended to us a Linux machine (Ubuntu 14.04 or Ubuntu 16.04 for example)
 * [Anaconda Python 2.7](https://www.continuum.io/downloads) is recommended, but other Python packages might also work just fine. Please install Anaconda2 (which is described as Anacoda for Python 2.7 int he download page). We have seen compilation issues if you install Anaconda3. If other packages that you work with (eg. tensorflow or pytorch) require Python 3.x, one can always create conda environments for it in Anaconda2.
 * One or more graphics cards (GPU) supporting NVIDIA CUDA. GTX10xx series cards are great, but GTX9xx series or Titan series cards are fine too.
 
2. Preparation
 * copy `Makefile.config.example` into `Makefile.config`
 * In `Makefile.config`, uncomment the line that says `WITH_PYTHON_LAYER`
 * Uncomment the line that says `USE_CUDNN`
 * If more than one GPUs are available, uncommenting `USE_NCCL` will help us to enable multi gpu training.
 
3. Install all the pre-requisites - (mostly taken from http://caffe.berkeleyvision.org/install_apt.html)
 * Change directory to the folder where caffe source code is placed.
   ```bash
   sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
   sudo apt-get install --no-install-recommends libboost-all-dev
   sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
   sudo apt-get install libopenblas-dev
   sudo apt-get install libturbojpeg-dev
   ```
 * Install [CUDNN](https://developer.nvidia.com/cudnn)
 * Install [NCCL](https://developer.nvidia.com/nccl/nccl-download)
 * Install the python packages required. (this portion is not tested and might need tweaking)
   For Anaconda Python:
   ```bash
   for req in $(cat python/requirements.txt); do conda install $req; done
   ```
   For System default Python: 
   ```bash
   for req in $(cat python/requirements.txt); do pip install $req; done
   ```
 * There may be other dependencies that are discovered as one goes through with the compilation process. The installation procedure will be similar to above.

4. Compilation
 * `make` (Instead, one can also do `make -j50` to speed up the compilaiton)
 * `make pycaffe` (To compile the python bindings)
 
5. Notes:
 * If you get compilation error related to libturbojpeg, create the missing symbolic link as explained here:<br>
 -- https://github.com/OpenKinect/libfreenect2/issues/36 <br>
 -- sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.0.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so

6. <b>Building on Ubuntu 18.04</b> - use the following instructions to easily build on Ubuntu 18.04
 * Downloand and install the latest Anaconda3. Do not use Anaconda2 on Ubuntu18.04 as some libraries (especially OpenCV) may produce link errors.<br>
 * Create a python 2.7 environment in Anaconda3 and activate it. Here we are assuming that the name of the environment is caffe - although this is not necessary.
   ```bash
   conda create -n caffe python=2.7 
   conda activate caffe
   ```
 * Now install additional packages.
   ```
   conda install cmake numpy opencv protobuf libprotobuf hdf5 numpy scikit-image
   ```
 * In the Makefile.confg, make the following changes.<br>
   -- Since the default opencv version in Anaconda is now 3.x, uncomment the line OPENCV_VERSION := 3<br>
   -- Add additional include folders to the line INCLUDE_DIRS. We need to add the full paths to the directories envs/caffe/include envs/caffe/include/python2.7 and envs/caffe/lib/python2.7/site-packages/numpy/core/include located inside the Anaconda installation.<br>
   -- Add additional lib folders to the line LIBRARY_DIRS. We need to add the full paths to the directories envs/caffe/lib and envs/caffe/lib/python2.7 located inside the Anaconda installation.<br>
 
 * At import time or runtime, if there is an issue in finding shared objects (.so files), we may need to specify additional paths into LD_LIBRARY_PATH. Note: This is required only if there is an issue in finding a .so file. The exact path that we may need to add may also be different.
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/a0393608/files/apps/anaconda3/envs/caffe/lib
 
 * After these installations, do cmake and make as explained before.<br>

7. <b>Appendix: Caffe's original instructions </b>
 * See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.
 * Check the users group in case you need help:
https://groups.google.com/forum/#!forum/caffe-users
