# Installation

*The following is a brief summary of installation instructions. For more details, please see Caffe's original instructions in the appendix below*.

The installation instructions for Ubuntu 14.04 can be summarized as follows (the instructions for other Linux versions may be similar).  
1. Pre-requisites
 * It is recommended to us a Linux machine (Ubuntu 14.04 or Ubuntu 16.04 for example)
 * [Anaconda Python 2.7](https://www.continuum.io/downloads) is recommended, but other Python packages might also work just fine. Please install Anaconda2 (which is described as Anacoda for Python 2.7 int he download page). We have seen compilation issues if you install Anaconda3. If other packages that you work with (eg. tensorflow or pytorch) require Python 3.x, one can always create conda environments for it in Anaconda2.
 * One or more graphics cards (GPU) supporting NVIDIA CUDA. GTX10xx series cards are great, but GTX9xx series or Titan series cards are fine too.
 
2. Preparation
 * copy Makefile.config.example into Makefile.config
 * In Makefile.config, uncomment the line that says WITH_PYTHON_LAYER
 * Uncomment the line that says USE_CUDNN
 * If more than one GPUs are available, uncommenting USE_NCCL will help us to enable multi gpu training.
 
3. Install all the pre-requisites - (mostly taken from http://caffe.berkeleyvision.org/install_apt.html)
 * Change directory to the folder where caffe source code is placed.
 * *sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler*
 * *sudo apt-get install --no-install-recommends libboost-all-dev*
 * *sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev*
 * *sudo apt-get install libturbojpeg*
 * Install CUDNN. (libcudnn-dev developer deb package can be downloaded from NVIDIA website) and then installed using dpkg -i path-to-deb
 * Install [NCCL](https://github.com/NVIDIA/nccl/releases) if there are more than one CUDA GPUs in the system
 * Install the python packages required. (this portion is not tested and might need tweaking)
  -- For Anaconda Python: <br> *for req in $(cat python/requirements.txt); do conda install $req; done*
  -- For System default Python: <br> *for req in $(cat python/requirements.txt); do pip install $req; done* 
 * There may be other dependencies that are discovered as one goes through with the compilation process. The installation procedure will be similar to above.

4. Compilation
 * *make* (Instead, one can also do "make -j50" to speed up the compilaiton)
 * *make pycaffe* (To compile the python bindings)
 
5. Notes:
 * If you get compilation error related to libturbojpeg, create the missing symbolic link as explained here:<br>
 -- https://github.com/OpenKinect/libfreenect2/issues/36 <br>
 -- sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.0.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so

 * <b>Building on Ubuntu 18.04 may face some road blocks</b> - but they are solvable. The exact solution may vary slightly depending on the packages in your system. The following are some guidelines: <br>
  (1) One issue is related to the opencv python package provided by anacoda. The solution is to remove that opencv python package (if you have it) and then install another one as follows.<br>
 -- conda remove opencv <br>
 -- conda install -c menpo opencv3 <br>
  (2) There may be symbol errors related to protobuf. Uninstalling system probuf library and installing anaconda protobuf package solved the issue in our case. <br>
 -- sudo apt remove libprotobuf-dev <br>
 -- conda install protobuf <br>
 (3) If using CUDA 10, the following error may occur.
cmake error: CUDA_cublas_device_LIBRARY (ADVANCED) 
This is due to an issue with cmake. Using cmake version >= 3.12.2 solves this issue. If anaconda has been installed recently, a recent version of cmake will be there in the anaconda bin directory and you can use it.

6. Appendix: Caffe's original instructions
 * See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.
 * Check the users group in case you need help:
https://groups.google.com/forum/#!forum/caffe-users
