# PyOCT: Imaging Reconstruction for Spectral-Domain Optical Coherence Tomography
PyOCT is developed to conduct normal spectral-domain optical coherence tomography (SD-OCT) imaging reconstruction with main steps as:
1. Reading Data
2. Background Subtraction 
3. Spectral Resampling 
3. Comutational Aberration Correction (Alpha-correction)
4. Camera Dispersion Correction (Beta-correction with camera calibration factors) 
5. Inverse Fourier Transform 
6. Obtain OCT Image

The algorithms was developed initially in Prof. Steven G. Adie research lab at Cornell University using MATLAB. The reconstruction speed has been improved with matrix-operation. Compared with MATLAB, Python language have a much better performance in loading data from binary files tested only in our lab computer. Currently, PyOCT only supports python 3.0+. 

This is the repository of codes that have been implemented with my own experience in coherent imaging, including optical coherence tomography (OCT) and digital holography microscopy (DHM), data process as well as miscellaneous tools that I used for my daily practice. 

The codes are free to use for your own under the MIT license while if you want to publish any methods or algorithms that will be developed based on my codes, please make an inquiry in advance. 

The example dataset can be download from [OCT Example Dataset](https://doi.org/10.7910/DVN/EV2P3I). 


## Quick start
PyOCT can be install using pip:

    $pip install PyOCT


If you want to run the latest version of the code, you can install from git:

    $python -m pip install -U git+git://github.com/NeversayEverLin/PyOCT.git


After successful installaiton, you can test program under python environment:

    $from PyOCT import Test
    $test = Test.OCTReconTest() 

To run the OCT imaging reconstruction, you can construct class OCTImagingProcessing() from PyOCTRecon module:

    $from PyOCT import PyOCTRecon 
    $OCTImage = PyOCTRecon.OCTImagingProcessing()  

Class OCTImagingProcessing require at least 3 positional arguments. All input parameters are:

* *root_dir*: required, root directory path where OCT raw data located, ENDING WITHOUT /; e.g.,  root_dir = 'D:/cuda_practice/OCT_Recon/OCTdata'.
* _sampleID_: required, sample data file name. ENDING WITHOUT _raw.bin; e.g., sampleID = 'OCT_100mV_2'. 
* _bkgndID_: required, background data file name, ending with _raw.bin; e.g., bkgndID = 'bkgnd_512_0_raw.bin'.
*  *Sample_sub_path*: optional, default as None; sub-directory where OCT raw data located. ENDING WITHOUT /. 
*  *Bkgnd_sub_path*: optional, default as None; sub-directory where OCT bkgnd data located. ENDING WITHOUT /.
*  *saveOption*: optional, bool, default as False. 
*  *saveFolder*: optional, name for folder to save data; default as None, which will save data in root directory.
*  *datatype*: optional,"real" or "complex", tell to save data or show data in complex format or single precison (float32) format. 
*  *verbose*: optional, bool, default as True. If True, the data processing will show processing information during each step. 
*  *frames*: optional, int, number of frames to read and process, defaults as 1.
*  *alpha2*, *alpha3*: optional, parameters for computational dispersion correction. 
*  *depths*: optional, nuumpy.linspace() created array, depths to be processed, default as: np.linspace(1024,2047,1024), indicating procesing 1024th z-pixel to 2048-pixel.
*  *gamma*: optional, power factor to do plotting, default as 0.4.
*  *wavelength*: optional, nominal central wavelength of OCT laser source in unit of nm, default as 1300. 
*  *XYConversion*: optional, 2 elements numpy array, calibration factor for galvo-scanning voltage to scanning field of view in x and y axis at unit of um/V, default as [660,660].
*  *camera_matrix*: optional, camera dispersion correction factor, numpy array as [c0,c1,c2,c3]; default as np.asarray([1.169980E3,1.310930E-1,-3.823410E-6,-7.178150E-10]) for 1300nm system.
*  *start_frame*: which frame to start reading and being processed. default is 1, indicating starting from first frame. 
*  *OptimizingDC*: [Required Further Developement] optional, bool, optimizing dispersion correction to search optimized alpha2 and alpha3. default as False. 

Class OCTImagingProcessing also provides several accesses/members to imaging processing data:

* *self.root_dir*: root directory of data set
* *self.sampleID*: sample ID
* *self.bkgndID*: background ID
* *self.Settings*: parameters of settings of reconstruction 
* *self.OCTData*: single precision OCT intensity data 
* *self.data*: complex OCT reconstruction data, only accessible when datatype is not "real".
* *self.InterferencePattern*: interference fringes of OCT imaging
* *self.DepthProfile*: depth profile (along z-axis) of reconstructed image
* *self.ShowXZ(OCTData)*: member function to show cross-section. 

## License
PyOCT is licensed under the terms of the MIT License (see the file LICENSE).


