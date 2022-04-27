# Optical imaging reconstruction for both spectral-domain OCT and off-axis digital holography microscopy
PyOCT is developed to conduct normal spectral-domain optical coherence tomography (SD-OCT) imaging reconstruction with main steps as:
1. Reading Data
2. Background Subtraction 
3. Spectral Resampling 
3. Comutational Aberration Correction (Alpha-correction)
4. Camera Dispersion Correction (Beta-correction with camera calibration factors) 
5. Inverse Fourier Transform 
6. Obtain OCT Image

For off-axis digital holography microscopy (DHM) reconstruction, importing HoloLib from PyOCT and using class of QPImage(). 

PyOCT only supports python 3.0+. 

## Quick start
PyOCT can be install using pip:

    $pip install PyOCT


If you want to run the latest version of the code, you can install from git:

    $python -m pip install -U git+git://github.com/NeversayEverLin/PyOCT.git

For sample dataset of OCT, please download from: https://www.dropbox.com/sh/qsoco6detbxmtp3/AABGNsepMjcAvAr1niRBz7_Qa?dl=0

After successful installaiton, you can test program under python environment:

    $from PyOCT import VolumeReconstruction
    $VolumeReconstruction.Run_test() 

To run the OCT imaging reconstruction, you can construct class OCTImagingProcessing() from PyOCTRecon module:

    $from PyOCT import PyOCTRecon 
    $OCTImage = PyOCTRecon.OCTImagingProcessing()  

Class OCTImagingProcessing require at least 3 positional arguments. All input parameters are:

* *root_dir*: required, root directory path where OCT raw data located, ENDING WITHOUT /; e.g.,  root_dir = 'D:/cuda_practice/OCT_Recon/OCTdata'.
* *SampleData*: optional, sample data, most of time won't need.
* *Settings*: optional, Settings, most time won't need.
* *sampleID*: required, sample data file name. ENDING WITHOUT _raw.bin; e.g., sampleID = 'OCT_100mV_2'. 
* *bkgndID*: required, background data file name, ending with _raw.bin; e.g., bkgndID = 'bkgnd_512_0_raw.bin'.
* *Sample_sub_path*: optional, default as None; sub-directory where OCT raw data located. ENDING WITHOUT /. 
* *Bkgnd_sub_path*: optional, default as None; sub-directory where OCT bkgnd data located. ENDING WITHOUT /.
* *saveOption*: optional, bool, default as False. 
* *saveFolder*: optional, name for folder to save data; default as None, which will save data in root directory.
* *RorC*: optional,"real" or "complex", tell to save data or show data in complex format or single precison (float32) format. 
* *verbose*: optional, bool, default as True. If True, the data processing will show processing information during each step. 
* *frames*: optional, int, number of frames to read and process, defaults as 1.
* *alpha2*, *alpha3*: optional, parameters for computational dispersion correction. 
* *depths*: optional, nuumpy.linspace() created array, depths to be processed, default as: np.linspace(1024,2047,1024), indicating procesing 1024th z-pixel to 2048-pixel.
* *gamma*: optional, power factor to do plotting, default as 0.4.
* *wavelength*: optional, nominal central wavelength of OCT laser source in unit of nm, default as 1300. 
* *XYConversion*: optional, 2 elements numpy array, calibration factor for galvo-scanning voltage to scanning field of view in x and y axis at unit of um/V, default as [660,660].
* *camera_matrix*: optional, camera dispersion correction factor, numpy array as [c0,c1,c2,c3]; default as np.asarray([1.169980E3,1.310930E-1,-3.823410E-6,-7.178150E-10]) for 1300nm system.
* *start_frame*: which frame to start reading and being processed. default is 1, indicating starting from first frame. 
* *OptimizingDC*: [Required Further Developement] optional, bool, optimizing dispersion correction to search optimized alpha2 and alpha3. default as False. 
* *singlePrecision*: only workable when RorC = 'real', Default as True, data will be converted into numpy.float32 single precision data type.
* *ReconstructionMethods*: 'cao' or 'nocao', default as 'NoCAO'. using bkgnd data as real time background estimation from signal dataset ("CAO") or directly from bkgnd file ("NoCAO")

Another class in PyOCT is *Batch_OCTProcessing()*, which using data processing provided by class OCTImagingProcessing() with additional inputs as:
* *Ascans*: number of Ascans.
* *Frames*: number of frames.
* *ChunkSize*: number of frames at each sub-segmentation dataset.

Batch_OCTProcessing() should be used when dataset is too large to be directly processed by whole volume which might exhaust your RAM/CPU. It will automatically segmented dataset into sub-segmentation dataset to be processed. The processed volume data and settings could be accessed by Batch_OCTProcessing.data or Batch_OCTProcessing.OCTData and Batch_OCTProcessing.Settings. You can still access to basic OCTImagingProcessing methods by accessing to methods like Batch_OCTProcessing.OCTRe.ShowXZ().

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

DHM image reconstruction:
This class implements various tasks for quantitative phase imaging, including phase unwrapping, background correction,numerical focusing, and data export.
Parameters:
* *data*: 2d ndarray (float or complex) or list, The experimental data (see which_data). If data is a file .mat or .h5 or .hdf5, it will automatically load data where the keyword is given by data_key or automatically search for "IMG" or "data". 
* *data_key*: the key for accessing data array if "data" is defined as a file format; Default is None, then it will iterates automaticaly through the data file and get the first keys of either "data" or "IMG" 
* *ref_data*: reference image. could be (X,Y) or (T,X,Y) 
* *meta_data*: dict, Meta data associated with the input data.
* *holo_kw*: dict,Special keyword arguments for phase retrieval from hologram data.default: {"batchSize":50,"cr":0.5,"trans":False onlyCPU":False, "verbose":verbose,"zero_pad":True,subtract_mean":True, "returnContrastMat":True} 
    batchSize: the batch size to divide raw data into small batches in case overflowing memory. If overmemory happens, reduce the batchSize.
    cr: proportion of image to be counted when calculating interference contrast. 
    trans: transpose the raw data. usually to make it identical dimension to MATLAB
    onlyCPU: only using CPU. If False, it will choose GPU computation resource. 
    verbose: show intermediate results. better to set as False when dealing with large amount of data
    zero_pad:True,do zero padding 
    subtract_mean:True, subtract mean of rawdata before recontruction.
    returnContrastMat:True, get the contrast results also. 
    bg_kw: dict, keyword for estimatign background title. here right now, default: {"fit_offset":"mean", "fit_profile":"tilt","border_px":6}
    computeBg: bool, default as True. whether or not to compute background tilt, using parameters from bg_kw. 
* *proc_phase*: bool, Process the phase data. This includes phase unwrapping. using :func:`skimage.restoration.unwrap_phase` and correcting for 2PI phase offsets (The offset is estimated from a 1px-wide border around the image  
* *slices*: int, default -1, indicating it will process all the frames. Otherwise, it will only process 0:slices frames. 
        
* *Members and attributes*:
        .data: recontructed complex data. This is actually the phase is not processed. 
        .field: recontructed complex data where the phase is unwrapped and background is corrected (if computeBg=True)
        .amp: get amplitude 
        .pha: get processed phase data. 
        .bg: get computed and fitted background fitted data. 
        .ref: processed reference image. 
        .info: get meta data. 
        .save: save data into file (.mat, .hdf5 or .h5) 
        .contrast_mat: contrast map versus z axis. 
        .out_int: intensity map versus z axis. 
        .zpos: z-axis positions for each frame, if available. 

        typical use:
        qpi = QPImage(file_name) 
        qpi.save(...)  

for a more general use, definition a function for being used:

```
def holoReconstruction(rootPath,dataName,refName='none',verbose=True):
    if dataName.endswith(".mat"):
        data_sName = dataName[:-4]
    elif dataName.endswith(".h5py"):
        data_sName = dataName[:-5]
    else:
        data_sName = dataName 

    savePath = os.path.join(rootPath,data_sName) 
    if not os.path.isdir(savePath):
        os.mkdir(savePath)     
    
    if not refName == 'none':
        refFile = hp.File(os.path.join(rootPath,refName),"r") 
        refDataRaw = np.asarray(refFile["data"]["IMG"])
        takeRef = True 
    else:
        takeRef = False
    dataFile = hp.File(os.path.join(rootPath,dataName),"r") 
    dataRaw = np.asarray(dataFile["data"]["IMG"]) 
    if dataRaw.ndim == 4:
        dataRaw = np.squeeze(np.mean(dataRaw,axis=0)) 
    if "zpos" in dataFile["data"].keys():
        zPos = np.squeeze(dataFile["data"]["zpos"][()]) #using zz or zpos
    else:
        zPos = np.arange(0,np.shape(dataRaw)[0],1)
    zPos = zPos/1.33
    if takeRef:
        qpimage = pl.QPImage(data=dataRaw,ref_data=refDataRaw,holo_kw={"cr":0.5,"onlyCPU":False,"subtract_mean":True,"batchSize":5})
    else:
        qpimage = pl.QPImage(data=dataRaw,holo_kw={"cr":0.5,"onlyCPU":False,"subtract_mean":True,"batchSize":5}) 
    
    if verbose:
        misc.show3DStack(qpimage.pha,cmap="rainbow",figTitle="volPhase")
        misc.show3DStack((qpimage.amp)**2,figTitle="int") 
    qpimage.save(savePath,fileName=data_sName+"_Recon.mat",format=".h5")  
    return 1 
```        
Example dataset could be download under the request to email address: linyuechuan1989@gmail.com 
## License
PyOCT is licensed under the terms of the MIT License (see the file LICENSE).# PyOCT
