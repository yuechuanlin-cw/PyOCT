from __future__ import division, print_function
import os
from h5py._hl.files import File 
import numpy as np 
import xml.etree.ElementTree as ET
import time
from numpy.ma.core import divide
from scipy.linalg import dft
import numpy.matlib 
import matplotlib.pyplot as plt 
import matplotlib 
from matplotlib.widgets import Slider
import re 
import h5py
from scipy.linalg.misc import norm 
from scipy.signal import fftconvolve
import matplotlib.patches as patches
import cv2 
import pickle
from scipy import ndimage
import scipy.stats
import matplotlib.colors
from matplotlib import cm 
import math 
from scipy.optimize import curve_fit 
from lmfit import Model 
import scipy.io 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy 
from cellpose import models
import skimage.filters
from PIL import Image
"""
1) imshow() will tranpose the array. That's the x-axis in imshow() corresponds to y-axis in numpy-array. 
fig22.align_labels()

2) load complex mat data 
matFile = h5py.File(os.path.join(currPath,"matlab_processedRes.mat"),'r')
matData = matFile['out'][()].value.view(np.complex)
matData = matData[0:10,:,:]
"""


def find_all_dataset(root_dir,saveFolder, saveOption='in'):
    """
    Looking for all datasets under root_dir and create a saveFolder under root_dir for data save. 
    : root_dir: root directory of all data files
    : saveFolder: name of folder where the data should be saved 
    Return:
    : NumOfFile: total of raw data files 
    : RawDataFileID: sorted raw data file ID
    : SettingsFileID: sorted settings file ID of corresponding raw data file 
    : BkgndFileID: background data file 
    : save_path: the path to save data 
    : saveOption: 'in' or 'out', indicating save the processed files into current root directory with folder name as saveFolder ('in')
    :               or save the processed files into an independent directory with saveFolder as a full directory path. 
    """
    if saveOption.lower() == 'in':
        save_path = os.path.join(root_dir,saveFolder) 
    elif saveOption.lower() == 'out':
        save_path = saveFolder 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    subfolders = os.listdir(root_dir)
    SettingsFileID = [] 
    RawDataFileID = []
    BkgndFileID = []
    for item in subfolders:
        if item.endswith('_settings.xml'):
            SettingsFileID.append(item) 
        if item.endswith('_raw.bin'):
            if 'bkgnd' not in item:
                RawDataFileID.append(item)
            else:
                BkgndFileID.append(item) 
    # sort file name by numerical order
    pattern = re.compile(r'_\d+_') 
    pattern2 = re.compile(r'\d+')
    RawDataFileID = sorted(RawDataFileID, key=lambda x:int(pattern2.findall(pattern.findall(x)[0])[0]))
    SettingsFileID = sorted(SettingsFileID , key=lambda x:int(pattern2.findall(pattern.findall(x)[0])[0]))
    NumOfFile = len(RawDataFileID) 
    return NumOfFile, RawDataFileID, BkgndFileID, SettingsFileID, save_path

def ListAllDataFile(data_path,endsWith,startsWith=None,searchPattern=r"_\d+_",searchPattern2=r"\d+",returnNum=False):
    """
    Serach for all data file under the condition of endsWith and return with a sorted results. 
    The data file name can only have one number indicating the sequential order of file name, otherwise it might not right. 
    searchPattern and searchPattern2 are intial and refined rearch target. 
    """
    if data_path.endswith("/") or data_path.endswith("\\"):
        data_path = data_path[:-1]
    dataID = []
    for dfile in os.listdir(data_path):
        if startsWith == None:
            if dfile.endswith(endsWith):
                dataID.append(dfile) 
        else:
            if dfile.startswith(startsWith) and dfile.endswith(endsWith):
                dataID.append(dfile) 
    searchPattern = re.compile(searchPattern)
    searchPattern2 = re.compile(searchPattern2)
    if searchPattern2 == None:
        dataID = sorted(dataID, key=lambda x:int(searchPattern.findall(x)[0])) 
    else:
        dataID = sorted(dataID, key=lambda x:int(searchPattern2.findall(searchPattern.findall(x)[0])[0])) 
    sortNum = []
    for x in dataID:        
        sortNum.append(int(searchPattern2.findall(searchPattern.findall(x)[0])[0]))
    if returnNum:
        return dataID, np.asarray(sortNum)
    else:
        return dataID 

def SaveData(save_path,FileName,inData,datatype='data',varName = 'OCTData'):
    """
    Save data in the format of .hdf5
    : save_path: directory path where the data will be saved. 
    : FileName: name of file name. Therefore, the file will be FileName.hdf5 
    : inData: input data. This should be an ndarray or Settings file. 
    
    """
    if save_path.endswith("/") or save_path.endswith("\\"):
        save_path = save_path[:-1]

    if datatype.lower() == 'data':
        if np.iscomplexobj(inData):
            DataFileSave = h5py.File(save_path+'/'+FileName+'.hdf5','w')
            DataFileSave.create_dataset(varName+'_real',shape=np.shape(inData),data=np.real(inData),compression="gzip")
            DataFileSave.create_dataset(varName+'_imag',shape=np.shape(inData),data=np.imag(inData),compression="gzip")
            DataFileSave.close()
        else:
            DataFileSave = h5py.File(save_path+'/'+FileName+'.hdf5','w')
            DataFileSave.create_dataset(varName,shape=np.shape(inData),data=inData,compression="gzip")
            DataFileSave.close()
    elif datatype.lower() == 'settings':
        SettingsFile = h5py.File(save_path+'/'+FileName+'.hdf5','w')
        for k, v in inData.items():
            SettingsFile.create_dataset(k,data=v)
        SettingsFile.close()         
    else:
        raise ValueError("Wrong data type!")    
def LoadSettings(path,FileName):
    """
    Loading Settings file. six
    path should NOT end with "/" or "\\".
    """
    Settings = dict.fromkeys([], []) 
    fid = h5py.File(path+'/'+FileName,'r')
    for key in fid.keys():
        Settings[key] = fid[key][()]
    return Settings 

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    """Calculating correlation coefficient between two input 2D array
    with same definition to corr2() in MATLAB
    """
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return np.abs(r)


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def Max2d(inData,nanVersion=False):
    #return maximum and also index (y,x) of 2D array 
    if nanVersion:
        return np.nanmax(inData), np.asarray(np.unravel_index(np.nanargmax(inData),inData.shape),dtype=int)
    else:
        return np.amax(inData), np.asarray(np.unravel_index(inData.argmax(),inData.shape),dtype=int) 

def patternMatch(template,rootImage,cropIndex = None, showFit = False):
    """Compare template image to rootImage and find the translation index required 
    for makeing template image matched with rootImage. That's by moving (transX,transY) to ensure 
    template image as much similar as to rootImage. Using normxcorr2() method which requires a small image region cropped from template.
    Therefore, cropIndex means the subimage of template used to deconvlve with rootImage. If cropIndex is None, then directly using template image as subimage.
    : template: to be compared, 2d numpy array as real. if cropIndex is None, both dimensions of template image must be smaller than rootImage. Using cropIndex must result in a smaller dimension of subimage compared to rootImage. 
    : rootImage: basic image, 2d nump.array as real. It is best template and rootImage has the same 
    : cropIndex: None as default, or (4,) list/array with [xmin,xmax,ymin,ymax]. 
    : showFit: present fit results 
    """
    if cropIndex == None:
        CropImage = template 
        centerofCropInTemplate = (0,0)
    else:
        cropIndex = np.asarray(cropIndex) 
        CropImage = template[cropIndex[0]:cropIndex[1],cropIndex[2]:cropIndex[3]]
        centerofCropInTemplate = (int(np.ceil((cropIndex[1]+cropIndex[0])/2)), int(np.ceil((cropIndex[2]+cropIndex[3])/2)))
    cTmp = normxcorr2(CropImage,rootImage,mode='same') 
    cMax, cPos = Max2d(cTmp) 
    transX, transY = (cPos[0] - centerofCropInTemplate[0],  cPos[1] - centerofCropInTemplate[1])
    if showFit:
        figC = plt.figure(figsize=(14,4))
        ax00 = plt.subplot2grid((1,3),(0,0),rowspan=1,colspan=1) 
        ax00.set_title("Matching Corr Map")
        ax01 = plt.subplot2grid((1,3),(0,1),rowspan=1,colspan=1) 
        ax01.set_title("Root image")
        ax02 = plt.subplot2grid((1,3),(0,2),rowspan=1,colspan=1) 
        ax02.set_title("Template Image") 
        imax0 = ax00.imshow(cTmp,aspect='equal') 
        ax01.imshow(rootImage,aspect='equal',cmap='gray')
        ax02.imshow(template,aspect='equal',cmap='gray') 
        figC.colorbar(imax0,ax=ax00,orientation='vertical',fraction=0.05,aspect=50)

        figT = plt.figure(figsize=(5,5))
        axT = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1) 
        axT2 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1) 
        axT.imshow(rootImage,cmap='gray',aspect='equal',interpolation = 'none')
        rect = patches.Rectangle((cPos[1]-np.shape(CropImage)[1]/2,cPos[0]-np.shape(CropImage)[0]/2), np.shape(CropImage)[1], np.shape(CropImage)[0], fill=False,linestyle='--',linewidth=2,edgecolor='tab:red')
        axT.add_patch(rect)
        axT2.imshow(CropImage,cmap='gray',aspect='equal',interpolation='none')

    return cTmp, cMax,  cPos, transX, transY


def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
    """

    # check the input
    if not isinstance( img_in, numpy.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (numpy.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = numpy.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = numpy.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum

def FindVrange(enFace,VmaxBound=[0.999,1.0],VminBound=[0.01,0.05]):
    tmp = np.sort(enFace.flatten())
    sizeTmp = np.size(tmp) 
    vmax = np.median(tmp[int(sizeTmp*VmaxBound[0]):int(sizeTmp*VmaxBound[1])]) 
    vmin = np.median(tmp[int(sizeTmp*VminBound[0]):int(sizeTmp*VminBound[1])])
    OCTnorm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax) 
    return [vmax, vmin,OCTnorm]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
     new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
           'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
     return new_cmap 

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 

def WriteIntoGif(path,fps,endsWith = '.png',saveFileName=None,inFormat="mp4",count=-1):
    import imageio 
    from progress.bar import Bar
    if path.endswith("/") or path.endswith("\\"):
        path = path[:-1]
    pngFiles = []
    pngIndx = []
    for file in os.listdir(path):
        if file.endswith(endsWith):
            pngFiles.append(file) 
            tmp = re.findall(r'\d+',file)
            pngIndx.append(int(tmp[0]))
    pngFiles = np.asarray(pngFiles)
    pngFiles = pngFiles[np.argsort(pngIndx)]
    bar2 = Bar(' Writing into GIF', max=len(pngFiles))
    images = []
    if not (count == -1):
        pngFiles = pngFiles[0:count]

    if inFormat == "gif":
        for ii in range(len(pngFiles)):
            bar2.next()
            tmpImg = imageio.imread(path+'/'+pngFiles[ii])
            if ii == 0:
                vSize = tmpImg.shape 
            tmpImg = Image.fromarray(tmpImg).resize((vSize[0],vSize[1]))
            images.append(np.asarray(tmpImg))
        if saveFileName:
            savename = saveFileName+".gif"
        else:
            savename = "animation.gif"
        imageio.mimsave(path+"/"+savename, images,fps=fps)
    elif inFormat == "mp4":        
        kwargs = {"fps":fps,"quality":10,"macro_block_size":2}
        writer = imageio.get_writer(path+"/"+"animation.mp4",format="FFMPEG",**kwargs) #pixelformat='vaapi_vld', 
        for ii in range(len(pngFiles)):
            bar2.next()
            tmpImg = imageio.imread(path+'/'+pngFiles[ii]) 
            if ii == 0:
                vSize = np.asarray(tmpImg.shape) 
                vSize[0] = vSize[0] - np.mod(vSize[0],2) 
                vSize[1] = vSize[1] - np.mod(vSize[1],2) 
            tmpImg =  np.asarray(Image.fromarray(tmpImg).resize((vSize[1],vSize[0])))
            writer.append_data(tmpImg)
        #for filename in pngFiles:
        #    bar2.next() 
        #    writer.append_data(imageio.imread(path+'/'+filename))
        writer.close()


def patternMatch_fft(plane_xy_shift,plane_xy,showFit = False):
    """
    pattern match in frequency domain by taking Fourier transform of input data
    here plane_xy_shift and plane_xy must has the same dimension. 
    """ 
    sx, sy = np.shape(plane_xy)
    fft_xy = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(plane_xy),s=[sx,sy]))
    fft_xy_shift = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(plane_xy_shift),s=[sx,sy]))
    cross_xy = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft_xy * np.conjugate(fft_xy_shift)))))
    max_corr_r, posxy = Max2d(cross_xy)
    shiftx, shifty = np.asarray(posxy)-[int(sx/2),int(sy/2)]

    if showFit:
        vmin = 2*np.amin(np.abs(plane_xy)**0.4)
        vmax = 0.7*np.amax(np.abs(plane_xy)**0.4)
        OCTnorm = matplotlib.colors.Normalize(vmin = vmin,vmax = vmax)
        fig = plt.figure(figsize=(5,3))
        ax00 = plt.subplot2grid((1,4),(0,0),rowspan=1,colspan=1) #xz
        ax01 = plt.subplot2grid((1,4),(0,1),rowspan=1,colspan=1)
        ax02 = plt.subplot2grid((1,4),(0,2),rowspan=1,colspan=1)
        ax03 = plt.subplot2grid((1,4),(0,3),rowspan=1,colspan=1)
        ax00.set_title("plane xy")
        ax01.set_title("plane shift")
        ax02.set_title("Fit res")
        ax03.set_title("Re shift plane")
        ax00.imshow(np.abs(plane_xy)**0.4,cmap='gray')
        ax01.imshow(np.abs(plane_xy_shift)**0.4,cmap='gray') 
        ax02.imshow(np.abs(cross_xy))
        ax03.imshow(np.roll(np.abs(plane_xy_shift)**0.4,(shiftx,shifty),axis=(0,1)),cmap='gray')

    return [max_corr_r,shiftx,shifty]

def patternMatch_fft_scan(testVol,refPlane,cAxis,numAve=2,showFit=False):
    """
    Do a pattern match by using fft method and also scan over a range along cAxis. 
    refPlane: 2d dim array (alread median filtered)
    testVol: 3d dim array
    cAxis: along which axis of testVol to do plane-b-plane compare
    numAve: number of axis to averged over cAxis to do compare
    """
    testVol = (testVol - np.mean(testVol))/np.std(testVol) 
    refPlane = (refPlane - np.mean(refPlane))/np.std(refPlane) #normalize to reduce the effects of variant brightness on motion correction 
    cAxisLen = np.shape(testVol)[cAxis] 
    CorrR = 0 
    shiftx = 0
    shifty = 0
    for i in np.arange(numAve,cAxisLen-numAve,step=1,dtype=int):
        if cAxis == 0:
            tmpPlane = np.amax(testVol[i-numAve:i+numAve,:,:],axis=0)
        elif cAxis == 1:
            tmpPlane = np.amax(testVol[:,i-numAve:i+numAve,:],axis=1)
        elif cAxis == 2:
            tmpPlane = np.amax(testVol[:,:,i-numAve:i+numAve],axis=2)
        tmpPlane = ndimage.median_filter(tmpPlane,size=(3,3))
        tmpCorrR,tmpshiftx,tmpshifty = patternMatch_fft(tmpPlane,refPlane,showFit=showFit)
        if np.abs(tmpCorrR) > np.abs(CorrR):
            CorrR = tmpCorrR 
            shiftx = tmpshiftx 
            shifty = tmpshifty 

    return [shiftx,shifty]

def patternMatch_fft_3d(testVol_raw,refVol_raw,testSurfPos,refSurfPos):
    """
    Pattern match for motion correction in 3d. 
        src_freq = np.fft.fftn(src_image_cpx)
        target_freq = np.fft.fftn(target_image_cpx)
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = np.fft.ifftn(image_product)
        #cross_correlation = ifftn(image_product) # TODO CHECK why this line is different
        new_cross_corr = np.abs(cross_correlation)
        CCmax = cross_correlation.max()
        maxima = np.unravel_index(np.argmax(new_cross_corr), new_cross_corr.shape)
        midpoints = np.array([np.fix(axis_size//2) for axis_size in shape])
        shifts = np.array(maxima, dtype=np.float32)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    """
    zshift = int(refSurfPos-testSurfPos) 
    testVol_raw = np.roll(testVol_raw,zshift,axis=0)
    testVol = testVol_raw[refSurfPos+20:refSurfPos+450,:,:]
    refVol = refVol_raw[refSurfPos+20:refSurfPos+450,:,:]
    testVol = (testVol - np.mean(testVol))/np.std(testVol) 
    refVol = (refVol - np.mean(refVol))/np.std(refVol) 
    test_freq = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(testVol))) 
    ref_freq = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(refVol))) 
    shape = test_freq.shape
    cross_correlation = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ref_freq * test_freq.conj()))))
    CCmax = cross_correlation.max() 
    maxima = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size//2) for axis_size in shape]) 
    shifts = np.array(maxima-midpoints, dtype=int)
    #shifts[0] = shifts[0]-int(testSurfPos-refSurfPos) 
    testVol_raw = np.roll(testVol_raw,shifts,axis=(0,1,2))    
    if shifts[0] > 30:
        print("Warning: z axis shift larger than 30 pixels!")
    if shifts[1] > 30:
        print("Warning: x axis shift larger than 30 pixels!")
    if shifts[2] > 30:
        print("Warining: y axis shift larger than 30 pixels!") 
    return shifts, testVol_raw 

def ListTXT(FileName,mode='w',data=None):
    if mode.lower() == 'w' or mode.lower() == "write":
        if data == None:
            raise ValueError("Please input DATA to be saved!")
        else:
            with open(FileName,"wb") as fp2:
                pickle.dump(data,fp2)
            output = 1
    elif mode.lower() == "r" or mode.lower() == "read":
        with open(FileName, "rb") as fp:   # Unpickling
            output = pickle.load(fp)
    else:
        raise ValueError("mode should only be either write or read !")
    
    return output


def mean_confidence_interval(data, confidence=0.95,inputType="n"):
    if inputType == "d":
        data = data.to_numpy()
    a = data.flatten() #1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val,reverse=False):
    if reverse:
        self.cmap_name = cmap_name+"_r"
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    print("color clim is {}".format(self.scalarMap.get_clim()))
  def get_rgb(self, val,alpha=1.0):
    return self.scalarMap.to_rgba(val,alpha=alpha) #,bytes=True)



def get_prime_factors(number):
    prime_factors = []

    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(int(i))
            number = number / i
    if number > 2:
        prime_factors.append(int(number))
    return prime_factors



def colorV(cat="raw"):
    #blue red
    if cat =="raw":
        return ["#019BD8", "#D81C28","#FF7F0E","#2CA02C","#e377c2"]
    elif cat == "three": #red, blue, gray 
        return ["#35739C","#DE3122","#BDBEC0"]
    elif cat == "three2": #blue, green, gray 
        return ["#C1C2E2","#8CB78D","#525252"]
    elif cat == "six":
        return ["#DF1D27","#F37E1F","#367DB7","#F6EB36","#4AAD4A","#A55528"]
    elif cat == "six2":
        return ["#E63946","#A8DADC","#F1FAEE","#457B9D","#8D99AE","#1D3557"]
    elif cat == "viz":
        return ["#fc4e51","#1dabe6","#1c366a","#c3ced0","#e43034","#af060f"]
    else:
        return ["#019BD8", "#D81C28","#FF7F0E","#2CA02C","#e377c2"]


def create_fig(figsize=[3,2],nrows=1,ncols=1,left=0.12, bottom=0.12, right=0.95, top=0.95,sharex=False,sharey=False,wspace=0.2,hspace=0.2):
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=sharex,sharey=sharey) 
    fig.set_size_inches(figsize[0],figsize[1])
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top,wspace=wspace,hspace=hspace)
    #fontsize=6,borderpad=0.2,labelspacing=0.2,handlelength=1.0,handletextpad=0.4,borderaxespad=0.2,columnspacing=1.0)
    return fig, ax 

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def Quartiles_Whiskers(vals):
    quartile1, medians, quartile3 = np.percentile(vals,[25,50,75]) 
    medians = np.mean(vals) 
    whiskers = adjacent_values(vals,quartile1,quartile3) 
    return quartile1,medians,quartile3,whiskers

def Gauss(x, amp, cen, wid, bis):
    #return bis  + amp/(np.sqrt(2*np.pi)) * np.exp(-(x-cen)**2/(2*wid**2))
    return bis + amp*np.exp(-2*(x-cen)**2/(wid**2))

def gauss_fwhm(x,y):
    cont_bi = np.sqrt(2*np.log(2))
    xsize = np.size(x) 
    amp = np.amax(y) 
    cen = x[np.argmax(y)]
    
   # halVal = (np.amax(y) - np.amin(y))/2 
   # indx = x[y>halVal] 
   # wid = (x[np.argmax(indx)] - x[np.argmin(indx)])/cont_bi
    mean = sum(x * y) / sum(y)
    wid = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    #print("Estimate FWHM is {}".format(wid))
    popt, pcov = curve_fit(Gauss,x,y,p0=[amp,cen,wid,np.amin(y)])
    fwhm = cont_bi * popt[2]
    #print(popt) 
    fitx = np.linspace(np.amin(x),np.amax(x),num=np.size(x)*2)
    fity =Gauss(fitx,*popt)
    return fwhm, fitx,fity, popt  #, x[np.argmin(indx)],x[np.argmax(indx)]



def fwhm(x,y):
    """
    Calculating of FWHM based on directly searching for 0.5Max. 
    FWHM of the waveform y(x) and its polarity. 
    Adapt from Patrick Egan, Rev 1.2 April 2006. 
    Which was originally developed in Matlab. 
    """
    x = np.squeeze(x)
    y = np.squeeze(y)
    y = y/np.amax(y) 
    N = np.size(y) 
    lev50 = 0.5 
    if y[0] < lev50:
        centerIndex = np.argmax(y) 
       # Pol = 1 
    else:
        centerIndex = np.argmin(y) 
       # Pol = -1 
    indx1 = 1 
    while np.sign(y[indx1]-lev50) == np.sign(y[indx1-1]-lev50):
        indx1 = indx1 + 1 
    
    interp = (lev50 - y[indx1-1])/(y[indx1]-y[indx1-1])
    tlead = x[indx1-1] + interp * (x[indx1]-x[indx1-1]) 
    indx2 = centerIndex + 1 
    while (np.sign(y[indx2]-lev50) == np.sign(y[indx2-1]-lev50)) and (indx2 < N-1):
        indx2 = indx2 + 1 
    if indx2 != N-1:
        #Ptype = 1 
        interp = (lev50-y[indx2-1]) / (y[indx2]-y[indx2-1]) 
        ttrail = x[indx2-1] + interp * (x[indx2]-x[indx2-1]) 
        width = ttrail - tlead 
    else:
        #Ptype = 2
        ttrail = np.nan 
        width = np.nan 


    return width 

def gauss_fwhm2(x,y,showDetail=False):
    cont_bi = np.sqrt(2*np.log(2))
    xsize = np.size(x) 
    amp = np.amax(y) 
    cen = x[np.argmax(y)]
    
    #mean = sum(x * y) / sum(y)
    wid = fwhm(x,y) #np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    gmodel = Model(Gauss) 
    if showDetail:
        print('parameter names: {}'.format(gmodel.param_names))
        print('independent variables: {}'.format(gmodel.independent_vars))
    params = gmodel.make_params(amp=amp,cen=cen,wid=wid,bis=np.amin(y)) 
    results = gmodel.fit(y,params,x=x)
    if showDetail:
        print(results.fit_report()) 

    
    fitx = np.linspace(np.amin(x)*1.1,np.amax(x)*1.1,num=np.size(x)*2) 
    fity = gmodel.eval(params=results.params,x=fitx) 
    dely = results.eval_uncertainty(sigma=3,x=fitx) #3sigma uncertainity 
    
    fwhm0 = results.params["wid"]*cont_bi
    return fwhm0,fitx,fity,results,dely  

def show3DStack(image_3d, axis = 0, cmap = "gray",  extent = (0, 1, 0, 1),trans = False, addPatch=False,NewPatch=None,figTitle="Img",excludeNaN=True,**kwargs):
    if axis == 0:
        image  = lambda index: image_3d[index, :, :]
    elif axis == 1:
        image  = lambda index: image_3d[:, index, :]
    else:
        image  = lambda index: image_3d[:, :, index]
    
    if excludeNaN:
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad("black",1.0)
    current_idx= 0
    figT, ax      = plt.subplots(1, 1, figsize=(6.5, 5))
    figT.suptitle(figTitle)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    if "clim" in kwargs.keys():
        clim = kwargs["clim"]
    #(np.nanmin(np.abs(image_3d)), np.nanmax(np.abs(image_3d)))
        if trans:
            fig = ax.imshow(np.transpose(image(current_idx)), cmap = cmap,  clim = clim, extent = extent)
        else:
            fig = ax.imshow(image(current_idx), cmap = cmap,  clim = clim, extent = extent)
    else:
        if trans:
            fig = ax.imshow(np.transpose(image(current_idx)), cmap = cmap,  clim = (np.nanmin(image(current_idx)), np.nanmax(image(current_idx))), extent = extent)
        else:
            fig = ax.imshow(image(current_idx), cmap = cmap,  clim = (np.nanmin(image(current_idx)), np.nanmax(image(current_idx))), extent = extent)
    ax.set_title("layer: " + str(current_idx))
    plt.colorbar(fig, ax=ax)
    plt.axis('off')
    ax_slider  = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_obj = Slider(ax_slider, "layer", 0, image_3d.shape[axis]-1, valinit=current_idx, valfmt='%d',color="green")
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax.set_title("layer: " + str(index))
        if trans:
            fig.set_data(np.transpose(image(index)))            
        else:
            fig.set_data(image(index))
        if "clim" not in kwargs.keys():
            fig.set_clim((np.nanmin(image(index)), np.nanmax(image(index))))
        if addPatch:
            ax.add_patch(NewPatch)
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < image_3d.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    #plt.show()
    return ax 



def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = scipy.stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return result

def ReadMat(fileName,keyWord=None):
    output = File(fileName,"r")
    if keyWord == None:
        res = output["data"] 
        output.close()
    else:
        res = np.asarray(output["data"][keyWord])
        print("Member in dataset include: ")
        print(output["data"].keys()) 
        output.close()
    return res 

def makeGaussian2D(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
        w(1/e2 radius) = fwhm/(2 sqrt(log2))
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def LoadMat(filePath,verbose=True,**kwargs):
    """
    Not working as expected 
    """
    try:
        dataFile = h5py.File(filePath,"r")
        if verbose:
            print("keys are: {}".format(dataFile["data"].keys())) 
        if "keys" in kwargs:
            return dataFile["data"][eval(keys)]
        else:
            return dataFile
    except:
        data = scipy.io.loadmat(filePath)
        if verbose:
            print(data.keys())
        if "keys" in kwargs:
            return data["data"][eval(keys)] 
        else:
            return data

def add_colorbar(ax,im,position="right",size="4%",pad=0.05,label="",label_fontsize=8):
    divider = make_axes_locatable(ax) 
    cax = divider.append_axes(position, size=size, pad=pad)
    cb=plt.colorbar(im,cax=cax)
    cb.ax.tick_params(labelsize=label_fontsize) 
    if len(label) != 0:
        cb.set_label(label,fontsize=label_fontsize)
    return cb 

def quandraticFit2d(input,verbose=False): 
    img = skimage.filters.gaussian(input,sigma=20)
    sizeX,sizeY = np.shape(img) 
    x = np.arange(0,sizeX,step=1,dtype=int) 
    y = np.arange(0,sizeY,step=1,dtype=int) 
    X, Y = np.meshgrid(x,y,copy=False) 
    X = X.flatten()
    Y = Y.flatten() 
    A =  np.array([X*0+1, X, Y, X**2,  Y**2, X*Y]).T  #np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T ##
    B = img.flatten()
    coeff, r, rank, _ = np.linalg.lstsq(A, B,rcond=-1)    
    zxy = coeff[1]*X+coeff[2]*Y+coeff[3]*(X**2) + coeff[4]*(Y**2)+coeff[5]*X*Y # coeff[1]*X+coeff[2]*Y+coeff[3]*X**2 + coeff[4]*X**2*Y+coeff[5]*X**2*Y**2+coeff[6]*Y**2+coeff[7]*X*Y**2+coeff[8]*X*Y #
    zxy = np.reshape(zxy,(sizeX,sizeY)) + coeff[0]

    if verbose: 
        figQ, axQ = create_fig()
        figQ.suptitle("fitted 2D bkgnd")
        axQ.imshow(zxy) 
        axQ.set_axis_off() 

        figQ2, axQ2 = create_fig(figsize=[7,6],ncols=2) 
        figQ2.suptitle("Image b/a bkgnd subtraction")
        axQ2[0].imshow(input) 
        axQ2[1].imshow(input-zxy) 
        axQ2[0].set_axis_off()
        axQ2[1].set_axis_off() 

    return zxy

def cellSeg(img,blur=True,verbose=False):
    if blur:
        img = skimage.filters.gaussian(img,sigma=5)
    img = np.asarray((img - np.amin(img))/(np.amax(img)-np.amin(img)) * 255,dtype=np.uint16) 
    #https://nbviewer.org/github/MouseLand/cellpose/blob/master/notebooks/run_cellpose.ipynb
    model = models.Cellpose(gpu=False,model_type="cyto")
    channels = [0,0] 
    masks, flows, styles, diams = model.eval(img, diameter=60, channels=channels,mask_threshold=np.amin(img))
    if verbose:
        fig, ax = create_fig(figsize=[6,5],nrows=1,ncols=2)
        ax[0].imshow(masks)
        ax[1].imshow(masks*img,cmap="jet")  
    return masks

def zeroPad2d(data, zero_pad=True):
    """Zero padd data 

    This is useful for padding zero in spatial domain where zeros are padded to XxY image without centering the image.
    Usually conducted before fft(fftshif(data))

    Parameters
    ----------
    data: 2d fload ndarray
        real-valued image data
    zero_pad: bool
        perform zero-padding to next order of 2
    """
    if zero_pad:
        # zero padding size is next order of 2
        (N, M) = data.shape
        order = int(max(64., 2**np.ceil(np.log(2 * max(N, M)) / np.log(2))))

        # this is faster than np.pad    
        datapad = np.zeros((order, order), dtype=data.dtype)               
        datapad[:data.shape[0], :data.shape[1]] = data
    else:
        datapad = data

    return datapad 




"""
inc_patch = mpatches.Patch(color=colors[0], label='C1')
rt_patch = mpatches.Patch(color=colors[1], label='C2')
lu_patch = mpatches.Patch(color=colors[2], label='C3')
ax1[0].legend(handles=[inc_patch, rt_patch,lu_patch],fontsize=8,borderpad=0.2,labelspacing=0.3,handlelength=1.0,handleheight=0.5,handletextpad=0.4,borderaxespad=0.2,columnspacing=1.0)

"""


"""
def MotionCorrection(img,preProcess=False,pw_rigid=True,**kwargs):
    #
    #Conducting motiong correction over 3D time-lapse data. 
    #Input:
    #    img: should be TxXxY data dim as real number 
    #Output:
    #    img_correct: motion corrected data. 
    
    import caiman 
    print("Doing motion correction...")
    timeM1 = time.time()
    T, X, Y = np.shape(img)
    
    if preProcess:
        out = img.copy() 
        #do median filter 
        for i in range(T):
            out[i,:,:] = ndimage.median_filter(out[i,:,:],size=5) 
    else:
        out = img.copy()

    if "refFrame" in kwargs.keys():
        refFrame = kwargs["refFrame"]
    else:
        refFrame = int(T/2) #using the central frame as reference frame 

    #check the complete list of parameters at: 
    # https://github.com/flatironinstitute/CaImAn/blob/master/caiman/motion_correction.py
    max_shifts = (20, 20)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (32, 32)  # overlap between pathes (size of patch strides+overlaps)
    num_frames_split = int(T/20)  # length in frames of each chunk of the movie (to be processed in parallel)
    if num_frames_split < 1:
        num_frames_split = 2 
    max_deviation_rigid = 10   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = pw_rigid  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    

    mc = cam.motion_correction.MotionCorrect(out, max_shifts=max_shifts,num_frames_split = num_frames_split,
                  strides=strides, overlaps=overlaps,pw_rigid = pw_rigid,use_cuda=True,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)
    mc.motion_correct(template=out[refFrame,:,:],save_movie=True)
    if pw_rigid:
        m_els = cam.load(mc.fname_tot_els) 
    else:
        m_els = cam.load(mc.mmap_file)
    timeM2 = time.time()
    print("Motion correction take {}min".format((timeM2-timeM1)/60)) 
    return m_els
"""
