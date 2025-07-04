import numpy as np
from scipy.io import savemat, loadmat 
from . import misc 
from scipy import signal
import matplotlib.pyplot as plt 
import h5py as hp 
from . import pyTorch_holo as pTH
from skimage.restoration import unwrap_phase 
from time import gmtime, strftime 
import lmfit 
import os 
import pathlib
from scipy import ndimage 
import tifffile 
from scipy.ndimage import median_filter
#: valid values for keyword argument `fit_offset` in :func:`estimate`
VALID_FIT_OFFSETS = ["fit", "gauss", "mean", "mode"]
#: valid values for keyword argument `fit_profile` in :func:`estimate`
VALID_FIT_PROFILES = ["offset", "poly2o", "tilt"]

def find_sideband(ft_data, which=+1, copy=True,returnIndx = False):
    """Find the side band position of a hologram
    The hologram is Fourier-transformed and the side band
    is determined by finding the maximum amplitude in
    Fourier space.
    Parameters
    ----------
    ft_data: 2d ndarray
        Fourier transform of the hologram image
    which: +1 or -1
        which sideband to search for:
        - +1: upper half
        - -1: lower half
    copy: bool
        copy `ft_data` before modification
    Returns
    -------
    fsx, fsy : tuple of floats
        coordinates of the side band in Fourier space frequencies
    """
    if copy:
        ft_data = ft_data.copy()

    if which not in [+1, -1]:
        raise ValueError("`which` must be +1 or -1!")

    ox, oy = ft_data.shape
    cx = ox // 2
    cy = oy // 2

    minlo = max(int(np.ceil(ox / 42)), 5)
    if which == +1:
        # remove lower part
        ft_data[cx - minlo:] = 0
    else:
        ft_data[:cx + minlo] = 0

    # remove values around axes
    center_around_px = int(ox/10)
    ft_data[cx - center_around_px:cx + center_around_px, :] = 0
    ft_data[:, cy - center_around_px:cy + center_around_px] = 0

    # find maximum
    am = np.argmax(np.abs(ft_data))
    iy = am % oy
    ix = int((am - iy) / oy)

    fx = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[0]))[ix]
    fy = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[1]))[iy]
    if returnIndx:
        return fx, fy, ix, iy 
    else:
        return fx, fy

def zeroPad2d(data, zero_pad=True):
    """Zero padd data 

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
        datapad = np.zeros((order, order), dtype=float)
        datapad[:data.shape[0], :data.shape[1]] = data
    else:
        datapad = data

    # Fourier transform
    #fft = np.fft.fftshift(np.fft.fft2(datapad))
    return datapad 


def mk_ellipse(XR,YR,X,Y):
    """
    make a elliptical shape for data filter
    Input:
    XR, YR: radius of ellipse
    X,Y: center of ellipse
    """
    XX, YY  = np.meshgrid(np.arange(0,X,step=1,dtype=int),np.arange(0,Y,step=1,dtype=int)) 
    return (((XX-X/2)/XR)**2 + ((YY-Y/2)/YR)**2)>1.0 


def HT2D(imgRaw,verbose=False,returnSimple=False,subtract_mean=True,zero_pad=True,**kwargs):
    """
    Hilbert transform for holography

    Input:
    img: 2D ndarray. 

    **kwargs: arguments 
        mi, mj: the (x,y) location of interference signal in frequency domain. 
        cmask: the mask to mask out the central signal in frequency domain. 
    """
    [xSizeRaw,ySizeRaw] = np.shape(imgRaw)   
    if subtract_mean:
        imgRaw = imgRaw - np.single(np.mean(imgRaw,dtype=np.float64))
    img = zeroPad2d(imgRaw,zero_pad=zero_pad) 
    [xSize,ySize] = np.shape(img)  
    if "mi" in kwargs.keys(): 
        mi = kwargs["mi"]  
        mj = kwargs["mj"]
        cmask = kwargs["cmask"] 
        fixedKR = True    
    else:
        fixedKR = False 
    
    Fimg = np.fft.fftshift(np.fft.fft2(img)) #frequency domain of raw image. 
    if not fixedKR:
        fsx, fsy = find_sideband(Fimg,which=1,copy=True) 
        mi = int(fsx * Fimg.shape[0])
        mj = int(fsy * Fimg.shape[1])

        # coordinates in Fourier space
        #assert Fimg.shape[0] == Fimg.shape[1]  # square-shaped Fourier domain
        fx = np.squeeze(np.fft.fftshift(np.fft.fftfreq(Fimg.shape[0])))
        fy = np.squeeze(np.fft.fftshift(np.fft.fftfreq(Fimg.shape[1]))) #
        [FXX,FYY] = np.meshgrid(fx**2,fy**2,indexing="ij")
        #TODO change the size of bandwidth to extract first order signal 
        dx = 0.091
        kkx = np.fft.fftshift(np.fft.fftfreq(Fimg.shape[0],d=dx))
        kky = np.fft.fftshift(np.fft.fftfreq(Fimg.shape[1],d=dx))
        dkx = np.abs(kkx[1] - kkx[0]) 
        dky = np.abs(kky[1] - kky[0])
        kres = 1.0*1.2*1.0/0.8 #1.0 NA, 0.8um wavelength 
        kxsize = int(kres/dkx)* np.abs(fx[1]-fx[0]) #change dx unit to 1     
        kysize = int(kres/dky)* np.abs(fy[1]-fy[0])     
        fxsize = kxsize #np.sqrt(fsx**2 + fsy**2) * 0.45
        fysize = kysize 
        #print("ksize {}".format(ksize))
        #print("fsize {}".format(np.sqrt(fsx**2 + fsy**2) * 0.45))

        sigmax = fxsize / 8
        taux = 2 * sigmax**2
        sigmay = fysize/8 
        tauy = 2*sigmay**2 
       # radsq = fx**2 + fy**2
       # disk = radsq <= fsize**2
        disk = (FXX+FYY)<(fxsize**2+fysize**2)
        gauss = np.exp(-(FXX/taux)-(FYY/tauy))
        cmask = signal.convolve(gauss, disk, mode="same")
        cmask /= cmask.max()

    dCF = Fimg * cmask #DC component 
    if verbose:
        fig, ax = misc.create_fig(figsize=[10,10],nrows=2,ncols=2)
        ax[1,0].imshow(np.log10(np.abs(dCF)+1e-6),cmap="rainbow")
        ax[1,0].set_title("dc with mask") 
        ax[1,0].set_axis_off() 
        ax[1,1].imshow(np.log10(np.abs(Fimg)+1e-6),cmap="rainbow")
        ax[1,1].scatter(int(mj+Fimg.shape[1]/2),int(mi+Fimg.shape[0]/2),s=20,alpha=0.5,color="red")
        ax[1,1].set_title("Fimg before mask") 
        ax[1,1].set_axis_off()
    Fimg = np.roll(Fimg,[-mi,-mj],axis=(0,1))  

    Fimg = Fimg * cmask #1st order inteference signal 
    if verbose:        
        ax[0,0].imshow(np.log10(np.abs(Fimg)+1e-6),cmap='rainbow') 
        ax[0,0].set_axis_off()
        ax[0,0].set_title("Fimg with mask") 
        ax[0,1].plot(cmask[:,int(np.shape(cmask)[1]/2)]) 
        ax[0,1].set_title("Mask") 
    Pimg = np.fft.ifft2(np.fft.ifftshift(Fimg))
    dc = np.abs(np.fft.ifft2(np.fft.ifftshift(dCF))) 
    contrast = 2 * np.sum(np.abs(Pimg),dtype=np.float64) / np.sum(np.abs(dc),dtype=np.float64)  #the contrast is defined as the amplitude ratio between interference signal to DC signal, with a factor of 2 accounting to both negative and positive frequencies. 
    if verbose:
        print("HT2D contrast is :{}".format(contrast)) 
        fig2, ax2 = misc.create_fig(figsize=[10,10],nrows=2,ncols=2) 
        ax2[0,0].imshow(np.angle(Pimg[:xSizeRaw,:ySizeRaw]),clim=[-3.14,3.14])
        ax2[0,0].set_axis_off() 
        ax2[0,1].imshow(unwrap_phase(np.angle(Pimg[:xSizeRaw,:ySizeRaw])),clim=[-3.14,3.14])
        ax2[0,1].set_axis_off()
        ax2[0,0].set_title("raw phase")
        ax2[0,1].set_title("Unwrapped phase")
        ax2[1,1].imshow(np.abs(Pimg[:xSizeRaw,:ySizeRaw])**2,cmap="rainbow") 
        ax2[1,1].set_axis_off() 
        ax2[1,1].set_title("ac int")
        ax2[1,0].imshow(np.abs(Pimg[:xSizeRaw,:ySizeRaw])**2/np.abs(dc[:xSizeRaw,:ySizeRaw])**2,cmap="rainbow") 
        ax2[1,0].set_title("dc int") 
        
    if returnSimple:
        return np.asarray(Pimg[:xSizeRaw,:ySizeRaw],dtype=np.complex64)
    else:
        return np.asarray(Pimg[:xSizeRaw,:ySizeRaw],dtype=np.complex64), contrast, mi, mj, cmask



def ima2(IMGRaw,verbose=False,subtract_mean=True,zero_pad=True,returnSimple=False,Sref = None, Sref_indx=None, **kwargs):
    """
    data a basic data processing of holography imaging from laser speckle ODT LCI system. 
    Input:
    IMG: raw data and must be three dimensional data;
        raw data from MATLAB (v7.3) is based on matrix order (x,y,z)
        while loading mat via h5py is in order (z,y,x) 
    Options: 
    kwargs: 
    ref: reference image data; 
    cr: range of image to be cropped when calculated intensity or contrast
        intensity is the square of abs (amplitude) of complex image
        contrast is the AC amplitude/DC amplitude average value.
        0.5 indicate 0.5 proportion of whole field will be included. To retain the whole FOV, set cr = 1.0 
    
    Output:
    out: complex images of reconstructed images
    Pref: complex images of ref
    out_inten: output intensity after cropped. defined as square of amplitude of complex image
    mi, mj: peak of AC or interference signal in frequency domain 
    cmask: filter to filter out DC or AC signal on center 
    contrast_mat: AC amplitude /DC amplitude 
    mi_raw,mj_raw: mi, mj before reference to center pixel; that's the coordinates in the original image.
    """
    if not IMGRaw.dtype == np.single: 
        IMGRaw = np.single(IMGRaw) 
    if IMGRaw.ndim == 2:
        IMGRaw = IMGRaw[np.newaxis,:,:]
    szRaw = np.shape(IMGRaw) #[Z,Y,X] 
    if subtract_mean:
        for i in range(szRaw[0]):
            IMGRaw[i,:,:] = IMGRaw[i,:,:] - np.single(np.mean(IMGRaw[i,:,:],dtype=np.float64)) #setting dtype as np.float64 to avoid overflow of summation.
    if zero_pad:
        for i in range(szRaw[0]):
            if i == 0:
                tmp = zeroPad2d(IMGRaw[i,:,:],zero_pad=zero_pad) 
                IMG = np.zeros((szRaw[0],np.shape(tmp)[0],np.shape(tmp)[1]),dtype=IMGRaw.dtype) 
                IMG[i,:,:] = tmp 
            else:
                IMG[i,:,:] = zeroPad2d(IMGRaw[i,:,:],zero_pad=zero_pad) 
        sz = np.shape(IMG)    
    else:
        sz = szRaw 
        IMG = IMGRaw
    
    if "ref" in kwargs.keys():
        ref = kwargs["ref"]
        SubRef = True 
        if ref.ndim == 3:
            ref = np.mean(ref,axis=0) #average 
        if verbose:
            print("Ref provided! ")
    else:
        SubRef = False

    #tmp = IMG - np.roll(IMG,[0,2,2],axis=(0,1,2)) #subtract the image by itself  
    #conttmp = np.squeeze(np.mean(IMG,axis=(1,2),dtype=np.float64)) 
    #Sref_indx = np.argmax(conttmp) 
    #Sref = np.squeeze(IMG[Sref_indx,:,:]) 
    
    #useing entropy to determine focal plane 
    #sEntropy = np.zeros((IMG.shape[0],))
    #for ii in range(IMG.shape[0]):
    #    tmpImg = IMG[ii,:,:]
    #    marg = np.histogramdd(np.ravel(tmpImg), bins = int(tmpImg.shape[1]/2))[0]/tmpImg.size
    #    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    #    sEntropy[ii] = np.sum(np.multiply(marg, np.log2(marg))) #inverse entropy 
    #sEntropy = median_filter(sEntropy,3) 
    #Sref_indx = np.argmax(sEntropy)
    #Sref = np.squeeze(IMG[Sref_indx,:,:]) 

    #directly use amax to get indx 
    if Sref is None:
        Sref_indx = np.argmax(np.amax(IMG,axis=(1,2))) 
        Sref = np.squeeze(IMG[Sref_indx,:,:]) 

    if "cr" in kwargs.keys():
        cr = kwargs["cr"] 
        if cr > 1.0:
            print("Warning: cr forced to be 1.0!")
            cr = 1.0
    else:
        cr = 0.4 
    #define DC window 
    if cr == 1.0:
        hh = np.arange(0,szRaw[1]+1,step=1,dtype=int) 
        ww = np.arange(0,szRaw[2]+1,step=1,dtype=int) 
    else:
        oszy = int(np.round(szRaw[1]*cr))
        oszx = int(np.round(szRaw[2]*cr))
        hh = np.arange(-np.round(oszy/2),np.round(oszy/2),step=1,dtype=int)  + int(np.round(szRaw[1]/2))
        ww = np.arange(-np.round(oszx/2),np.round(oszx/2),step=1,dtype=int)  + int(np.round(szRaw[2]/2))
 
    if "mi" in kwargs.keys():
        mi = kwargs["mi"]
        mj = kwargs["mj"]
        if "cmask" in kwargs.keys():
            cmask = kwargs["cmask"]
        else:
            #TODO: something wrong in this code 
            # coordinates in Fourier space
            # only works for not zeroPad processing         
            fx = np.squeeze(np.fft.fftshift(np.fft.fftfreq(IMG.shape[1])))
            fy = np.squeeze(np.fft.fftshift(np.fft.fftfreq(IMG.shape[2]))) #
            [FXX,FYY] = np.meshgrid(fx**2,fy**2,indexing="ij")
            #TODO change the size of bandwidth to extract first order signal 
            dx = 0.091
            kkx = np.fft.fftshift(np.fft.fftfreq(IMG.shape[1],d=dx))
            kky = np.fft.fftshift(np.fft.fftfreq(IMG.shape[2],d=dx))
            dkx = np.abs(kkx[1] - kkx[0]) 
            dky = np.abs(kky[1] - kky[0])
            kres = 1.2*1.0/0.8#1.0 NA, 0.8um wavelength 
            kxsize = int(kres/dkx)* np.abs(fx[1]-fx[0]) #change dx unit to 1     
            kysize = int(kres/dky)* np.abs(fy[1]-fy[0])     
            fxsize = kxsize #np.sqrt(fsx**2 + fsy**2) * 0.45
            fysize = kysize 
            #print("ksize {}".format(ksize))
            #print("fsize {}".format(np.sqrt(fsx**2 + fsy**2) * 0.45))

            sigmax = fxsize / 8
            taux = 2 * sigmax**2
            sigmay = fysize/8 
            tauy = 2*sigmay**2 
            # radsq = fx**2 + fy**2
            # disk = radsq <= fsize**2
            disk = (FXX+FYY)<(fxsize**2+fysize**2)
            gauss = np.exp(-(FXX/taux)-(FYY/tauy))
            cmask = signal.convolve(gauss, disk, mode="same")
            cmask /= cmask.max()
    else:
        _, _, mi, mj, cmask = HT2D(Sref,zero_pad=zero_pad, subtract_mean=False, verbose=verbose) 
    IMGk = np.fft.fftshift(np.fft.fft2(IMG,axes=(1,2)),axes=(1,2))
    DCk = IMGk * np.repeat(cmask[np.newaxis,:,:], sz[0], axis=0)
    
    
    IMGk = np.roll(IMGk,(0,-mi,-mj),axis=(0,1,2)) 
    IMGk = IMGk *  np.repeat(cmask[np.newaxis,:,:], sz[0], axis=0) 
    out = np.fft.ifft2(np.fft.ifftshift(IMGk,axes=(1,2)),axes=(1,2))     
    DC = np.abs(np.fft.ifft2(np.fft.ifftshift(DCk,axes=(1,2)),axes=(1,2))) 

    if SubRef:
        Pref = HT2D(ref,verbose=verbose,returnSimple=True,zero_pad=zero_pad) 
        refPhase = np.conjugate(Pref) 
        refAmp =  np.abs(refPhase) * np.exp(1j*np.zeros(np.shape(refPhase)))   
        for i in range(sz[0]):
            out[i,:,:] = out[i,:,:] * refPhase/refAmp  
    else:
        Pref = np.zeros(np.shape(out),dtype=np.short)   
    
    if zero_pad:
        out_crop = out[:,:szRaw[1],:szRaw[2]]
        DC_crop = DC[:,:szRaw[1],:szRaw[2]]
    else:
        out_crop = out 
        DC_crop = DC
    #calculate intensity distribution of an image in a smaller area around center 

    out_cut = out_crop[:,hh[0]:hh[-1],ww[0]:ww[-1]]
    DC_cut = DC_crop[:,hh[0]:hh[-1],ww[0]:ww[-1]]
    if verbose:
        print("Shape of cropped image is:{}".format(DC_cut.shape))

    contrast_mat = np.single(np.squeeze(2*np.sum(np.abs(out_cut),axis=(1,2),dtype=np.float64)/np.sum(np.abs(DC_cut),axis=(1,2),dtype=np.float64)))
    out_inten = np.squeeze(np.single(np.mean(np.abs(out_cut)**2,axis=(1,2),dtype=np.float64)))
    
    if verbose:
        print("Found peak position {} {}".format(mi,mj))
   
    #out = np.conjugate(out) 

    if returnSimple:
        return np.asarray(out_crop,dtype=np.complex64)
    else:
        return np.asarray(out_crop,dtype=np.complex64), Pref[:sz[1],:sz[2]], out_inten, mi, mj, cmask, contrast_mat
 

def ShowRecontructedImg(dataRaw,out,mi_raw, mj_raw,trans=False):
    fig, ax = misc.create_fig(figsize=[10,10],nrows=2,ncols=2) 
    if trans:
        ax[0,0].imshow(np.transpose(dataRaw)**0.4,cmap="gray")
    else:
        ax[0,0].imshow((dataRaw)**0.4,cmap="gray")
    ax[0,0].set_title("raw data")

    if trans:
        ax[0,1].imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(dataRaw)))),cmap="gray")
    else:
        ax[0,1].imshow(np.transpose(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(dataRaw))))),cmap="gray")
    c1 = plt.Circle(((mi_raw+0.5)*int(np.shape(dataRaw)[0]),(mj_raw-0.5)*int(np.shape(dataRaw)[1])),10, color='g',fill=False,ls=":")
    ax[0,1].add_patch(c1) 
    ax[0,1].set_title("spectrum") 

    if trans:
        ax[1,0].imshow(np.transpose(np.abs(out))**0.4,cmap="gray")
    else:
        ax[1,0].imshow((np.abs(out))**0.4,cmap="gray")
    ax[1,0].set_title("amplitude")

    if trans:
        im = ax[1,1].imshow(np.transpose(np.angle(out)),cmap="jet")
    else:
        im = ax[1,1].imshow((np.angle(out)),cmap="jet")
    ax[1,1].set_title("phase") 
    fig.colorbar(im)

    return fig, ax  


def PhaseReconstruction_Batch(inputData,batchSize=50,cr=0.5,trans=False,ref=np.empty((0,)),onlyCPU=False, verbose=False,zero_pad=True,subtract_mean=True, returnContrastMat=False,**kwargs):
    """
    Compute phase raw data with a batch calculation. Return a complex output signal. 
    Note: Z >= batchSize
    """
    [Z,xRaw,yRaw] = np.shape(inputData) 
    inputData = np.single(inputData) 
    if subtract_mean:
        for i in range(Z):
            inputData[i,:,:] = inputData[i,:,:] - np.single(np.mean(inputData[i,:,:],dtype=np.float64))

    NumGroup = int(np.floor(Z/batchSize)) 
    output = np.zeros(inputData.shape,dtype=np.complex64)
    contrast_mat = np.zeros((Z,)) 
    out_int = np.zeros((Z,)) 
    xrange = np.arange(0,batchSize,step=1,dtype=int) 
    ##get the focal plane 
    Sref_indx = np.argmax(np.amax(inputData,axis=(1,2))) 
    Sref = np.squeeze(inputData[Sref_indx,:,:]) 
    #print("Focal plane found in plane of {}".format(Sref_indx))

    if (Z <2) or (onlyCPU):  #now, since padding, GPU always faster than CPU, even for a single frame. 
        #print("CPU mode selected!")
        if verbose:
            print("CPU mode selected!")
        for i in range(NumGroup):
            #out_crop, Pref[:sz[1],:sz[2]], out_inten, mi, mj, cmask, contrast_mat
            if np.size(ref)==0:
                output[i*batchSize+xrange,:,:],_,out_int[i*batchSize+xrange], mi, mj, cmask, contrast_mat[i*batchSize+xrange]= ima2(inputData[i*batchSize+xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx, **kwargs)
            else:
                output[i*batchSize+xrange,:,:],_,out_int[i*batchSize+xrange], mi, mj, cmask, contrast_mat[i*batchSize+xrange]= ima2(inputData[i*batchSize+xrange,:,:],verbose=verbose,cr=cr,ref=ref,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
            #if i == 0:
            #    ShowRecontructedImg(inputData[0,:,:],output[0,:,:],mi_raw,mj_raw,trans)
        if (Z % batchSize): 
            tmp_xrange = np.arange(0,Z % batchSize,step=1,dtype=int)   
            if np.size(ref) == 0:
                try:   
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,)
                except:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask,contrast_mat[NumGroup*batchSize + tmp_xrange] = ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
            else:
                try:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange]= ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,ref=ref,zero_pad=zero_pad,subtract_mean=False,verbose=verbose,Sref=Sref, Sref_indx=Sref_indx,)
                except:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
    else:
        #print("GPU mode selected!")
        if verbose:
            print("GPU mdoe selected!")
        for i in range(NumGroup):
            #out, Pref, out_inten, mi, mj, cmask, contrast_mat, mi_raw,mj_raw
            if np.size(ref)==0:
                output[i*batchSize+xrange,:,:],_,out_int[i*batchSize+xrange], mi, mj, cmask, contrast_mat[i*batchSize+xrange]= pTH.ima2(inputData[i*batchSize+xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
            else:
                output[i*batchSize+xrange,:,:],_,out_int[i*batchSize+xrange],mi, mj, cmask, contrast_mat[i*batchSize+xrange]= pTH.ima2(inputData[i*batchSize+xrange,:,:],cr=cr,ref=ref,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
            #if i == 0:
                
        if (Z % batchSize): 
            tmp_xrange = np.arange(0,Z % batchSize,step=1,dtype=int)   
            if np.size(ref) == 0:
                try:   
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = pTH.ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,verbose=verbose,subtract_mean=False,zero_pad=zero_pad,Sref=Sref, Sref_indx=Sref_indx,)
                except:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = pTH.ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
            else:
                try:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange], mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = pTH.ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,ref=ref,zero_pad=zero_pad,subtract_mean=False,verbose=verbose,Sref=Sref, Sref_indx=Sref_indx,)
                except:
                    output[NumGroup*batchSize + tmp_xrange,:,:],_,out_int[NumGroup*batchSize + tmp_xrange],mi, mj, cmask, contrast_mat[NumGroup*batchSize + tmp_xrange] = pTH.ima2(inputData[NumGroup*batchSize+tmp_xrange,:,:],cr=cr,ref=ref,verbose=verbose,zero_pad=zero_pad,subtract_mean=False,Sref=Sref, Sref_indx=Sref_indx,**kwargs)
    
    if verbose:
        ShowRecontructedImg(inputData[int(Z/2),:,:],output[int(Z/2),:,:],mi,mj,trans)

    if returnContrastMat:
        return output,contrast_mat, out_int
    else:
        return output




def offset_gaussian(data):
    """Fit a gaussian model to `data` and return its center"""
    nbins = 2 * int(np.ceil(np.sqrt(data.size)))
    mind, maxd = data.min(), data.max()
    drange = (mind - (maxd - mind) / 2, maxd + (maxd - mind) / 2)
    histo = np.histogram(data, nbins, density=True, range=drange)
    dx = abs(histo[1][1] - histo[1][2]) / 2
    hx = histo[1][1:] - dx
    hy = histo[0]
    # fit gaussian
    gauss = lmfit.models.GaussianModel()
    pars = gauss.guess(hy, x=hx)
    out = gauss.fit(hy, pars, x=hx)
    return out.params["center"]


def offset_mode(data):
    """Compute Mode using a histogram with `sqrt(data.size)` bins"""
    nbins = int(np.ceil(np.sqrt(data.size)))
    mind, maxd = data.min(), data.max()
    histo = np.histogram(data, nbins, density=True, range=(mind, maxd))
    dx = abs(histo[1][1] - histo[1][2]) / 2
    hx = histo[1][1:] - dx
    hy = histo[0]
    idmax = np.argmax(hy)
    return hx[idmax]


def profile_tilt(data, mask):
    """Fit a 2D tilt to `data[mask]`"""
    params = lmfit.Parameters()
    params.add(name="mx", value=0)
    params.add(name="my", value=0)
    params.add(name="off", value=np.average(data[mask]))
    fr = lmfit.minimize(tilt_residual, params, args=(data, mask))
    bg = tilt_model(fr.params, data.shape)
    return bg


def profile_poly2o(data, mask):
    """Fit a 2D 2nd order polynomial to `data[mask]`"""
    # lmfit
    params = lmfit.Parameters()
    params.add(name="mx", value=0)
    params.add(name="my", value=0)
    params.add(name="mxy", value=0)
    params.add(name="ax", value=0)
    params.add(name="ay", value=0)
    params.add(name="off", value=np.average(data[mask]))
    fr = lmfit.minimize(poly2o_residual, params, args=(data, mask))
    bg = poly2o_model(fr.params, data.shape)
    return bg


def poly2o_model(params, shape):
    """lmfit 2nd order polynomial model"""
    mx = params["mx"].value
    my = params["my"].value
    mxy = params["mxy"].value
    ax = params["ax"].value
    ay = params["ay"].value
    off = params["off"].value
    bg = np.zeros(shape, dtype=float) + off
    x = np.arange(bg.shape[0]) - bg.shape[0] // 2
    y = np.arange(bg.shape[1]) - bg.shape[1] // 2
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    bg += ax * x**2 + ay * y**2 + mx * x + my * y + mxy * x * y
    return bg


def poly2o_residual(params, data, mask):
    """lmfit 2nd order polynomial residuals"""
    bg = poly2o_model(params, shape=data.shape)
    res = (data - bg)[mask]
    return res.flatten()


def tilt_model(params, shape):
    """lmfit tilt model"""
    mx = params["mx"].value
    my = params["my"].value
    off = params["off"].value
    bg = np.zeros(shape, dtype=float) + off
    x = np.arange(bg.shape[0]) - bg.shape[0] // 2
    y = np.arange(bg.shape[1]) - bg.shape[1] // 2
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    bg += mx * x + my * y
    return bg


def tilt_residual(params, data, mask):
    """lmfit tilt residuals"""
    bg = tilt_model(params, shape=data.shape)
    res = (data - bg)[mask]
    return res.flatten()



def compute_bkgnd(data,fit_offset="mean",fit_profile="tilt",border_m=0, border_perc=0,border_px=0,from_mask=None,ret_mask=False):
    """Estimate the background value of an image

    Parameters
    ----------
    data: np.ndarray
        Data from which to compute the background value
    fit_profile: str
        The type of background profile to fit:

        - "offset": offset only
        - "poly2o": 2D 2nd order polynomial with mixed terms
        - "tilt": 2D linear tilt with offset (default)
    fit_offset: str
        The method for computing the profile offset

        - "fit": offset as fitting parameter
        - "gauss": center of a gaussian fit
        - "mean": simple average
        - "mode": mode (see `qpimage.bg_estimate.mode`)
    border_m: float
            Assume that a frame of `border_m` meters around the
            image is background. The value is converted to
            pixels and rounded.
    border_perc: float
            Assume that a frame of `border_perc` percent around
            the image is background. The value is converted to
            pixels and rounded. If the aspect ratio of the image
            is not one, then the average of the data's shape is
            used to compute the percentage in pixels.
    border_px: float
            Assume that a frame of `border_px` pixels around
            the image is background.
    from_mask: boolean np.ndarray or None
        Use a boolean array to define the background area.
        The boolean mask must have the same shape as the
        input data. `True` elements are used for background
        estimation.
    ret_mask: bool
        Return the boolean mask used to compute the background.

    Notes
    -----
    If both `border_px` and `from_mask` are given, the
    intersection of the two is used, i.e. the positions
    where both, the frame mask and `from_mask`, are
    `True`.
    """
    if data.ndim == 2:
        data = data[np.newaxis,:,:]
    #get border in px 
    border_list = [] 
    if border_m:
        if border_m < 0:
            raise ValueError("border_m must be greater than zero!")
    if border_perc:
        if (border_perc<0) or (border_perc>50):
            raise ValueError("border_perc must be in interval [0,50]!")
        size = np.average([np.shape(data)[1],np.shape(data)[2]])
        border_list.append(size*border_perc/100) 
    if border_px:
        border_list.append(border_px)
    #get maximum border size
    if border_list:
        border_px = int(np.round(np.max(border_list))) 
    elif from_mask is None:
        raise ValueError('Neither from_mask nor border_* given!')
    elif np.all(from_mask== 0):
        raise ValueError('from_mask must not be all-zeros!')

    Z = np.shape(data)[0] 
    bg = np.zeros(data.shape,dtype=data.dtype) 
    if fit_profile not in VALID_FIT_PROFILES:
        msg = "`fit_profile` must be one of {}, got '{}'".format(
            VALID_FIT_PROFILES,
            fit_profile)
        raise ValueError(msg)
    if fit_offset not in VALID_FIT_OFFSETS:
        msg = "`fit_offset` must be one of {}, got '{}'".format(
            VALID_FIT_OFFSETS,
            fit_offset)
        raise ValueError(msg)
    # initial mask image
    if from_mask is not None:
        assert isinstance(from_mask, np.ndarray)
        mask = from_mask.copy()
    else:
        mask = np.ones_like(data, dtype=bool)
    # multiply with border mask image (intersection)
    if border_px > 0:
        border_px = int(np.round(border_px))
        mask_px = np.zeros_like(mask)
        mask_px[:,:border_px, :] = True
        mask_px[:,-border_px:, :] = True
        mask_px[:,:, :border_px] = True
        mask_px[:,:, -border_px:] = True
        # intersection
        np.logical_and(mask, mask_px, out=mask)
    # compute background image
    if fit_profile == "tilt":
        for i in range(Z):
            bg[i,:,:] = profile_tilt(data[i,:,:], mask[i,:,:])
    elif fit_profile == "poly2o":
        for i in range(Z):
            bg[i,:,:] = profile_poly2o(data[i,:,:], mask[i,:,:])
    else:
        bg = np.zeros_like(bg, dtype=float)
    # add offsets
    if fit_offset == "fit":
        if fit_profile == "offset":
            msg = "`fit_offset=='fit'` only valid when `fit_profile!='offset`"
            raise ValueError(msg)
        # nothing else to do here, using offset from fit
    elif fit_offset == "gauss":
        for i in range(Z):
            bg[i,:,:] = bg[i,:,:] + offset_gaussian((data[i,:,:] - bg[i,:,:])[mask[i,:,:]])
    elif fit_offset == "mean":
        for i in range(Z):
            bg[i,:,:] = bg[i,:,:] + np.mean((data[i,:,:] - bg[i,:,:])[mask[i,:,:]])
    elif fit_offset == "mode":
        for i in range(Z):
            bg[i,:,:] = bg[i,:,:] + offset_mode((data[i,:,:] - bg[i,:,:])[mask[i,:,:]])

    if ret_mask:
        ret = (np.squeeze(bg), np.squeeze(mask))
    else:
        ret = np.squeeze(bg)
    return ret

def divmod_neg(a, b):
    """Return divmod with closest result to zero"""
    q, r = divmod(a, b)
    # make sure r is close to zero
    sr = np.sign(r)
    if np.abs(r) > b/2:
        q += sr
        r -= b * sr
    return q, r


def proc_unwrap_phase(pha):
    """
    Conducting phase unwraping, taking inout accounts of nans 
    """
    if pha.ndim == 2:
        pha = pha[np.newaxis,:,:]
    Z = np.shape(pha)[0]
    nanmask = np.isnan(pha)
    if np.sum(nanmask):
        pham = pha.copy() 
        pham[nanmask] = 0
        for i in range(Z):
            pham[i,:,:] = np.ma.masked_array(pham[i,:,:],mask=nanmask[i,:,:])
            pha[i,:,:] = unwrap_phase(pham[i,:,:],seed=47)
            pha[nanmask[i,:,:]] = np.nan 
    else:
        for i in range(Z):
            pha[i,:,:] = unwrap_phase(pha[i,:,:],seed=47)
    #remove 2PI offset that might be present in the border phase 
    for i in range(Z):
        border = np.concatenate((pha[i,0, :],
                                 pha[i,-1, :],
                                 pha[i,:, 0],
                                 pha[i,:, -1]))
        twopi = 2*np.pi
        minimum = divmod_neg(np.nanmin(border), twopi)[0]
        offset = minimum * twopi
        pha[i,:,:] = pha[i,:,:] - offset        

    return np.squeeze(pha)

def loadTiff(fileName):
    with tifffile.TiffFile(fileName) as tif:
            img = []
            tt = []
            for ii in range(len(tif.pages)):
                img.append(tif.pages[ii].asarray())
                for tag in tif.pages[ii].tags.values(): 
                    name, value = tag.name, tag.value
                    if name == "DateTime":
                        tt.append(np.single(value))
            tt = np.asarray(tt) 
            img = np.array(img)
    
    return img, tt 

class QPImage(object):
    _instance = 0 
    def __init__(self,inData=None,data_key='IMG',ref_data=None,meta_data=None,holo_kw=None,bg_kw = None, proc_phase=False,computeBg=False, slices=0, autoRun = True, verbose=False,**kwargs):
        """Quantitative phase image manipulation

            This class implements various tasks for quantitative phase
            imaging, including phase unwrapping, background correction,
            numerical focusing, and data export.

            Parameters
            ----------
            inData: 2d ndarray (float or complex) or list
                The experimental data (see `which_data`). If data is a file .mat or .h5 or .hdf5, it will automatically load data where the keyword is given by
                data_key or automatically search for "IMG" or "data". 
            data_key: the key for accessing data array if "data" is defined as a file format; Default is None,
                then it will iterates automaticaly through the data file and get the first keys of either "data" or "IMG" 
            ref_data: reference image. could be (X,Y) or (T,X,Y)
            meta_data: dict
                Meta data associated with the input data.
                see :data:`qpimage.meta.META_KEYS`
            holo_kw: dict
                Special keyword arguments for phase retrieval from
                hologram data (`which_data="hologram"`).
                default: {"batchSize":50,"cr":0.5,"trans":False,"onlyCPU":False, "verbose":verbose,"zero_pad":True,"subtract_mean":True, "returnContrastMat":True} 
                    batchSize: the batch size to divide raw data into small batches in case overflowing memory. If overmemory happens, reduce the batchSize.
                    cr: proportion of image to be counted when calculating interference contrast. 
                    trans: transpose the raw data. usually to make it identical dimension to MATLAB
                    onlyCPU: only using CPU. If False, it will choose GPU computation resource. 
                    verbose: show intermediate results. better to set as False when dealing with large amount of data
                    zero_pad:False by default; if True,do zero padding 
                    subtract_mean:True, subtract mean of rawdata before recontruction.
                    returnContrastMat:True, get the contrast results also. 
            bg_kw: dict
                keyword for estimatign background title. 
                here right now, default: {"fit_offset":"mean", "fit_profile":"tilt","border_px":6}
            computeBg: bool, default as True. whether or not to compute background tilt, using parameters from bg_kw. 

            proc_phase: bool
                Process the phase data. This includes phase unwrapping
                using :func:`skimage.restoration.unwrap_phase` and
                correcting for 2PI phase offsets (The offset is estimated
                from a 1px-wide border around the image  
            slices: int, 
                default 0, indicating it will process all the frames. Otherwise, it will only process 0:slices frames. 
        Members and attributes:
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

        """
        if ((ref_data is None)):
            self.ref = np.empty((0,))
            self.subtract_ref = False
        else:
            self.subtract_ref = True  #will do reference subtraction by: reconstructing phase of reference follow the same way of image recontruction and then do a subtraction.
            if isinstance(ref_data,str):
                if os.path.isfile(ref_data):
                    ref_data0 = loadTiff(ref_data)
                else:
                    raise ValueError("The refData must be eitehr a full-path to a file or a NDarray!")
            else:
                ref_data0 = ref_data 
            if ref_data0.ndim == 3:
                self.ref =np.squeeze(np.mean(np.double(ref_data0),axis=0))
            else:
                self.ref = np.squeeze(np.double(ref_data0))

        # now force "ref" as empty to avoid doing reference subtraction in main procesing. 
        self.holo_kw = {"batchSize":50,"cr":0.5,"trans":False,"ref":np.empty((0,)),"onlyCPU":False, "verbose":verbose,"zero_pad":False,"subtract_mean":True, "returnContrastMat":True}  #
        if holo_kw is not None:
            for keys in holo_kw:
                self.holo_kw[keys] = holo_kw[keys]

        self.bg_kw = {"fit_offset":"mean", "fit_profile":"tilt","border_m":0, "border_perc":0, "border_px":80,"from_mask":None, "ret_mask":False}
        
        if bg_kw is not None:
            for keys in bg_kw:
                self.bg_kw[keys] = bg_kw[keys] 

        self.meta = {}
        if meta_data is not None:
            for keys in meta_data.keys():
                self.meta[keys] = meta_data[keys]

        if not "time" in self.meta.keys():
            self.meta["time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        if (inData is not None and not isinstance(inData, (np.ndarray, list, tuple))):         
            if isinstance(inData,(str,pathlib.Path)):
                if inData.endswith(".mat") or inData.endswith(".hdf5") or inData.endswith(".h5"):
                    self.meta["label"] = inData.split(".")[0]                    
                    dFile, data = self._get_data(inData,data_key=data_key)
                    for k in dFile.keys():
                        if (k == "data") or (k == "IMG"):
                            pass 
                        else:
                            self.meta[k] = dFile[k][:] 
                        #print(self.meta[k])              
                elif inData.endswith(".tif"):
                    with tifffile.TiffFile(inData) as tif:
                        img = []
                        tt = []
                        for ii in range(len(tif.pages)):
                            img.append(tif.pages[ii].asarray())
                            for tag in tif.pages[ii].tags.values(): 
                                name, value = tag.name, tag.value
                                if ii == 0 and (not name == "DateTime"):
                                    self.meta[name] = value
                                if name == "DateTime":
                                    tt.append(np.single(value))
                        if "tl" in os.path.basename(inData):
                            self.meta["timepoint"] = np.array(tt)
                        elif "zstack" in os.path.basename(inData):
                            self.meta["zpos"] = np.array(tt)
                        img = np.array(img)
                        data = np.array(img) 
                        del img 
                        del tt 
                else:
                    raise ValueError("input should be either NDarray, .h5py, .h5 or .mat file!")
            else:
                try:
                    data = inData[()] #in case some times it will give the inData as a type of hdf5 dataset 
                    self.meta["label"] = "img_"
                    inData = [] 
                except:
                    raise ValueError(f"input type {type(inData)} should be either NDarray, .h5py, .h5 or .mat file!")
        else:
            self.meta["label"] = "img_"
            data = inData.copy() 
            inData = [] 
            #print(f"Warning: {type(inData)} is not associated data type found!")

        #make sure the xy axis are equally sized
        #if data.shape[1] > data.shape[2]:
        #    data = data[:,0:data.shape[2],:]
        #elif data.shape[1] < data.shape[2]:
        #    data = data[:,:,0:data.shape[1]]   


        #print(data.shape)
        QPImage._instance += 1 

        if data is not None:
            #if data.ndim == 2:
            #    data = data[np.newaxis,:,:]
            if autoRun:
                if data.ndim == 3:
                    if slices == 0:
                        self.out, self.contrast_mat, self.out_int = self._get_amp_pha(data[:,:,:],**kwargs) 
                    else:
                        self.out, self.contrast_mat, self.out_int = self._get_amp_pha(data[0:int(slices),:,:],**kwargs)
                else:
                    self.out,self.contrast_mat, self.out_int = self._get_amp_pha(data,**kwargs) 
                #conducting median filter on complex signals (using 7 as kernel size since the PSF is sampling by 7 pixels) 
                #TODO: considering using median filter. 
                #if self.out.ndim == 3:
                #    for ii in range(np.shape(self.out)[0]):
                #        self.out[ii,:,:] = ndimage.median_filter(np.real(self.out[ii,:,:]),size=7) + 1j*ndimage.median_filter(np.imag(self.out[ii,:,:]),size=7)
                #else:
                #    self.out = ndimage.median_filter(np.real(self.out),size=7) + 1j*ndimage.median_filter(np.imag(self.out),size=7)
                # done with median filter
                if self.subtract_ref:
                    tmp_ref, _, _ =  np.squeeze(self._get_amp_pha(self.ref,**kwargs))
                    for pp in range(np.shape(self.out)[0]):  
                        self.out[pp,:,:] = self.out[pp,:,:]/tmp_ref 

                tmpPha = np.angle(self.out) 
                    
                if proc_phase:
                    self.pha = proc_unwrap_phase(tmpPha)                    
                else:
                    self.pha = tmpPha
                    if self.subtract_ref:
                        self.ref_out = tmp_ref
                        del tmp_ref
                    del tmpPha    
                if computeBg:
                    if self.subtract_ref:
                        self.pha, self.ref_out = self.compute_bg()
                    else:
                        #self.pha = self.compute_bg() 
                        pass   
                   
                del data 

    def _get_data(self,inData,data_key=None): 
        if not os.path.exists(inData):
            raise ValueError(inData+" not exist!") 
        try:
            dFile = hp.File(inData,"r")
            fileType = 0 
        except:
            dFile = loadmat(inData) 
            fileType = 1 
        
        dFile0 = dFile["data"]
        if data_key is not None:
            data0 = dFile0[data_key] 
        else:
            data0 = dFile0[self._get_keys(dFile0,data_key)] #get first order of key words
            data_key = self._get_keys(dFile0,data_key)
        try:
            out = np.asarray(data0) #if so, it turn out that it is a dict containing all parameters
           # print("0 out {}".format(type(out)))
            searchMeta = 1 
        except: 
            data_key = self._get_keys(data0,data_key)
            out = np.asarray(data0[data_key])  
            searchMeta = 2 
        else:
            raise ValueError("No Data Keys Found!")
        
        if searchMeta == 1:
            outFile = dFile0 
            #print(dFile0.keys())
            #try:
            #    for keys0 in dFile0.keys():
            #        if not keys0 == data_key:
            #            self.meta[keys0] = dFile0[keys0] 
            #except:
            #    pass 
            #else:
            #    raise Warning("No other meta keys found in dataset!")
        elif searchMeta == 2:
            outFile = dFile0[data_key]
           # tmp = dFile0[data_key] 
           # print(tmp.keys()) 
           # try:
           #     for keys0 in tmp.keys():
           #         if not keys0 == data_key:
           #             self.meta[keys0] = tmp[keys0]         
           # except:
           #     pass 
           # else:
           #     raise Warning("No other meta keys found in dataset!")
                 
        try:
            del dFile0["IMG"]
        except:
            pass 
        try:
            del dFile0["data"] 
        except:
            pass 
        try:
            del dFile0["ref"] 
        except:
            pass 
        return outFile, out 
    


    def _get_keys(self,data0,data_key):
        data_keyList = []
        for item in data0.keys():
            if item.isalpha():
                data_keyList.append(item)     
        for item in data_keyList:
            if (item == "ref") or (item == "bkgnd"):
                ref_key = item
            elif (item == "data") or (item == "IMG") or (item == data_key):
                dout_key = item 

        if len(dout_key) == 0:
            raise ValueError("Key not found in dataset!")
            
        return dout_key  #currently ignore ref data saved in a same dataset. 

    def _get_amp_pha(self,data,**kwargs):
        #HT2D(imgRaw,verbose=True,returnSimple=False,subtract_mean=True,zero_pad=True,*kwargs):
        if data.ndim == 2:
            out = HT2D(data,verbose=self.holo_kw["verbose"],returnSimple=True,subtract_mean=self.holo_kw["subtract_mean"],zero_pad=self.holo_kw["zero_pad"],**kwargs)  
            return out, np.zeros((np.shape(out)[0],)),np.zeros((np.shape(out)[0],))
        else:
            out,contrast_mat, out_int = PhaseReconstruction_Batch(data,batchSize=self.holo_kw["batchSize"],cr=self.holo_kw["cr"],trans=self.holo_kw["trans"],ref=self.holo_kw["ref"],onlyCPU=self.holo_kw["onlyCPU"], verbose=self.holo_kw["verbose"],zero_pad=self.holo_kw["zero_pad"],subtract_mean=self.holo_kw["subtract_mean"], returnContrastMat=True,**kwargs)
            return out, contrast_mat, out_int 
    
    def save(self,saveFolder,fileName=None,format=".mat"):
        """Save the processed data. 

        Parameters
        ----------
        saveFolder: folder path where to save the file 
        fileName: end without format name. 
        format: format to save the file. if fileName ends with .mat, .h5 or .hdf5, then format will be ignored and 
            file will be saved according to the format specified by filename. 
        
        """
        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder) 
        if fileName is not None:
            if fileName.endswith(".mat") or fileName.endswith(".h5") or fileName.endswith(".hdf5"):
                fileName0, format0 = fileName.split(".")
                format0 = "." + format0 
            else:
                fileName0 = fileName 
                format0 = format 
            self.saveFileName = fileName0 + format0
        else:
            format0 = format
            self.saveFileName = os.path.basename(self.meta["label"]) + str(QPImage._instance) + format0 
 
        if format0 == ".mat":
            mdict = {"data":self.out,"pha":self.pha,"ref":self.ref} #,"meta":self.meta}
            for keyk in self.meta.keys():
                mdict[keyk] = self.meta[keyk]
            savemat(os.path.join(saveFolder,self.saveFileName),mdict=mdict,appendmat=False)
        elif (format0 == ".h5") or (format0 == ".hdf5"):
            tmpFile = hp.File(os.path.join(saveFolder,self.saveFileName),"w")
            tmpFile.create_dataset("data",shape=self.out.shape,data=self.out) 
            tmpFile.create_dataset("pha",shape=self.pha.shape,data=self.pha)  
            tmpFile.create_dataset("contrast_mat",shape=self.contrast_mat.shape,data=self.contrast_mat)
            tmpFile.create_dataset("out_int",shape=self.out_int.shape,data=self.out_int) 
            for k in self.meta.keys():
                tmpFile.create_dataset(k,data=self.meta[k]) 
            tmpFile.close() 
        
        return 1 

    @property
    def amp(self):
        return np.abs(self.out)
    
    @property
    def intensity(self):
        return (np.abs(self.out))**2

    @property
    def field(self):
        return self.amp * np.exp(1j*self.pha) 

    @property
    def info(self):
        return self.meta 
    
    def compute_bg(self):
        if self.bg_kw["ret_mask"]:
            self.bg, self.mask =  compute_bkgnd(self.pha,fit_offset=self.bg_kw["fit_offset"], fit_profile=self.bg_kw["fit_profile"],border_m=self.bg_kw["border_m"], border_perc=self.bg_kw["border_perc"], border_px=self.bg_kw["border_px"],from_mask=self.bg_kw["from_mask"], ret_mask=self.bg_kw["ret_mask"]) 
            if self.subtract_ref:
                ref_out_bkgnd, _ = compute_bkgnd(self.ref_out,fit_offset=self.bg_kw["fit_offset"], fit_profile=self.bg_kw["fit_profile"],border_m=self.bg_kw["border_m"], border_perc=self.bg_kw["border_perc"], border_px=self.bg_kw["border_px"],from_mask=self.bg_kw["from_mask"], ret_mask=self.bg_kw["ret_mask"]) 
        else:
            self.bg = compute_bkgnd(self.pha,fit_offset=self.bg_kw["fit_offset"], fit_profile=self.bg_kw["fit_profile"],border_m=self.bg_kw["border_m"], border_perc=self.bg_kw["border_perc"], border_px=self.bg_kw["border_px"],from_mask=self.bg_kw["from_mask"], ret_mask=self.bg_kw["ret_mask"]) 
            if self.subtract_ref:
                ref_out_bkgnd = compute_bkgnd(self.ref_out,fit_offset=self.bg_kw["fit_offset"], fit_profile=self.bg_kw["fit_profile"],border_m=self.bg_kw["border_m"], border_perc=self.bg_kw["border_perc"], border_px=self.bg_kw["border_px"],from_mask=self.bg_kw["from_mask"], ret_mask=self.bg_kw["ret_mask"]) 
        self.pha = self.pha - self.bg 
        if self.subtract_ref:
            self.ref_out = self.ref_out - ref_out_bkgnd 
            del ref_out_bkgnd
        if self.subtract_ref:
            return self.pha, self.ref_out
        else:
            return self.pha 


