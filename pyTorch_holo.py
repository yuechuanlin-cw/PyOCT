from cv2 import CamShift
from matplotlib.colors import cnames
import numpy as np
from PyOCT import misc
from numpy.testing import verbose 
from scipy import signal
import matplotlib.pyplot as plt 
import h5py as hp 
from skimage import filters 
import torch 
name = "Pytorch/nGPU" 
device_type = "cuda"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("GPU Available? {}".format(torch.cuda.is_available())) 
#print("Current device {}".format(torch.cuda.current_device()))
#print("GPU running on {}".format(torch.cuda.get_device_name(0))) 
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def find_sideband(ft_data, which=+1, copy=True):
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

def torch_FFT2d(inputNDarray,dim = (-2,-1), shift=True,to_numpy=True,inputType="numpy"):
    if inputType == "numpy":
        inputNDarray = np.asarray(inputNDarray,dtype=np.complex64)
        inD = torch.from_numpy(inputNDarray) 
    else:
        inD = inputNDarray 
    if shift:
        tmp = torch.fft.fftshift(torch.fft.fft2(inD.to(device),dim=dim),dim=dim) 
    else:
        tmp = torch.fft.fft2(inD.to(device),dim=dim) 
    tmpCPU = tmp.cpu()
    del tmp 
    if to_numpy:
        return tmpCPU.numpy()
    else:
        return tmpCPU 

def torch_iFFT2d(inputNDarray,dim = (-2,-1), shift=True,to_numpy=True,inputType="numpy"):
    if inputType == "numpy":
        inputNDarray = np.asarray(inputNDarray,dtype=np.complex64)
        inD = torch.from_numpy(inputNDarray) 
    else:
        inD = inputNDarray 
    if shift:
        tmp = torch.fft.ifft2(torch.fft.ifftshift(inD.to(device),dim=dim),dim=dim) 
    else:
        tmp = torch.fft.ifft2(inD.to(device),dim=dim)     
    tmpCPU = tmp.cpu()
    del tmp 
    if to_numpy:
        return tmpCPU.numpy()
    else:
        return tmpCPU 

def torch_FFT3d(inputNDarray,zDim=0,shift=True,direction="f"):
    inputNDarray = np.asarray(inputNDarray,dtype=np.complex64)
    Z = np.shape(inputNDarray)[zDim]
    inD = torch.from_numpy(inputNDarray) 
    outD = torch.zeros(inD.shape,dtype=torch.complex64)  
    if direction == "f":
        for i in range(Z):
            outD[i,:,:] = torch_FFT2d(inD[i,:,:],shift=shift,to_numpy=False,inputType="tensor") 
    else:
        for i in range(Z):
            outD[i,:,:] = torch_iFFT2d(inD[i,:,:],shift=shift,to_numpy=False,inputType="tensor") 
    return outD.numpy() 


def torch_FFT3d_batch(inputNDarray,batchSize=50,zDim=0,shift=True,direction="f"):
    inputNDarray = np.asarray(inputNDarray,dtype=np.complex64)
    Z = np.shape(inputNDarray)[zDim]
    inD = torch.from_numpy(inputNDarray) 
    outD = torch.zeros(inD.shape,dtype=torch.complex64)  

    NumGroup = int(np.floor(Z/batchSize)) 
    xrange = np.arange(0,batchSize,step=1,dtype=int) 
    if direction == "f":
        for i in range(NumGroup):
            outD[i*batchSize+xrange,:,:] = torch_FFT2d(inD[i*batchSize+xrange,:,:],dim=(1,2),shift=shift,to_numpy=False,inputType="tensor") 
    else:
        for i in range(NumGroup):
            outD[i*batchSize+xrange,:,:] = torch_iFFT2d(inD[i*batchSize+xrange,:,:],dim=(1,2),shift=shift,to_numpy=False,inputType="tensor") 

    if (Z % batchSize):
        tmp_xrange = np.arange(0,Z % batchSize, step=1, dtype=int) 
        if direction == "f":
            outD[NumGroup*batchSize+tmp_xrange,:,:] = torch_FFT2d(inD[NumGroup*batchSize+tmp_xrange,:,:],dim=(1,2),shift=shift,to_numpy=False,inputType="tensor") 
        else:
            outD[NumGroup*batchSize+tmp_xrange,:,:] = torch_iFFT2d(inD[NumGroup*batchSize+tmp_xrange,:,:],dim=(1,2),shift=shift,to_numpy=False,inputType="tensor") 

    return outD.numpy() 


def mk_ellipse(XR,YR,X,Y):
    """
    make a elliptical shape for data filter
    Input:
    XR, YR: radius of ellipse
    X,Y: center of ellipse
    """
    XX, YY  = np.meshgrid(np.arange(0,X,step=1,dtype=int),np.arange(0,Y,step=1,dtype=int)) 
    return (((XX-X/2)/XR)**2 + ((YY-Y/2)/YR)**2)>1.0 


def HT2D(imgRaw,subtract_mean=True,returnSimple=False,zero_pad=True,**kwargs):
    """
    Hilbert transform for holography

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
    
    #Fimg = np.fft.fftshift(np.fft.fft2(img)) 
    Fimg = torch_FFT2d(img) 
    if not fixedKR:    
        fsx, fsy = find_sideband(Fimg,which=+1,copy=True) 
        mi = int(fsx * Fimg.shape[0])
        mj = int(fsy * Fimg.shape[1])        
            
        # coordinates in Fourier space
        fx = np.squeeze(np.fft.fftshift(np.fft.fftfreq(Fimg.shape[0])))
        fy = np.squeeze(np.fft.fftshift(np.fft.fftfreq(Fimg.shape[1]))) #
        [FXX,FYY] = np.meshgrid(fx**2,fy**2,indexing="ij")
        #TODO change the size of bandwidth to extract first order signal 
        dx = 0.091
        kkx = np.fft.fftshift(np.fft.fftfreq(Fimg.shape[0],d=dx))
        kky = np.fft.fftshift(np.fft.fftfreq(Fimg.shape[1],d=dx))

        dkx = np.abs(kkx[1] - kkx[0]) 
        dky = np.abs(kky[1] - kky[0]) 
        kres = 1.6*1.0/0.8 #1.0 NA, 0.8um wavelength 
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
    Fimg = np.roll(Fimg,[-mi,-mj],axis=(0,1))  

    Fimg = Fimg * cmask #1st order inteference signal 

    #Pimg = np.fft.ifft2(np.fft.ifftshift(Fimg))
    Pimg = torch_iFFT2d(Fimg) 
    #dc = np.abs(np.fft.ifft2(np.fft.ifftshift(dCF))) 
    dc = np.abs(torch_iFFT2d(dCF)) 
    contrast = 2 * np.sum(np.abs(Pimg)) / np.sum(np.abs(dc)) 
   # print("dc value is {}".format(np.sum(np.abs(dc)) ))
   # print("Contrast is {}".format(contrast))
    if returnSimple:
        return np.asarray(Pimg[:xSizeRaw,:ySizeRaw],dtype=np.complex64) 
    else:
        return np.asarray(Pimg[:xSizeRaw,:ySizeRaw],dtype=np.complex64) , contrast, mi, mj, cmask



def ima2(IMGRaw,subtract_mean=True,zero_pad=True,Sref = None, Sref_indx = None, **kwargs):
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
            IMGRaw[i,:,:] = IMGRaw[i,:,:] - np.single(np.mean(IMGRaw[i,:,:],dtype=np.float64))
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
        print("Ref provided! ")
    else:
        SubRef = False

    #tmp = IMG #- np.roll(IMG,[0,2,2],axis=(0,1,2)) #subtract the image by itself  
    #conttmp = np.squeeze(np.mean(IMG,axis=(1,2),dtype=np.float64)) 
    #Sref_indx = np.argmax(conttmp) 
    #Sref = np.squeeze(IMG[Sref_indx,:,:]) 
    if Sref is None: 
        Sref_indx = np.argmax(np.amax(IMG,axis=(1,2))) 
        Sref = np.squeeze(IMG[Sref_indx,:,:]) 
    #print(Sref_indx)

    if "cr" in kwargs.keys():
        cr = kwargs["cr"] 
        if cr > 1.0:
            print("Warning: cr forced to be 1.0!")
            cr = 1.0
    else:
        cr = 0.5 
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
            # coordinates in Fourier space
            fx = np.squeeze(np.fft.fftshift(np.fft.fftfreq(IMG.shape[1])))
            fy = np.squeeze(np.fft.fftshift(np.fft.fftfreq(IMG.shape[2]))) #
            [FXX,FYY] = np.meshgrid(fx**2,fy**2,indexing="ij")
            #TODO change the size of bandwidth to extract first order signal 
            dx = 0.091
            kkx = np.fft.fftshift(np.fft.fftfreq(IMG.shape[1],d=dx))
            kky = np.fft.fftshift(np.fft.fftfreq(IMG.shape[2],d=dx))
            dkx = np.abs(kkx[1] - kkx[0]) 
            dky = np.abs(kky[1] - kky[0]) 
            kres = 1.2*1.0/0.8 #1.0 NA, 0.8um wavelength 
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
        _, _, mi, mj, cmask = HT2D(Sref,zero_pad=False, subtract_mean=False)
    IMGk = torch_FFT3d(IMG,direction="f")  
    DCk = IMGk * np.repeat(cmask[np.newaxis,:,:], sz[0], axis=0)


    IMGk = np.roll(IMGk,(0,-mi,-mj),axis=(0,1,2)) 
    IMGk = IMGk *  np.repeat(cmask[np.newaxis,:,:], sz[0], axis=0) 
    
    out = torch_FFT3d(IMGk,direction="b") 

    DC = np.abs(torch_FFT3d(DCk,direction="b"))

    if SubRef:
        Pref = HT2D(ref,returnSimple=True,zero_pad=zero_pad)
        refPhase = np.conjugate(Pref) 
        refAmp = np.abs(refPhase) * np.exp(1j*np.zeros(np.shape(refPhase)))   
        for i in range(sz[0]):
            out[i,:,:] = out[i,:,:] * refPhase/refAmp  #np.exp(-1j * np.angle(Pref)) 
    else:
        Pref = np.zeros(np.shape(out),dtype=np.short)   

    if zero_pad:
        out_crop = out[:,:szRaw[1],:szRaw[2]]
        DC_crop = DC[:,:szRaw[1],:szRaw[2]]
    else:
        out_crop = out 
        DC_crop =DC 
    #calculate intensity distribution of an image in a smaller area around center 
    out_cut = out_crop[:,hh[0]:hh[-1],ww[0]:ww[-1]]
    DC_cut = DC_crop[:,hh[0]:hh[-1],ww[0]:ww[-1]]

    contrast_mat = np.single(np.squeeze(2*np.sum(np.abs(out_cut),axis=(1,2),dtype=np.float64)/np.sum(np.abs(DC_cut),axis=(1,2),dtype=np.float64)))
    out_inten = np.squeeze(np.single(np.mean(np.abs(out_cut),axis=(1,2),dtype=np.float64)))**2

   
    return np.asarray(out_crop,dtype=np.complex64), Pref[:sz[1],:sz[2]], out_inten, mi, mj, cmask, contrast_mat
 

def ShowRecontructedImg(dataRaw,out,mi_raw, mj_raw,trans=False):
    fig, ax = misc.create_fig(figsize=[10,10],nrows=2,ncols=2) 
    if trans:
        ax[0,0].imshow(np.transpose(dataRaw)**0.4,cmap="gray")
    else:
        ax[0,0].imshow((dataRaw)**0.4,cmap="gray")
    ax[0,0].set_title("raw data")

    if trans:
        ax[0,1].imshow((np.log10(np.abs(np.fft.fftshift(np.fft.fft2(dataRaw))))),cmap="gray") #without tranpose to match the x-NDarray to x-imshow. 
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
