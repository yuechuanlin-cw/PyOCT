# PF-OCE imaging reconstruction and processing 
import os 
import numpy as np 
from PyOCT import PyOCTRecon 
import matplotlib.pyplot as plt 
import matplotlib 
import re 
import scipy
import h5py 
from scipy import ndimage
# set font of plot 
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
font = {'weight': 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# define directory info
#root_dir = 'Z:/home/yl3248/data/Feb24_2020_Collagen/LuProtocol_P1' 

# import data 
#FullVolumDataFile = h5py.File(root_dir+'/TestDataSet/FullVolumOCTData_Complex.hdf5', 'r') 
#VolumeOCT = FullVolumDataFile['OCTData_real'][()] + 1j*FullVolumDataFile['OCTData_imag'][()] 
#FullVolumDataFile.close()

#import settings
#SettingsFile = h5py.File(root_dir+'/OCTProcessedData/Settings.hdf5','r')
#Settings = {}
#for keys in SettingsFile.keys():
#        #print(keys)
#        Settings[keys] = SettingsFile[keys][()]

def QuadraticFit2d(inData,Settings, verbose=False):
        """
        Quadratic 2D fit with coefficients as 
        a0 + a1*x + a2*y + a3*x**2 + a4*x**2*y + a5*x**2*y**2 + a6*y**2 +  a7*x*y**2 + a8*x*y
        Here only considering one single time point fit. 
        : inData: [Z,X,Y] data, complex or abs.  
        : Settings: settings dictionary 
        """
        [sizeZ,sizeX,sizeY] = np.shape(inData)
        x = np.arange(0,sizeX,step=1) 
        y = np.arange(0,sizeY,step=1)  
        surfData = np.squeeze(np.argmax(np.abs(inData),axis=0)) 
        X,Y = np.meshgrid(x,y,copy=False) 
        X = X.flatten()
        Y = Y.flatten()
        A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T #np.array([X*0+1, X, Y, X**2,  Y**2, X*Y]).T 
        B = surfData.flatten()
        coeff, r, rank, _ = np.linalg.lstsq(A, B,rcond=-1) # rcond = -1, using machine precision as to define zeros. rcond = None means that values smaller than machine precision * Max of A will be considered as zeros. 
        if verbose:
                print("QuadraticFit2d results:")
                print("coeff0: {}".format(coeff[0]))
                print("r : {}".format(r))
                print("rank: {}".format(rank))
        zxy = coeff[1]*X+coeff[2]*Y+coeff[3]*X**2 + coeff[4]*X**2*Y+coeff[5]*X**2*Y**2+coeff[6]*Y**2+coeff[7]*X*Y**2+coeff[8]*X*Y #coeff[1]*X+coeff[2]*Y+coeff[3]*(X**2) + coeff[4]*(Y**2)+coeff[5]*X*Y 
        zxy = np.reshape(zxy,(sizeX,sizeY))

        return coeff[0], zxy  


def SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150, verbose = False):
        from scipy.signal import find_peaks
        if inData.ndim == 3:
                tempData = np.squeeze(np.median(np.median(np.abs(inData),axis=1,keepdims=True),axis=2,keepdims=True))
        elif inData.ndim == 2: 
                tempData = np.squeeze(np.median(np.abs(inData),axis=1,keepdims=True))
        elif inData.ndim == 1:
                tempData = np.abs(inData)
        else:
                raise ValueError("Wrong input data for SearchingCoverGlass !")
        # Method I: based on local maximum 
        tempData = ndimage.median_filter(tempData,size=3) 
        mid_index = int((start_index+end_index)/2)
        peaks = np.zeros((2,),dtype=np.int)
        peaks[0] = int(np.argmax(tempData[start_index:mid_index]) + start_index)
        peaks[1] = int(np.argmax(tempData[mid_index:end_index])+mid_index) 
        if verbose:
                print("Glass position is {} and {} pixel".format(peaks[0],peaks[1])) 
                figSG = plt.figure(figsize=(5,4))
                figSG.suptitle("Positions of Coverslip")
                axSG = plt.subplot2grid((1,1),(0,0))
                axSG.plot(tempData) 
                axSG.scatter(peaks[0],tempData[peaks[0]],s=80,facecolors='none', edgecolors='tab:red',linewidths=2,linestyles='dotted')
                axSG.scatter(peaks[1],tempData[peaks[1]],s=80,facecolors='none', edgecolors='tab:red',linewidths=2,linestyles='dotted')
                axSG.set_yscale('log') 
        Settings['CoverSlip_Positions'] = peaks 
        Settings["RefGlassPos"] = peaks[1] 
        return peaks, Settings

def PhaseRegistration(inData, Settings, dzc = 5,start_index = 5, end_index = 150,verbose=False):
        """
        Phase registration w.r.t coverslip phase as mitigation of phase instability 
        Pre-assumption: coverslip is flat and level without any strong scatters around
        : inData: raw complex OCT reconstructed signal, input OCT(x,y,z) data after normal imaging reconstruction.
        : Settings: dictionary of OCT settings 
        : dzc: number of pixels over which the coverglass phase will be averaged based on OCT-intensity weights
        : start_index, end_index: range of index to search position of coverslip which should cover both two flat surface of coverslip
        """
        if verbose:
                print("Phase Registration ...")
        if "RefGlassPos" not in Settings.keys():  # if not, means not coherence gate curvature removal implemented. Then we just search the position of surf glass        
                peaks, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150,verbose=verbose)

        pos_coverglass = Settings["RefGlassPos"]
        surface_phi = np.average(inData[pos_coverglass-dzc:pos_coverglass+dzc,:,:],axis=0,weights=np.abs(inData[pos_coverglass-dzc:pos_coverglass+dzc,:,:]))
        surface_phi_2d = np.exp(-1j*np.angle(surface_phi)) #np.conj(surface_phi)
        S = np.multiply(inData,np.repeat(surface_phi_2d[np.newaxis,:,:],np.shape(inData)[0],axis=0))
        return S, Settings    

def ObtainGridMesh(inData,Settings):
        """Find grid mesh coordinates corrsponding to spatial and frequency domain as the same dimension of input data. 
        return x,y,qx,qy in Settings. 
        return:
        inData, Settings. 
        where Settings include:
        x,y: x and y coordinates corresponding to two axis in enface of OCT image. 
        qx,qy: frequency coordinates with center pixles as 0. 
        xm,ym: 2D x,y cooridnates as same dimension of enface in OCT image. 
        """
        [Z,X,Y] = np.shape(inData) 
        dx = Settings['xPixSize']
        dy = Settings['yPixSize']
        x = (2*np.pi)*(1/X)*np.arange(0,X,1)*dx 
        y = (2*np.pi)*(1/Y)*np.arange(0,Y,1)*dy
        Settings['x'] = x
        Settings['y'] = y # in unit of um 

        # setting qx and qy array for future use 
        qx = 2*np.pi/(Settings['xPixSize']*1e-6)*(1/X)*np.arange(0,X,1) 
        qx = qx - qx[int(np.floor((X-1)/2))] 

        qy = 2*np.pi/(Settings['yPixSize']*1e-6)*(1/Y)*np.arange(0,Y,1)
        qy = qy - qy[int(np.floor((X-1)/2))] 
        Settings['qx'] = qx 
        Settings['qy'] = qy # in unit of 1/meter. 
        qxm,qym = np.meshgrid(qx,qy)
        qxm = np.transpose(qxm)
        qym = np.transpose(qym) 
        Settings['qxm'] = qxm 
        Settings['qym'] = qym 
        [xm,ym] = np.asarray(np.meshgrid(x,y))
        Settings['xm'] = np.transpose(xm)
        Settings['ym'] = np.transpose(ym)
        return Settings 


## S3: Defocus 
def SearchingFocalPlane(inData,Settings,start_bias = 50, extend_num_pixels = 240,start_index = 5, end_index = 150,showFitResults=False,smethod="per"):
        """
        Search focal plane by fitting Guassian profile 
        Initially it will search from the position start_bias pixels from coverslip and extend extend_num_pixels pixels
        The pixels used to locate focal plane is: [CoverSlip_Positions + start_bias: CoverSlip_Positions + start_bias+extend_num_pixels]
        : start_bias: int, number of pixels the starting position away from covergalss slip position 
        : extend_num_pixels:int, number of pixels extended from starting position 
        Return:
        : zf:  position of focal plane in um as OPL, w.r.t zero path. 
        : Settings
        """
        from scipy import ndimage  
        from lmfit import Model
        [Z,X,Y] = np.shape(inData)
        #if 'CoverSlip_Positions' not in Settings.keys():
        #        _, Settings = SearchingCoverGlass(inData,Settings,start_index = 5, end_index = 150)
        #peaks = Settings['CoverSlip_Positions']
        #pos_cover = peaks[1] 
        if "RefGlassPos" not in Settings.keys():  # if not, means not coherence gate curvature removal implemented. Then we just search the position of surf glass        
                _, Settings = SearchingCoverGlass(inData,Settings,start_index = start_index, end_index = end_index)
  
        pos_cover = Settings["RefGlassPos"]
        # method II using quadratic fit
        if smethod.lower()=='2dfit':
                zf, _  = QuadraticFit2d((inData[(pos_cover+start_bias):(pos_cover+start_bias+extend_num_pixels),:,:])**0.4,Settings,verbose=showFitResults)  
                zf = int(zf + pos_cover+ start_bias) # now this will serve as the reference position of cover glass 
        elif smethod.lower()=='max':
                zf = int(np.argmax(np.squeeze(np.amax(np.amax(np.abs(inData[(pos_cover+start_bias):(pos_cover+start_bias+extend_num_pixels),:,:]),axis=1,keepdims=True),axis=2,keepdims=2))))
                zf = int(zf + pos_cover+ start_bias) 
        elif smethod.lower() == 'per':
                tmp = np.sort(np.reshape(np.abs(inData),(Z,X*Y)),axis=1) #np.sort(np.abs(inData.flatten()))
                sizeTmp = X*Y
                zprofile = np.squeeze(np.median(tmp[(pos_cover+start_bias):(pos_cover+start_bias+extend_num_pixels),int(sizeTmp*0.996):int(sizeTmp*0.999)],axis=1)) 
                zf = int(np.argmax(zprofile))
                zf = int(zf + pos_cover+ start_bias) 
        if showFitResults:
                print("Focal position found at {} pixel".format(zf)) 
        res_zf = zf # in pixel 
        zf = zf*Settings['zPixSize'] #in um         
        Settings['zf'] = zf # in um
        return res_zf, Settings


def ViolentDefocus(inData,Settings,showFitResults=False,proctype='oce',verbose=False,start_bias = 50, extend_num_pixels = 240):
        if verbose:
                print("Defocusing...")
        # searching focal plane, regardless zf is known or not  
        if 'zf' not in Settings.keys():
                zfpos,Settings = SearchingFocalPlane(inData, Settings,showFitResults=showFitResults,start_bias = start_bias, extend_num_pixels = extend_num_pixels) 
        #zfpos = int(Settings['zf']/Settings['zPixSize'])
        zf = Settings['zf']
        if verbose:
                print("Defocus will happen at {} pixel".format(zfpos))

        [Z,X,Y] = np.shape(inData)
        k_m = Settings['k'] * 1e9 # here k is in the unit of 1/nm, convert to 1/m
        N = np.size(k_m) 
        kc = np. mean(k_m) 
        dz = Settings['zPixSize']*1e-6 
        if 'qxm' not in Settings.keys(): 
                Settings = ObtainGridMesh(inData,Settings)
        qxm = Settings['qxm']
        if proctype.lower() == 'oce':
                qym = Settings['qym']*0 
        elif proctype.lower() == 'oct':
                qym = Settings['qym']
        qr = np.fft.ifftshift(np.sqrt(qxm**2+qym**2)) #optimize out fftshift

        #aperture to prevent imaginary #'s as low pass filter 
        aperture = (qr <= 2*Settings['refractive_index']*kc)
        #aperture = np.ones(np.shape(qr))
        #defocus kernel 
        phase = np.sqrt(np.multiply(aperture,(2*Settings['refractive_index']*kc)**2-qr**2)) 


        #PERFORM CAO
        output = inData
        for i in range(np.shape(inData)[0]):
                planeFD = np.fft.fft2(np.squeeze(inData[i,:,:]))
                correction = np.multiply(aperture,np.exp(-1j*(dz*i-zf*1e-6)*phase))
                plane = np.fft.ifft2(np.multiply(planeFD,correction))
                output[i,:,:] = plane 

        return output, Settings


def CoherenceGateRemove(inData,Settings,proctype='OCE',verbose=False,start_index = 5, end_index = 150):
        """
        Coherence gate curvature removal
        zxy0: fitted estimated tilt/curvature from first time-lapse volume 
        zc0: reference glass position in units of pixels 
        """
        if 'RefGlassPos' not in Settings.keys():
                _, Settings = SearchingCoverGlass(inData,Settings,start_index = start_index, end_index = end_index)
        pos_cover = Settings['RefGlassPos']

        if 'CoherenceGateCoeffMat' not in Settings.keys(): 
                #_, Settings = SearchingCoverGlass(inData,Settings,start_index = start_index, end_index = end_index)
                #pos_cover = Settings['RefGlassPos']
                zc0, zxy0 = QuadraticFit2d(inData[pos_cover-20:pos_cover+20,:,:],Settings,verbose=verbose)  
                zc0 = int(zc0 + pos_cover-20) # now this will serve as the reference position of cover glass 
                Settings['RefGlassPosCoeff'] = zc0 
                Settings['CoherenceGateCoeffMat'] = zxy0 
                zct = zc0 
        else:
                zc0 = Settings['RefGlassPosCoeff']
                zxy0 = Settings['CoherenceGateCoeffMat']
               # pos_cover, _ = SearchingCoverGlass(inData,Settings,start_index = start_index, end_index = end_index)
                zct, _ = QuadraticFit2d(inData[pos_cover-20:pos_cover+20,:,:],Settings,verbose=verbose)  
                zct = int(zct + pos_cover-20) # now this will serve as the reference position of cover glass  
        if verbose:
                print("Reference Coverglass position now will be always at {} pixel".format(zc0)) 
        
        #using the same coefficient from first image at a time lapsed images to make sure consistent correction  

        [Z,X,Y] = np.shape(inData)
        k_m = Settings['k'] * 1e9 # here k is in the unit of 1/nm, convert to 1/m
        N = np.size(k_m) 
        kc = np. mean(k_m) 
        dz = Settings['zPixSize']*1e-6 
        if 'qxm' not in Settings.keys(): 
                Settings = ObtainGridMesh(inData,Settings)
        qxm = Settings['qxm']
        if proctype.lower() == 'oce':
                qym = Settings['qym']*0 
        elif proctype.lower() == 'oct':
                qym = Settings['qym']
        qr = np.fft.ifftshift(np.sqrt(qxm**2+qym**2)) #optimize out fftshift
        aperture = (qr <= 2*Settings['refractive_index']*kc)
        #aperture = np.ones(np.shape(qr))
        #defocus kernel         
        qz = np.sqrt(np.multiply(aperture,(2*Settings['refractive_index']*kc)**2-qr**2)) #np.sqrt((2*Settings['refractive_index']*kc)**2-qr**2)
        inData = np.fft.fft(inData,axis=0) 
        for i in range(np.shape(inData)[0]):
                inData[i,:,:] = inData[i,:,:] * np.exp(1j*qz*(zxy0+(zct-zc0)))         
        inData = np.fft.ifft(inData,axis=0) 
        _, Settings = SearchingCoverGlass(inData,Settings,start_index = start_index, end_index = end_index)
        return inData, Settings

def FocalPlaneRegistration(inData,Settings,proctype='OCT',showFitResults=False,start_bias = 50, extend_num_pixels = 240,verbose=False):
        if 'zf' not in Settings.keys():
                _,Settings = SearchingFocalPlane(inData, Settings,showFitResults=showFitResults,start_bias = start_bias, extend_num_pixels = extend_num_pixels) 
        zfpos = int(Settings['zf']/Settings['zPixSize'])
        if 'FocalPlaneCorrMat' not in Settings.keys():
                _, zf0 = QuadraticFit2d(inData[zfpos-40:zfpos+40,:,:],Settings,verbose=verbose)  
                Settings['FocalPlaneCorrMat'] = zf0
        else:
               # b0 = Settings['b0']
                zf0 = Settings['FocalPlaneCorrMat']  
        [Z,X,Y] = np.shape(inData)
        k_m = Settings['k'] * 1e9 # here k is in the unit of 1/nm, convert to 1/m
        N = np.size(k_m) 
        kc = np. mean(k_m) 
        dz = Settings['zPixSize']*1e-6 
        if 'qxm' not in Settings.keys(): 
                Settings = ObtainGridMesh(inData,Settings)
        qxm = Settings['qxm']
        if proctype.lower() == 'oce':
                qym = Settings['qym']*0 
        elif proctype.lower() == 'oct':
                qym = Settings['qym']
        qr = np.fft.ifftshift(np.sqrt(qxm**2+qym**2)) #optimize out fftshift
        aperture = (qr <= 2*Settings['refractive_index']*kc)
        #aperture = np.ones(np.shape(qr))
        #defocus kernel         
        qz = np.sqrt(np.multiply(aperture,(2*Settings['refractive_index']*kc)**2-qr**2)) #np.sqrt((2*Settings['refractive_index']*kc)**2-qr**2)
        inData = np.fft.fft(inData,axis=0) 
        for i in range(np.shape(inData)[0]):
                inData[i,:,:] = inData[i,:,:] * np.exp(1j*qz*zf0)   
        inData = np.fft.ifft(inData,axis=0)      
        _,Settings = SearchingFocalPlane(inData, Settings,showFitResults=showFitResults,start_bias = start_bias, extend_num_pixels = extend_num_pixels) 
        return inData, Settings



## S2: Bulk demodulation
def Gauss(x, amp, cen, wid, bis):
        return bis  + amp * np.exp(-(x-cen)**2 /wid)

def Gauss2( x, amp1, cen1, wid1, amp2, cen2, wid2):
    #(c1, mu1, sigma1, c2, mu2, sigma2) = params
    return amp1 * np.exp( - (x - cen1)**2.0 / (2.0 * wid1**2.0) ) + amp2 * np.exp( - (x - cen2)**2.0 / (2.0 * wid2**2.0) )

def BulkDemodulation(inData, Settings,showFitResults=False,proctype='oct',excludeGlass=True):
        """
        inData must be a full volume OCT image 
        : best practice is to exclude the glass regions. The all-bright of glass/coverslip will distort the results. 
        """
        if excludeGlass:
                pos_cover = Settings["RefGlassPos"] 
                tmp = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(inData[(pos_cover+10):-1,:,:]))))
        else:
                tmp = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(inData))))

        lineX = np.squeeze(np.sum(tmp,axis=(0,2))) 
        lineY = np.squeeze(np.sum(tmp,axis=(0,1))) 
        lineX = ndimage.median_filter(lineX,size=3) 
        lineY = ndimage.median_filter(lineY,size=3) 
        Xpeak_init = np.argmax(lineX) 
        Ypeak_init = np.argmax(lineY) 
        lineX = np.reshape(lineX,(np.size(lineX),1))
        lineY = np.reshape(lineY,(np.size(lineY),1)) #N*1 
        # find peak of FFT magnitude profile along X and Y 
        bound = 100 
        basis = np.arange(-100,100,step=1,dtype=int) 
        basis = np.reshape(basis,(np.size(basis),1)) 
        fitter = np.linalg.pinv(np.concatenate((basis**0,basis**1,basis**2),axis=1)) 
        xfit = np.squeeze(np.matmul(fitter,lineX[Xpeak_init+np.squeeze(basis),:]))
        yfit = np.squeeze(np.matmul(fitter,lineY[Ypeak_init+np.squeeze(basis),:])) 
        Xpeak = Xpeak_init - (0.5*xfit[1]/xfit[2])
        Ypeak = Ypeak_init - (0.5*yfit[1]/yfit[2]) 
        
        if showFitResults:
                fig, ax = plt.subplots(nrows=2,ncols=2)
                ax[0,0].imshow(np.amax(tmp,axis=0)**0.4)
                ax[0,1].plot(basis,lineX[Xpeak_init+np.squeeze(basis),:],color="red") 
                ax[1,0].plot(basis,lineY[Ypeak_init+np.squeeze(basis),:],color="red")
                #ax[0,1].plot(basis,xfit,"g--") 
                #ax[1,0].plot(basis,yfit,"g--") 
               # ax[0,1].scatter(Xpeak,np.amax(xfit),s=50)
               # ax[1,0].scatter(Ypeak,np.amax(yfit),s=50) 
                print("Xpeak is {}".format(Xpeak))
                print("Ypeak is {}".format(Ypeak))

        # demodulation 
        if 'xm' not in Settings.keys():
                Settings = ObtainGridMesh(inData,Settings) 
        xm = Settings['xm']/Settings['xPixSize']
        ym = Settings['ym']/Settings['yPixSize'] 
        [Z,X,Y] = np.shape(inData) 
        if proctype.lower() == 'oct':
                Xshift = Xpeak - (np.floor(X/2)+1)
                Yshift = Ypeak - (np.floor(Y/2)+1) 
        elif proctype.lower() == 'oce':
                Xshift = Xpeak - (np.floor(X/2)+1)
                Yshift = 0 
        else:
                raise ValueError("proctype can only be either oct or oce!")
        phase = 2*np.pi*(Xshift*xm + Yshift*ym)
        phase = phase - np.mean(phase) 
        demodulator = np.exp(-1j*phase) 
        for i in range(Z):
                inData[i,:,:] = inData[i,:,:] * demodulator       

        return inData, Settings

def FullCAO(inData,Settings,verbose=False,proctype='oct',start_index=5,end_index=200,start_bias=100,extend_num_pixels=150,singlePrecision=False,excludeGlass=True):
        """
        Conduct full imaging reconstruction processing with: coherence gate curvature removal, focal plane registartion, phase registration, bulk demodulation and computation adapative optics
        inData: basic reconstructed OCT image data. complex and as dim [z,x,y]
        Settings: parameters 
        start_indx, end_index: position ranges to search for cover glass positions along z direction 
        start_bias, extend_num_pixels: start_bias+ref_glass_pos:start_bias+ref_glass_pos+extend_num_pixels as the range to serach focal plane 
        excludeGlass=True: exclude galss regions for bulk modulation calculation; strong signal over FOV from glass regions might distort the bulk demodulation. 
        return:
        inData, Settings. 
        """
        inData, Settings = CoherenceGateRemove(inData,Settings,verbose=verbose,proctype=proctype,start_index = start_index, end_index = end_index)
        inData, Settings = FocalPlaneRegistration(inData,Settings,proctype=proctype)
        inData, Settings = PhaseRegistration(inData,Settings)
        inData, Settings = BulkDemodulation(inData,Settings,showFitResults=verbose) 
        inData, Settings = ViolentDefocus(inData,Settings,showFitResults=verbose,proctype=proctype,start_bias = start_bias, extend_num_pixels = extend_num_pixels)
        if singlePrecision:
                inData = np.abs(inData)
                inData = inData.astype(np.float32) 

        return inData, Settings
