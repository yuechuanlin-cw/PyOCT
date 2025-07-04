"""
2D FFT based on PyCUDA
Fourier transform on GPU 
"""
import numpy as np 
import pycuda.autoinit 
import pycuda.gpuarray as gpuarray 
import skcuda.fft as cu_fft 
def fft2_gpu_c2c(x,fftshift=True):
    """
    C2C FFT
    This function produce an output that is compatible with numpy.fft.fft2.
    The input x is a 2D numpy array 
    """
    if x.dtype != np.complex128:
        x = x.astype(np.complex128)
    #get the shape of the initial numpy array 
    n1, n2 = x.shape 
    xgpu = gpuarray.to_gpu(x) 
    #Initialise empty output GPUarray 
    y = gpuarray.empty((n1,n2),np.complex128) 
    #FFT 
    plan_forward = cu_fft.Plan((n1,n2),np.complex128,np.complex128)
    cu_fft.fft(xgpu,y,plan_forward) 

    #Must divide by the total number of pixels in the image to get the normalization right 
    yout = y.get()/n1/n2
    if fftshift:
        yout = np.fft.fftshift(yout)
    return yout 
def fft2_gpu(x,fftshift=False): 
    """
    R2C FFT
    This function produce an output that is compatible with numpy.fft.fft2.
    The input x is a 2D numpy array 
    """
    #converting the input array to single precision float 
    if x.dtype != "float64":
        x = x.astype(np.float64)
    
    #get the shape of the initial numpy array 
    n1, n2 = x.shape 

    # from numpy array to GPUarray 
    xgpu = gpuarray.to_gpu(x)

    #initialize output GPUarray
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal 
    ysize = n2//2 + 1 
    y = gpuarray.empty((n1,ysize),np.complex128)

    #forward FFT 
    plan_forward = cu_fft.Plan((n1,n2),np.float64,np.complex128)

    cu_fft.fft(xgpu,y,plan_forward) 

    left = y.get() 

    #to make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version 
    # we must take care of handling even or odd sized array to get the correct size of the final array
    if n2//2 == n2/2: 
        #even 
        right = np.roll(np.fliplr(np.flipud(left))[:,1:-1],1,axis=0)
    else:
        #odd 
        right = np.roll(np.fliplr(np.flipud(left))[:,:-1],1,axis=0)
    print(right.shape)
    print(left.shape)
    #get a numpy array back to compatible with np.fft 
    if fftshift is False:
        yout = np.hstack((left,right))
    else:
        yout = np.fft.fftshift(np.hstack((left,right)))
    
    return yout

def ifft2_gpu(y,fftshift=False):
    """
    C2C iFFT
    do numpy.fft.ifft2 
    The input y is a 2D complex numpy array 
    """

    #get the shape of the initial numpy array 
    n1, n2 = y.shape 

    #from numpy array to GPUarray. Take the only first n2/2+1 non-redundant FFT coefficients when R2C.
    # For C2C, the dimensions of input and output are the same. 
    #if fftshift is False:
    #    y2 = np.asarray(y[:,0:n2//2+1],np.complex64)
    #else:
    #    y2 = np.asarray(np.fft.ifftshift(y)[:,0:n2//2+1],np.complex64) 
    if fftshift:
        y2 = np.fft.ifftshift(y) 
    else:
        y2 = y 
    ygpu = gpuarray.to_gpu(y2) 

    #Initialise empty output GPUarray 
    x = gpuarray.empty((n1,n2),np.complex128) 
    

    #inverse FFT 
    plan_backward = cu_fft.Plan((n1,n2),np.complex128,np.complex128)
    cu_fft.ifft(ygpu,x,plan_backward) 

    #Must divide by the total number of pixels in the image to get the normalization right 
    xout = x.get()/n1/n2

    return xout 


if __name__ == "__main__":
    from skimage import color, data 
    import matplotlib.pyplot as plt 
    from PyOCT import misc 
    Test = False
    if Test:
        import h5py as hp 
        import os 
        rootPath = "C:/Users/yuech/Dropbox (MIT)/laserSpeckleData/210920_LCI_system_test"
        fileName = "SampleScan5.mat"

        dataFile = hp.File(os.path.join(rootPath,fileName),"r")
        print("Member in dataset include: ")
        print(dataFile["data"].keys()) 
   
        img = np.asarray(dataFile["data"]["IMG"])[60,:,:] #column major data         
    else:
        im = data.coins()
        img = color.rgb2gray(im) 
    #img = img + 1j*np.zeros(np.shape(img),dtype=img.dtype)  


    print(img.shape)
    fft1 = np.fft.fftshift(np.fft.fft2(img))
    print("Numpy fft output size {}".format(np.shape(fft1)))
    fft2 = fft2_gpu(img,fftshift=True) 
    print("Numpy fft output size {}".format(np.shape(fft2)))
    fig, ax = misc.create_fig(figsize=[4,4],nrows=1,ncols=2)
    ax[0].imshow(np.log10(np.abs(fft1)),cmap="gray")
    ax[1].imshow(np.log10(np.abs(fft2))-np.log10(np.abs(fft1)),cmap="gray") #-np.log10(np.abs(fft1))
    ax[0].set_title("Numpy FFT")
    ax[1].set_title("diff GPU accelerated FFT")

    ifft1 = np.fft.ifft2(np.fft.ifftshift(fft1))
    ifft2 = ifft2_gpu(fft2,fftshift=True)  

    fig1, ax1 = misc.create_fig(figsize=[4,4],nrows=2,ncols=2)
    ax1[0,0].imshow(np.abs(ifft1),cmap="gray")
    ax1[0,1].imshow(np.abs(ifft2),cmap="gray")
    ax1[0,0].set_title("Numpy iFFT")
    ax1[0,1].set_title("diff GPU accelerated iFFT")

    ax1[1,0].imshow(np.angle(ifft1),cmap="gray")
    ax1[1,1].imshow(np.angle(ifft2),cmap="gray")
    ax1[1,0].set_title("Numpy iFFT")
    ax1[1,1].set_title("diff GPU accelerated iFFT")

    plt.show()