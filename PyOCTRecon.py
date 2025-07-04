"""
Spectral-domain Optical Coherence Tomography Imaging Reconstruction
: Author: Yuechuan Lin 
: Organization: Cornell University 
: Created at: April 09, 2020 
"""
import os 
import numpy as np 
import xml.etree.ElementTree as ET
import time
from scipy.linalg import dft
import numpy.matlib 
import matplotlib.pyplot as plt 
import matplotlib 
from PyOCT import CAO 
from progress.bar import Bar

class OCTImagingProcessing():
    """OCT Imaging Processing
        All data file, including background file, must be under the same root directory.
    :: root_dir: required, root directory path where OCT raw data located, ENDING WITHOUT /; e.g.,  root_dir = 'D:/cuda_practice/OCT_Recon/OCTdata'.
    :: sampleID: required, sample data file name. ENDING WITHOUT _raw.bin; e.g., sampleID = 'OCT_100mV_2'. 
    :: bkgndID: required, background data file name, ending with _raw.bin; e.g., bkgndID = 'bkgnd_512_0_raw.bin'.
    :: Sample_sub_path: optional, default as None; sub-directory where OCT raw data located. ENDING WITHOUT /. 
    :: Bkgnd_sub_path: optional, default as None; sub-directory where OCT bkgnd data located. ENDING WITHOUT /.
    :: saveOption: optional, bool, default as False. 
    :: saveFolder: optional, name for folder to save data; default as None, which will save data in root directory.
    :: RorC: optional,"real" or "complex", tell to save data or show data in complex format or single precison (float32) format. 
    :: verbose: optional, bool, default as True. If True, the data processing will show processing information during each step. 
    :: frames: optional, int, number of frames to read and process, defaults as 1.
    :: alpha2, alpha3: optional, parameters for computational dispersion correction. 
    :: depths: optional, nuumpy.linspace() created array, depths to be processed, default as: np.linspace(1024,2047,1024), indicating procesing 1024th z-pixel to 2048-pixel.
    :: gamma: optional, power factor to do plotting, default as 0.4.
    :: wavelength: optional, nominal central wavelength of OCT laser source in unit of nm, default as 1300. 
    :: XYConversion: optional, 2 elements numpy array, calibration factor for galvo-scanning voltage to scanning field of view in x and y axis at unit of um/V, default as [660,660].
    :: camera_matrix: optional, camera dispersion correction factor, numpy array as [c0,c1,c2,c3]; default as np.asarray([1.169980E3,1.310930E-1,-3.823410E-6,-7.178150E-10]) for 1300nm system.
    :: start_frame: which frame to start reading and being processed. default is 1, indicating starting from first frame. 
    :: singlePrecision: bool, if True, the processed OCT intensity data will be convert to single precision (np.float32). Only applicable when RorC is 'real'. 
    :: OptimizingDC: [Required Further Developement] optional, bool, optimizing dispersion correction to search optimized alpha2 and alpha3. default as False. 
    :: ReconstructionMethods: OCT reconstruction with CAO or without CAO. By default is 'NoCAO'.Could Choose 'CAO' to do imaging reconstruction.
    """
    def __init__(self,root_dir, SampleData = None, Settings = None, sampleID=None,bkgndID = None,Sample_sub_path = None,Bkgnd_sub_path = None,saveOption=False,saveFolder=None,RorC='real',verbose=True, frames=1,alpha2=None,alpha3=None,depths=np.linspace(1024,2047,1024),gamma = 0.4,wavelegnth = 1300,XYConversion=[660,660],camera_matrix=np.asarray([1.169980E3,1.310930E-1,-3.823410E-6,-7.178150E-10]),start_frame = 1, singlePrecision = True, OptimizingDC=False,ReconstructionMethods='NoCAO'):
        _time1 = time.time() 
        self.ReconstructionMethods = ReconstructionMethods
        self.saveOption = saveOption
        self.saveFolder = saveFolder 
        self.RorC = RorC
        self.verbose = verbose
        self.start_frame = start_frame - 1 # converting first frame to 0 as definition of Python style. 
        self.singlePrecision = singlePrecision
        
        if self.verbose:
            print("*******************************************")
            if not self.saveOption:
                print('Warning: The Data NOT saved !!') 
            print("Starting OCT Imaging Reconstruction ....")
            print("Configuring All Settings: ")
        self.root_dir = root_dir

        if Settings is None:
            self.sampleID = sampleID # sampleID should pass without _raw.bin part
            self.bkgndID = bkgndID 
            self.Settings = {}
            if Sample_sub_path:
                self.SamplePath = root_dir + '/' + Sample_sub_path + '/'
            else:
                self.SamplePath = root_dir + '/'

            self.Config()   #construct Settings
        else:
            self.Settings = Settings 

        if alpha2:
            self.Settings['alpha2'] = alpha2 
        if alpha3:
            self.Settings['alpha3'] = alpha3 
        self.Settings['depths']  = depths.astype(int) 
 
        self.Settings['Frames'] = int(frames)
        self.Settings['wavelength'] = wavelegnth
        self.XYConversion = np.asarray(XYConversion)
        self.Settings['XConversion'] = self.XYConversion[0] # micro-meter/Voltage
        self.Settings['YConversion'] = self.XYConversion[1]
        self.Settings['c0'] = camera_matrix[0]
        self.Settings['c1'] = camera_matrix[1]
        self.Settings['c2'] = camera_matrix[2]
        self.Settings['c3'] = camera_matrix[3]
        self.Settings['gamma'] = gamma 
        self.Settings['refractive_index'] = 1.3496 
        val_p = np.linspace(1,self.Settings['NumCameraPix'],self.Settings['NumCameraPix']) 
        val_tmp = self.Settings['c0']*(val_p**0) + self.Settings['c1']*(val_p**1)+self.Settings['c2']*(val_p**2)+self.Settings['c3']*(val_p**3)
        self.Settings['k'] = 2*np.pi/val_tmp
        val_ks = self.Settings['NumCameraPix']/(self.Settings['NumCameraPix']-1)*np.abs(self.Settings['k'][0]-self.Settings['k'][-1]) # N*dk
        self.Settings['lambda'] = np.mean(np.linspace(val_tmp[0],val_tmp[-1],self.Settings['NumCameraPix'])) # central wavelength of OCT beam ]
        self.Settings['zPixSize'] = (np.pi/(1000*val_ks))/self.Settings['refractive_index'] 
        self.Settings['xPixSize'] = self.Settings['XConversion'] * self.Settings['GalvoXV']/self.Settings['Ascans'] 
        self.Settings['yPixSize'] = self.Settings['YConversion'] * self.Settings['GalvoYVmax']/self.Settings['Frames']
        self.Settings['XFOV'] = self.Settings['XConversion'] * self.Settings['GalvoXV']
        self.Settings['YFOV'] = self.Settings['YConversion'] * self.Settings['GalvoYVmax']
        
        if self.verbose:
            print("\n///////////////////////////////////////////////////\n")
            print("Settings List is:")
            for key in self.Settings.keys():
                if key == 'depths':
                    print("{}: {}, {}".format("depths_min_max", np.amin(self.Settings[key]),np.amax(self.Settings[key])))
                else: 
                    print("{}: {}".format(key, self.Settings[key]))
            print("End of Configuration")
            print("\n//////////////////////////////////////////////////\n")

        """Load data
        """
        Z = self.Settings['NumCameraPix']
        X = self.Settings['Ascans']
        Y = self.Settings['Frames']
        numVoxels = X*Y*Z
        if SampleData is None:
            self.SampleData = self.readData(self.SamplePath+self.sampleID+"_raw.bin",self.Settings,"data",self.start_frame,numVoxels) 
        else:
            self.SampleData = SampleData
        """Normal OCT Reconstruction 
        """
        if self.ReconstructionMethods.lower() == "nocao":
            if self.bkgndID:
                self.bkgndID = bkgndID 
            else:
                raise ValueError("Background ID required !")
            if Bkgnd_sub_path:
                self.BkgndPath = root_dir + '/' + Bkgnd_sub_path + '/'
            else:
                self.BkgndPath = root_dir + '/' 
            self.BkgndData = self.readData(self.BkgndPath+bkgndID,self.Settings,"background",0,numVoxels) 
            self.BkgndData = np.mean(self.BkgndData,axis=1)
            self.SampleData = np.apply_along_axis(self.bkgndSubtraction,0,self.SampleData) # in dim as [Z,X*Y] 
            
            """Reconstruction matrix
            """
            F = np.fft.fftshift(np.conj(dft(self.Settings['NumCameraPix'])))
            A = np.eye(self.Settings["NumCameraPix"]) 
            beta = self.Resample()   
            alpha = self.AlphaMatrix() 
            FDepth = F[self.Settings['depths'],:]
            self.SampleSpectrum = np.squeeze(np.median(np.abs(beta.dot(self.SampleData)),axis=1))
            self.Process = FDepth.dot(alpha.dot(beta.dot(A))) 

            """Final Step Reconstructing OCT Imaging
            """        
            self.data = self.Process.dot(self.SampleData) #np.reshape(self.Process.dot(np.reshape(self.SampleData,(Z,X*Y))),(np.size(self.Settings['depths']),X,Y))

        elif self.ReconstructionMethods.lower() == "cao":
            # backgrond subtraction happens after resampling 
            # # resampling 
            beta = self.Resample()  
            self.SampleData = beta.dot(self.SampleData) 
            self.SampleData = np.reshape(self.SampleData,[Z,X,Y])
            self.SampleSpectrum = np.squeeze(np.median(np.abs(self.SampleData),axis=(1,2)))
            self.SampleDataTmp = np.fft.ifft(self.SampleData,axis=0) #np.reshape(np.fft.ifft(self.SampleData,axis=0),[Z,X,Y]) # to Z-domain
            # here SampleData in k-domain while SampleDataTmp in z-domain 
            # background estimation and subtraction 
            #self.SampleDataTmp = np.reshape(self.SampleDataTmp,[Z,X,Y]) 
            #self.BkgndDataxl = np.repeat(np.median(self.SampleData,axis=1,keepdims=True),X,axis=1) #[Z,1,Y] back to [Z,X,Y]
            self.BkgndDataxl = np.repeat(np.repeat(np.median(np.median(np.real(self.SampleDataTmp),axis=1,keepdims=True),axis=2,keepdims=True),X,axis=1),Y,axis=2) + 1j * np.repeat(np.repeat(np.median(np.median(np.imag(self.SampleDataTmp),axis=1,keepdims=True),axis=2,keepdims=True),X,axis=1),Y,axis=2)
            self.BkgndDataxl = np.fft.fft(self.BkgndDataxl,axis=0) 
            self.fx = np.repeat(np.mean(np.multiply(self.BkgndDataxl,self.SampleData),axis=0,keepdims=True)/np.mean(self.BkgndDataxl**2,axis=0,keepdims=True),Z,axis=0)
            self.BkgndDataxl = np.multiply(self.BkgndDataxl,self.fx)
            self.SampleData = np.reshape(self.SampleData - self.BkgndDataxl, [Z,X*Y])
            self.BkgndData = np.squeeze(np.median(np.abs(self.BkgndDataxl),axis=(1,2)))
            #del self.BkgndDataxl 

            #construction matrix
            F = np.fft.fftshift(np.conj(dft(self.Settings['NumCameraPix'])))
            A = np.eye(self.Settings["NumCameraPix"])      
            alpha = self.AlphaMatrix() 
            FDepth = F[self.Settings['depths'],:]
            self.Process = FDepth.dot(alpha.dot(A))   
            self.data = self.Process.dot(self.SampleData) 

            # TEST PURPOSE 
            #self.BkgndDataxl = np.reshape(self.BkgndDataxl,[Z,X*Y]) 
            #self.BkgndDataxl = self.Process.dot(self.BkgndDataxl)
            #self.BkgndDataxl = np.abs(np.reshape(self.BkgndDataxl,[np.size(self.Settings['depths']),X,Y]))
            del self.SampleDataTmp 
            del self.fx 
            del self.BkgndDataxl

        self.OCTData =  np.abs(self.data)
        if self.singlePrecision:
            self.OCTData = self.OCTData.astype(np.float32) # convert to single precision 
        #self.data = self.data.astype(np.complex64)

        if self.RorC.lower() == "real":
            del self.data

        """Some other possible processed data
        """   
        self.InterferencePattern = np.squeeze(np.mean(self.SampleData,axis=1,keepdims=True))
        self.DepthProfile = np.squeeze(np.mean(self.OCTData,axis=1,keepdims=True))#,axis=2,keepdims=True)) #np.squeeze(np.mean(np.mean(self.OCTData,axis=1,keepdims=True),axis=2,keepdims=True))

        """Save Data 
        """
        if self.saveOption:
            if self.verbose:
                print("\n*************************************")
                print("Starting Saving Data ...")
            self.saveData()  
        
        """ Test Print        
        print(np.shape(F))
        print(np.shape(FDepth))
        print(np.shape(alpha))
        print(np.shape(beta))
        print(np.shape(A))
        """

        """ End part
        """
        _time2 = time.time() 
        if self.verbose:
            print("The total reconstruction time is : {0:.6f} sec".format(_time2-_time1))
            print("End Of OCT Imaging Reconstruction!")

    def Config(self):
        self.XMLvariableConversion = np.asarray([['Image Number', 'img_number'],
        ['f_wait (1/s)' ,'fwait'],
        ['Camera Rate (Hz)' ,'CameraRate'],
        ['Galvo Rate (Hz)', 'GalvoRate'],
        ['Points in each waveform' ,'PointsPerWaveform'],
        ['Number of Buffers' ,'NumBuffers'],
        ['Exposure Time (us)' ,'ExposureTime'],
        ['Line Scan Rate (Hz)', 'LineScanRate'],
        ['Num Camera Pixels' ,'NumCameraPix'],
        ['Number of A-scans' ,'Ascans'],
        ['Number of Frames', 'Frames'],
        ['X-Galvo Amplitude (V)', 'GalvoXV'],
        ['Y-Galvo Min (V)', 'GalvoYVmin'],
        ['Y-Galvo Max (V)' ,'GalvoYVmax'],
        ['X Galvo Offset' ,'GalvoXVoffset'],
        ['Y Galvo Offset', 'GalvoYVoffset'],
        ['Calculated Frame Rate (Hz)' ,'fpsCalc'],
        ['Scan Mode Select' ,'ScanModeSelect'],
        ['gamma' ,'gamma'],
        ['scale' ,'scale'],
        ['Alpha 2' ,'alpha2'],
        ['Alpha 3' ,'alpha3'],
        ['Beta 2' ,'beta2'],
        ['Beta 3', 'beta3'],
        ['wctr' ,'wctr'],
        ['Shift' ,'shift'],
        ['Modulation Amplitude', 'ModulationAmp'],
        ['Start Frequency', 'wStart'],
        ['End frequency' ,'wEnd'],
        ['Ramp Fraction', 'RampFraction'],
        ['Start Delay', 'StartDelay'],
        ['End Early' ,'EndEarly'],
        ['Acceleration' ,'accel'],
        ['Flyback Velocity' ,'Vflyback'],
        ['Scanning Velocity' ,'Vscanning'],
        ['Min Jerk Time' ,'JerkTimeMin'],
        ['Max Jerk Time' ,'JerkTimeMax'],
        ['Fast axis Range (mm)' ,'FastAxisRange'],
        ['Slow axis step size (mm)' ,'SlowAxisStepSize'],
        ['Actual Frame Rate (Hz)' ,'fpsTrue']])
        self.ImportSettings()         

    def ImportSettings(self): # path should end with /             
        Stree = ET.parse(self.SamplePath+self.sampleID+"_settings.xml")   
        nodeTypes = ['I32', 'DBL', 'SGL']
        Stree_root = Stree.getroot()
        for item in nodeTypes:
            nodeIterTree = list(Stree_root.iter(item))
            for node in nodeIterTree:
                NameInXMLConversion = [] 
                NameListTree = list(node.iter("Name"))
                for jp in NameListTree:
                    Name = str(jp.text)
                    NameInXMLConversion = np.argwhere(self.XMLvariableConversion[:,0]==Name)
                ValListTree = list(node.iter("Val")) 
                for jq in ValListTree:
                    if item == "I32":
                        Val = int(jq.text)
                    elif item == "DBL":
                        Val = float(jq.text) 
                    elif item == "SGL":
                        Val = float(jq.text)                
                if NameInXMLConversion.size > 0:
                    tmpName = self.XMLvariableConversion[NameInXMLConversion[0],1]
                    self.Settings[tmpName[0]] = Val  

    def readData(self, filename, Settings,datatype, start_read, readVol):
        """Reading binary data
        :filename: .bin file to be read into 
        :Settings: OCT setting parameters 
        :datatype: "data" or "background"
        :start_read: start reading position in bytes 
        :readVol: total number of size to read as Z*X*Y  
        """
        _timeReadStart = time.time() 
        fid = open(filename,'rb')
        if self.verbose:
            print("...................................")
            if datatype == "data":
                print("Loading Sample Data ...")
            elif datatype == "background":
                print("Loading Background Data ...")
        if datatype == "data":
            Z = int(Settings['NumCameraPix'])
            X = int(Settings['Ascans']) 
            fid.seek((start_read*X*Z) * np.uint16(0).nbytes,0)
            # offset = N * X * Z indicates offet to start from Nth frame (y)
            raw_string = np.fromfile(fid,dtype='uint16',count=readVol)  # numpy read file as row-vector, which is different from matlab as column-vector
            Y = int(readVol/(Z*X))          
            output = np.double(np.transpose(np.reshape(np.swapaxes(np.reshape(raw_string,(Y,X,Z)),0,1),(X*Y,Z)))) # now as [Z,X*Y] 
            _timeReadEnd = time.time() 
            if self.verbose:
                print("Total time of loading sample data is {0:.5f} sec".format(_timeReadEnd-_timeReadStart)) 
                print("End of Loading Sample Data ...")
                print("...................................\n")
            fid.seek(0,0) #repositioning to the beginning of file
            fid.close()
            return output 
        elif datatype == "background":
            raw_string = np.fromfile(fid,dtype ='uint16') 
            Z = int(Settings['NumCameraPix'])
            X = int(np.size(raw_string)/Z)
            output = np.double(np.transpose(np.reshape(raw_string,(X,Z))))
            _timeReadEnd = time.time()
            if self.verbose:
                print("Total time of loading sample data is {0:.5f} sec".format(_timeReadEnd-_timeReadStart)) 
                print("End of Loading Background Data ...")
                print("...................................\n") 
            fid.close()
            return output 
        else:
            fid.close()
            raise ValueError


    def OneDimPlot(self,InputLineData_y,InputLineData_x=None,PlotMethod ='line',FigTitle=None,XLabel=None,YLabel=None,XLim=None,YLim=None):
        """1D line or scatter plot
        :InputLineData_y: required, y 
        :InputLineData_x: optional, default as None. If None, InputLineData will be generated automatically as linear number sequence
        :PlotMethod: optional, "line" or "scatter", the way to plot the data 
        :FigTitle: optional, the title of figure. Default is None
        :XLabel,YLabel: optional, label of axis. Default is None
        :XLim, Ylim: optional, limits of axis. Default is None. If given, each should have a format as [vmin, vmax] 
        """ 
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
        font = {'weight': 'normal',
                'size'   : 15}
        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        figLine = plt.figure(constrained_layout=False,figsize=(5,4))
        plt.tight_layout() 
        if FigTitle:
            figLine.suptitle(FigTitle, fontsize=13)            
        axLine = plt.subplot2grid((15,15),(0,1),rowspan=14,colspan=14)
        if InputLineData_x == None:
            InputLineData_x = np.linspace(1,np.size(InputLineData_y),np.size(InputLineData_y)) 

        if PlotMethod.lower() == "line":            
            axLine.plot(InputLineData_x,InputLineData_y,linewidth=0.2)
        elif PlotMethod.lower() == "scatter":
            axLine.scatter(InputLineData_x,InputLineData_y,s=10,c='#1f77b4',marker='o',alpha=0.7,linewidths=0.1,edgecolors='face')
        if XLabel:
            axLine.set_xlabel(r'Depths (pixels)')
        if YLabel:
            axLine.set_ylabel(r'OCT Intensity (ar.u)')
        if XLim:
            axLine.set_xlim(XLim)
        if YLim:
            axLine.set_ylim(YLim)


    def bkgndSubtraction(self,arry):
        return arry-self.BkgndData
    
    def Resample(self):
        N = self.Settings['NumCameraPix'] 
        k_acquired = np.transpose(self.Settings['k']) 
        Rj = (N-1)*(k_acquired - k_acquired[0])/(k_acquired[-1] - k_acquired[0]) + 1 # Input (raw) indices
        Ri =  np.linspace(1,N,N) # Output indices
        Rj = np.reshape(Rj,(1,np.size(Rj))) 
        Ri = np.reshape(Ri,(np.size(Ri),1))  
        rescale = Rj[0,1:] - Rj[0,0:-1]
        rescale = np.append(rescale,rescale[-1])
        rescale = np.reshape(rescale,(np.size(rescale),1))
        resample = np.matlib.repmat(rescale,1,N)*np.sinc(np.matlib.repmat(Rj,N,1) - np.matlib.repmat(Ri,1,N)) 
        resample[np.isnan(resample)] = 0 
        resample[np.abs(resample)< 1E-3] = 0 
 
        return resample  

    def AlphaMatrix(self):
        N = self.Settings["NumCameraPix"]
        a2 = self.Settings['alpha2']
        a3 = self.Settings['alpha3'] 
        w0 = self.Settings['wctr'] 
        kz = np.transpose(np.linspace(1,N,N)/N - w0)
        CorrectionVector = np.exp(1j*np.pi*(a2*(kz**2)+a3*(kz**3)))
        alpha = np.diag(CorrectionVector) 
        return alpha
 
    def ShowXZ(self,InputData,frame_toshow=0,figTitle = None,figHandle=None): 
        """Show 2D image with depth profile, laser spectrum and interference pattern 
        : InputData: 2D numpy array; if InputData is 3D as [Z,X,Y] dim, it will only show its first frame. 
        """
        from matplotlib import cm 
        from matplotlib.ticker import AutoMinorLocator
        # set font of plot 
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
        font = {'weight': 'normal',
                'size'   : 12}
        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        if figHandle == None:
            figOCTXZ = plt.figure(constrained_layout=False,figsize=(10,5))
        else:
            figOCTXZ = figHandle 
        plt.tight_layout(pad=0.0)
        plt.subplots_adjust(wspace=20,hspace=5,left=0.01, right=0.98) # fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
        if figTitle:
            figOCTXZ.suptitle(figTitle,fontsize=13)
        else:
            figOCTXZ.suptitle('OCT Cross Section XZ', fontsize=13)
        axOCTXZ = plt.subplot2grid((15,15),(0,1),rowspan=14,colspan=7)
        axDepth = plt.subplot2grid((15,15),(9,8),rowspan=4,colspan=7)
        axSpectrum_tmp = plt.subplot2grid((15,15),(1,8),rowspan=4,colspan=7)
        axSpectrum = axSpectrum_tmp.twiny() 
        axIntere = plt.subplot2grid((15,15),(5,8),rowspan=4,colspan=7) 
        
        for ax0 in [axDepth,axSpectrum,axIntere]:
            ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax0.yaxis.set_minor_locator(AutoMinorLocator(2))
        axIntere.xaxis.set_ticklabels([])
        axSpectrum_tmp.xaxis.set_ticklabels([])
        
        if np.size(np.shape(InputData)) == 3:
            ShowData = np.abs(InputData[:,:,frame_toshow])
        elif np.size(np.shape(InputData)) == 2:
            ShowData = np.abs(InputData)
        else:
            raise ValueError 
        MaxAmp = np.amax(ShowData**self.Settings['gamma'])
        OCTnorm = matplotlib.colors.Normalize(vmin = 2*np.amin(ShowData**self.Settings['gamma']),vmax = 0.7*np.amax(ShowData**self.Settings['gamma']))
        axOCTXZ.imshow(ShowData**self.Settings['gamma'],cmap=cm.hot,norm=OCTnorm,interpolation=None, aspect='auto',vmin = 2*np.amin(ShowData**self.Settings['gamma']),vmax = 0.7*np.amax(ShowData**self.Settings['gamma']))
        #axOCTXZ.imshow(ShowData**self.Settings['gamma'],cmap=cm.hot,norm=OCTnorm,vmin=1*np.amin(ShowData**self.Settings['gamma']),vmax=0.8*np.amax(ShowData**self.Settings['gamma']))
        axOCTXZ.set_xlabel(r"$x$ (pixels)")
        axOCTXZ.set_ylabel(r"$z$ (pixels)")
        axDepth.plot(np.linspace(1,np.size(self.DepthProfile),np.size(self.DepthProfile)),np.abs(self.DepthProfile),linewidth=1,color='C2')
        axSpectrum.plot(np.linspace(1,np.size(self.BkgndData),np.size(self.BkgndData)),np.abs(self.BkgndData),linewidth=1,color='C1',label='bkgnd') 
        axSpectrum.plot(np.linspace(1,np.size(self.SampleSpectrum),np.size(self.SampleSpectrum)),np.abs(self.SampleSpectrum),linewidth=1,color='C3',label='signal')
        axIntere.plot(np.linspace(1,np.size(self.InterferencePattern),np.size(self.InterferencePattern)),np.real(self.InterferencePattern),linewidth=1,color='C1')
        axDepth.set_xlabel(r"$z$ for Depth profile (pixels)")
        axDepth.set_ylabel(r"Profile",labelpad=-6)        
        axIntere.set_ylabel(r"Interference",labelpad=2) 
        axSpectrum_tmp.set_ylabel(r"Spectrum",labelpad=0)
        axDepth.set_xlim([np.amin(np.linspace(1,np.size(self.DepthProfile),np.size(self.DepthProfile))),np.amax(np.linspace(1,np.size(self.DepthProfile),np.size(self.DepthProfile)))])
        axSpectrum.set_xlim([np.amin(np.linspace(1,np.size(self.BkgndData),np.size(self.BkgndData))),np.amax(np.linspace(1,np.size(self.BkgndData),np.size(self.BkgndData)))])
        axIntere.set_xlim([np.amin(np.linspace(1,np.size(self.InterferencePattern),np.size(self.InterferencePattern))),np.amax(np.linspace(1,np.size(self.InterferencePattern),np.size(self.InterferencePattern)))]) 
        axSpectrum.set_xlabel(r'$z$ for Spectrum and Interference (pixels)')
        axDepth.set_yscale('log')
        axSpectrum.legend(fontsize=6,loc='best')

        return figOCTXZ, [axOCTXZ, axDepth, axSpectrum, axIntere] 

    def saveData(self):
        _datasaveTime1 = time.time() 
        #import scipy.io 
        import h5py
        if not self.saveFolder:
            savePath = self.root_dir
        else:
            if not os.path.exists(self.root_dir+'/'+self.saveFolder):
                os.mkdir(self.root_dir+'/'+self.saveFolder)
            savePath = self.root_dir+'/'+self.saveFolder
             
        #scipy.io.savemat(savePath+'/'+'Settings.mat',appendmat=False ,mdict={'Settings': self.Settings})
        SettingsFile = h5py.File(savePath+'/'+'Settings.hdf5','w')
        for k, v in self.Settings.items():
            SettingsFile.create_dataset(k, data=v)  
        SettingsFile.close()       
        if self.RorC.lower() == "real":
            if not self.verbose:
                print(' Start saving data as Real')
            DataFileSave = h5py.File(savePath+'/'+self.sampleID+'_OCTData.hdf5','w')
            DataFileSave.create_dataset('OCTData',shape=np.shape(self.OCTData),data=self.OCTData,compression="gzip")
            DataFileSave.close()
           # scipy.io.savemat(savePath+'/'+self.sampleID+"_OCTData.mat",appendmat=False,mdict={"data": self.OCTData}) 
        else:
            if not self.verbose:
                print(' Start saving data as Complex')
            DataFileSave = h5py.File(savePath+'/'+self.sampleID+'_OCTData_Complex.hdf5','w')
            DataFileSave.create_dataset('OCTData_real',shape=np.shape(self.data),data=np.real(self.data),compression="gzip")
            DataFileSave.create_dataset('OCTData_imag',shape=np.shape(self.data),data=np.imag(self.data),compression="gzip")
            DataFileSave.close()
            #scipy.io.savemat(savePath+'/'+self.sampleID+"_OCTData_complex.mat",appendmat=False,mdict={"data": self.data}) 
        _datasaveTime2 = time.time() 
        if self.verbose:
            print("Data Saving Time Cost is {0:.5f} sec".format(_datasaveTime2-_datasaveTime1))
            print("End of data saving !!")
            print("*************************************\n") 

class Batch_OCTProcessing():
    """Processing batch OCT image files
    Processing OCT data by segmenting data set into sub-frames. You can still access to OCTImagingProcessing() class methods by accessing .OCTRe.Methods() 

    """
    def __init__(self,root_dir, ChunkSize= 128,Ascans=120,Frames=128,SampleData = None, Settings = None, sampleID=None,bkgndID = None,Sample_sub_path = None,Bkgnd_sub_path = None,saveOption=False,saveFolder=None,RorC='real',verbose=False, alpha2=None,alpha3=None,depths=np.linspace(1024,2047,1024),gamma = 0.4,wavelegnth = 1300,XYConversion=[660,660],camera_matrix=np.asarray([1.169980E3,1.310930E-1,-3.823410E-6,-7.178150E-10]),start_frame = 1, singlePrecision = True, OptimizingDC=False,ReconstructionMethods='NoCAO',downPrecision=True):
        self.ChunkSize = ChunkSize 
        self.Ascans = Ascans
        self.Frames = Frames 
        self.Segments = int(np.trunc(np.floor(self.Frames/self.ChunkSize))) 
        self.data = np.zeros((np.size(depths),self.Ascans,self.Frames),dtype=np.complex)
        self.Settings = Settings 
        _time01 = time.time()
        if verbose:
            bar = Bar(' OCT Batch Reconstruction...', max=self.Segments)
        for i in range(self.Segments):
            if verbose:
                bar.next()
                print("")
            self.OCTRe = OCTImagingProcessing(root_dir=root_dir, SampleData=SampleData, Settings=None, sampleID=sampleID,bkgndID=bkgndID,Sample_sub_path=Sample_sub_path,Bkgnd_sub_path=Bkgnd_sub_path,saveOption=saveOption,saveFolder=saveFolder,RorC='complex',verbose=False, frames=self.ChunkSize,alpha2=alpha2,alpha3=alpha3,depths=depths,gamma=gamma,wavelegnth=wavelegnth,XYConversion=XYConversion,camera_matrix=camera_matrix,start_frame=i*self.ChunkSize+1, singlePrecision=singlePrecision, OptimizingDC=False,ReconstructionMethods=ReconstructionMethods)
            if i == 0:
                self.Settings = self.OCTRe.Settings 
                if verbose:
                    for key in self.Settings.keys():
                        print("{}: {}".format(key,self.Settings[key]))
            self.data[:,:,i*self.ChunkSize+np.arange(0,self.ChunkSize,step=1,dtype=np.int)] = np.reshape(self.OCTRe.data,(np.size(self.Settings['depths']),self.Settings['Ascans'],int(np.size(self.OCTRe.data)/(np.size(self.Settings['depths'])*self.Settings['Ascans'])))) 
        if (self.Segments*self.ChunkSize - self.Frames) != 0:
            diff = int(self.Frames-self.Segments*self.ChunkSize)
            if diff < 0:
                raise ValueError("Segmentation is not Right ! Check ChunkSize!")
            else:
                self.OCTRe = OCTImagingProcessing(root_dir=root_dir, SampleData=SampleData, Settings=None, sampleID=sampleID,bkgndID=bkgndID,Sample_sub_path=Sample_sub_path,Bkgnd_sub_path=Bkgnd_sub_path,saveOption=saveOption,saveFolder=saveFolder,RorC='complex',verbose=False, frames=diff,alpha2=alpha2,alpha3=alpha3,depths=depths,gamma=gamma,wavelegnth=wavelegnth,XYConversion=XYConversion,camera_matrix=camera_matrix,start_frame=self.Segments*self.ChunkSize+1, singlePrecision=singlePrecision, OptimizingDC=False,ReconstructionMethods=ReconstructionMethods)
                self.data[:,:,(self.Segments)*self.ChunkSize+np.arange(0,diff,step=1,dtype=np.int)] = np.reshape(self.OCTRe.data,(np.size(self.Settings['depths']),self.Settings['Ascans'],int(np.size(self.OCTRe.data)/(np.size(self.Settings['depths'])*self.Settings['Ascans'])))) 
        if downPrecision:
            self.data = self.data.astype(np.complex64)
        
        self.OCTData =  np.abs(self.data)
        if singlePrecision:
            self.OCTData = self.OCTData.astype(np.float32) # convert to single precision 
        #self.data = self.data.astype(np.complex64)

        if RorC.lower() == "real":
            del self.data
        _time02 = time.time()
        if verbose:
            print("Total Reconstruction time is {} sec".format(_time02-_time01))

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    import tkinter as tk
    from tkinter import filedialog
    from tkinter.filedialog import askopenfilename
    root = tk.Tk()
    root_dir = filedialog.askdirectory(parent = root,initialdir="/",title='Please select a directory to load and save data...')
    sampleID_full =  askopenfilename(filetypes=[("Binary files", "*.bin")],title="Please select your data file")
    sampleID = os.path.basename(sampleID_full)
    sampleID = sampleID[0:-8]
    bkgndID_full = askopenfilename(filetypes=[("Binary files", "*.bin")],title="Please select your Background file") 
    bkgndID = os.path.basename(bkgndID_full)
    root.destroy()
    OCTRe = OCTImagingProcessing(root_dir,sampleID,bkgndID,frames=3,alpha2=-50,alpha3=-12,saveOption=False)  
    OCTRe.ShowXZ(OCTRe.OCTData) 
    plt.show() 

