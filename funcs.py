## Functions used in "BP Deblur"

import numpy as np
from scipy import io
import astra
import torch
from torch.autograd import Variable
import pylab
from skimage.transform import resize, rotate
from skimage.morphology import closing
from model import DeepRFT as myNet
import sr


def BP_reconstruction(Input_signal, angles, result_size=512, \
                      det_width=1.3484, det_count=560, source_origin=410.66, \
                      origin_det=143.08, eff_pixelsize=0.1483 ):

    '''
    Back projection. The code is based on HelTomo.
    Inputs:
        Input_signal: measured sinogram
        angles: projection angles
        result_size: pixel number of the reconstruction domain
        det_width: distance between the centers of two adjacent detector pixels
        det_count: number of detector pixels in a single projection
        source_origin: distance between the source and the center of rotation
        origin_det: distance between the center of rotation and the detector array
        eff_pixelsize: effictive size of pixels
    Output:
        Bp: result of the back projection method
    '''

    ##Distances from specified in terms of effective pixel size
    source_origin=source_origin/eff_pixelsize
    origin_det=origin_det/eff_pixelsize
    
    ##Transform angles to radians
    angles=np.radians(angles)

    ##Define the geomotry
    vol_geom = astra.create_vol_geom(result_size, result_size) 
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles,source_origin,origin_det)   
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
   
    ##Get the projection matrix
    W = astra.optomo.OpTomo(proj_id)
    ##Back projection
    Bp = W.T.dot(Input_signal.ravel())
    Bp = np.reshape(Bp, (result_size,result_size))

    astra.projector.delete(proj_id)

    return Bp


def Deep_Deblur(Input_albedo, group_number, device,img_resolution=128):
    '''
    Use network to enhance the result of back projection.
    Inputs:
        Input_albedo: result of the back projection method
        group_number: number of limited-angle tomography difficulty group
        img_resolution: resolution of input and output(if changed, the network should be retrained)
    Output:
        output: deblur result 
    '''
    
    ##Define the network and load pretrained weights to gpu
    net = myNet()
    try:
        net.load_state_dict(torch.load('./pre-trained weights/level_%s.pkl'%(group_number)))
    except:
        net=torch.nn.DataParallel(net)
        net.load_state_dict(torch.load('./pre-trained weights/level_%s.pkl'%(group_number)))
    
    net = net.to(device)

    ##Normalization
    Input_albedo=Input_albedo/np.max(Input_albedo)
    
    ##Deblur
    with torch.no_grad():
        albedo = Variable(torch.from_numpy(Input_albedo)).reshape(1,1,img_resolution,img_resolution)
        albedo = albedo.to(device).type(torch.cuda.FloatTensor)

        output = net(albedo)
        output = output.data.cpu().numpy()
        output = output.reshape(1,1,img_resolution,img_resolution)
        output=np.squeeze(output/np.max(output))
    
    return output



def Load_process(data_path,output_path,group_number):
    '''
    Load data from data path and reconstruct the phantom, then save the results to output path.
    Inputs:
        data_path: the path of the input mat file
        output_path: the path of the output png image
        group_number: difficulty level, to determine which pre-trained network to load 
    Output:
        None

    '''
    ##load data
    data=io.loadmat(data_path)['CtDataLimited'] 
    ##extract information from data
    sinogram=data['sinogram'][0][0]
    
    parameters=data['parameters'][0][0][0][0]  

    eff_pixel_size=parameters[31][0][0]
    det_width=parameters[9][0][0]
    det_count=parameters[32][0][0]

    angles=parameters[11]

    source_origin=parameters[6][0][0]
    origin_det=parameters[7][0][0]-source_origin

    output_size=512
    deblur_size=128
    
    ##detecting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on ',device)
    
    ##let the angles start from 0
    angle_min=np.min(angles)
    angles=angles-angle_min
    
    ##back projection
    BP=BP_reconstruction(sinogram,angles,result_size=output_size, det_width=det_width, det_count=det_count, source_origin=source_origin, origin_det=origin_det, eff_pixelsize=eff_pixel_size)

    

    ##deblur
    BP=resize(BP,output_shape=(deblur_size, deblur_size))
    result=Deep_Deblur(BP,group_number,device)
    ##clear gpu memory
    torch.cuda.empty_cache()
    ##super resolution
    SR=sr.super_resolution(result,device)

    ##totate the reconstruction to original orientation
    SR=rotate(SR,angle_min,order=0)    

    ##save results
    pylab.gray()
    #pylab.imsave('BP.png',BP)
    #pylab.imsave('Deblur.png',result)
    pylab.imsave(output_path,SR)

    #io.savemat('result.mat',{'albedo':SR})
    return 

def find_mat(data_list):
    '''
    Find files with .mat format.
    Input:
        data_list: file names
    Output:
        tmp: name of mat files 
    '''
    tmp=[]
    for i in range(len(data_list)):
        tmp_name=data_list[i]
        if tmp_name[-4:]=='.mat':
            tmp.append(tmp_name)
    return tmp 

