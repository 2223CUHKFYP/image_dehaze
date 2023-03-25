import cv2
import math
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from PNGFileFinder import PNGFileFinder
from ImageSaver import PNGFileSaver
from tqdm import tqdm
from functools import partial
import argparse


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res





def process_image(file_path, output_path):
    src = cv2.imread(file_path)

    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)
    path_split = file_path.split('/')
    # input -> output
    tmp2 = output_path + '/' + path_split[-3] +'/' + path_split[-2] 
    
    # check if the folder already exists
    os.makedirs(tmp2, exist_ok=True)
    # if not os.path.exists(tmp2):
    #     # create the new folder using the os module
    #     os.makedirs(tmp2)
 
    tmp =  output_path + '/' + path_split[-3] +'/' + path_split[-2] + '/' + path_split[-1]
    print(tmp)
    cv2.imwrite(tmp,  J*255)

def list_files(root_dir, extensions=[".jpg"]):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

if __name__ == '__main__':
    print("test")
    
    parser = argparse.ArgumentParser(description='Process images in a directory.')
    parser.add_argument('--input_path', type=str, help='Path to the input directory.' )
    parser.add_argument('--output_path', type=str, help='Path to the output directory.')

    args = parser.parse_args()

    # get a list of file paths for all PNG files in the current directory
    imagePath = list_files(args.input_path)

    # create a process pool with the number of available CPU cores
    pool = Pool(cpu_count())

    # use functools.partial to fix output_path argument
    process_image_with_output_path = partial(process_image, output_path=args.output_path)

    # use tqdm to display a progress bar
    for _ in tqdm(pool.imap_unordered(process_image_with_output_path, imagePath), total=len(imagePath)):
        pass

    # close the process pool
    pool.close()
    pool.join()

    print("Done")
