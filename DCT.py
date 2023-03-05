#a program that examines the effect of approximating an image
#with a partial set of DCT coefficients.
#8 × 8 DCT

import cv2
import numpy as np
from numpy import *
import math

###############################################################
#                get the matrix of 8x8 dct
###############################################################
def get_dct_mat():
    #8 x 8 DCT matrix
    dct_mat = np.zeros((8,8))

    for x in range(8):
        dct_mat[0,x] = sqrt(2.0/8) / sqrt(2.0)
    for u in range(1,8):
        for x in range(8):
            dct_mat[u,x] = sqrt(2.0/8) * cos((pi/8) * u * (x + 0.5))

    np.set_printoptions(precision=4)

    return dct_mat
    
################################################################
##                   dct
################################################################
def dct(img_blk,dct_mat):
    #Discrete	Cosine	Transform
    #img_blk: 8x8 block from original image

    dct_ret = np.dot(np.dot(dct_mat,img_blk),dct_mat.transpose())

    return dct_ret  

################################################################
##                   inverse dct
################################################################
def idct(img_blk,dct_mat):
    #inverse Discrete	Cosine	Transform
    #img_blk: 8x8 block

    #according to property of DCT,
    #dct transpse is the inverse of dct

    idct_ret = np.dot(np.dot(dct_mat.transpose(),img_blk),dct_mat)

    return idct_ret
###################################################################
#               reconstruct using K elements
###################################################################
def zigzag(arr,k):
    arr = np.array(arr)
    ret = np.zeros(arr.shape)
    
    #reconstruct in zigzag order
    (i,j) = (0,0)
    direction = 'r' #{'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    
    for e in range(k):
        ret[i][j] = arr[i][j]
        if direction == 'r':
            j += 1
            if i == arr.shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i += 1
            j -= 1
            if i == arr.shape[0] - 1:
                direction = 'r'
            elif j == 0:
                direction = 'd'
        elif direction == 'd':
            i += 1
            if j == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i -= 1
            j += 1
            if j == arr.shape[1] - 1:
                direction = 'd'
            elif i == 0:
                direction = 'r'

    return ret
        

#########################################################################
##################      calculate PSNR     ##############################
#########################################################################
def calc_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

################################################################
#                     main                          
################################################################
def main():
    #Using an 8 × 8 DCT,
    #reconstruct the image with K < 64 coefficients,
    #when K = 2, 4, 8, 16, and 32
    K = 32
    img_name = "cat.png"
    img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    dim = (512,512)
    img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)

    #array to store img after dct
    img_dct = np.zeros((512,512))
    #array to store img after reconstruction (idct)
    img_idct = np.zeros((512,512))
    #array to store img after quantization
    #img_qtz = np.zeros((512,512))
    #array to store img after de-quantization
    #img_de_qtz = np.zeros((512,512))
    #array to store img with K coefficients
    img_k = np.zeros((512,512))

    # quantization block from slides
    #qtzn_blk = [[16, 11, 10, 16, 24, 40, 51, 61],
     #        [12, 12, 14, 19, 26, 58, 60, 55],
      #       [14, 13, 16, 24, 40, 57, 69, 56],
       #      [14, 17, 22, 29, 51, 87, 80, 62],
        #     [18, 22, 37, 56, 68, 109, 103, 77],
         #    [24, 35, 55, 64, 81, 104, 113, 92],
          #   [49, 64, 78, 87, 103, 121, 120, 101],
           #  [72, 92, 95, 98, 112, 100, 103, 99]]

    #get 8x8 dct matrix
    dct_mat = get_dct_mat()

    #split to 8x8 blocks and use dct for each
    for i in range(0,512,8):
        for j in range(0,512,8):
            blk = img[i:i+8, j:j+8]
            dct_blk = dct(blk,dct_mat)
            img_dct[i:i+8,j:j+8] = np.copy(dct_blk)

    #quantization
    #for i in range(0,512,8):
     #   for j in range(0,512,8):
      #      img_qtz[i:i+8,j:j+8] = np.around(np.divide(img_dct[i:8+8,j:j+8],qtzn_blk))

    #de-quantization
    #for i in range(0,512,8):
     #   for j in range(0,512,8):
      #      img_de_qtz[i:i+8,j:j+8] = np.around(np.multiply(img_qtz[i:8+8,j:j+8],qtzn_blk))

    #keep the first K dct coefficient for every 8x8 block
    for i in range(0,512,8):
        for j in range(0,512,8):
            blk = img_dct[i:i+8, j:j+8]
            img_k[i:i+8,j:j+8] = zigzag(blk,K)

    #inverse dct for each 8x8 block
    for i in range(0,512,8):
        for j in range(0,512,8):
            blk = img_k[i:i+8, j:j+8]
            idct_blk = idct(blk,dct_mat)
            img_idct[i:i+8,j:j+8] = np.copy(idct_blk)

            
    #save reconstructed image
    cv2.imwrite("reconstructed image with K = " + str(K) + ".png",img_idct)
    #calculate psnr
    psnr_val = calc_psnr(img_idct,img)

    print("psnr of the reconstructed image with K = ", K, "vs original image : ", psnr_val)

    return 0

if __name__ == "__main__":
    main()
    
            
            

    
    
