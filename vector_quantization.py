import cv2
import numpy as np
from collections import defaultdict
import math

#part 1 - Vector quantization
#a program to implement vector quantization on a gray-scale image
#using a “vector” that consists of a 4x4 block of pixels


########################################################################
####### generate "vector” that consists of a 4x4 block of pixels #######
########################################################################
def get_vector(img_list):
    vec = []
    
    for img in img_list:
        #print(img.shape)
        dim = (512,512)
        resized = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
        #print(resized.shape)
        for i in range(0, resized.shape[0], 4):
            for j in range(0, resized.shape[1], 4):
                
                vec.append(resized[i:i+4, j:j+4].reshape((4*4)))

    print(np.array(vec).shape)

    return (np.array(vec))

#generate codebook
def get_codebook(vec,size):
    #vec: "vector” that consists of a 4x4 block of pixels
    #size: codebook size, 
    #generate codebook of given size from a list of images 

    codebk = []
    code_init = []
    #get the average of the vector
    avg_vec = vec / len(vec)
    sum_vec = np.sum(avg_vec, axis=0)
    #print("sum_vec shape: ",sum_vec.shape)

    code_init = sum_vec.tolist()
    codebk.append(code_init)

    #get distortion
    distor = calc_distortion(code_init,vec)

    #generate codebook of size
    while len(codebk) < size:
        codebk = enlarge_codebook(vec,codebk, 0.05, distor)

    return codebk
           
#####################################################################
#########        enlarge the codebook     ###########################
#####################################################################
        
def enlarge_codebook(vec, codebk,eps,distor):
    #vec: "vector” that consists of resized 4x4 block of pixels
    #codebk: codebook to be enlarged
    #eps:the threshold that used during splitting and looping
    #distor: initial distortion

    new_cb =[]
    for c in codebk:
        c1 = (np.array(c) * (1.0 + eps)).tolist()
        c2 = (np.array(c) * (1.0 - eps)).tolist()

        new_cb.extend((c1,c2))

    #k-means
    avg_distor = 0
    err = eps + 1
    itern = 0
    while err > eps:
        #print(err)
        #get nearest neighbor
        n_vec = [None] * len(vec) #to store nearest vector from codebook
        dic_vec = defaultdict(list) #a dictionary to map vec to codebk values
        dic_id = defaultdict(list) #map ideces

        for i,v in enumerate(vec):
            min_distor = None
            n_cb_id = None

            for j,c in enumerate(new_cb):
                d = calc_mse(v,c)
                #save nearest
                if min_distor is None or d < min_distor:
                    min_distor = d
                    n_vec[i] = c
                    n_cb_id = j

            dic_vec[n_cb_id].append(v)
            dic_id[n_cb_id].append(i)

        #update codebook
        for j in range(len(new_cb)):
            vc = dic_vec.get(j) or []
            num_vecs_near = len(vc)
            if num_vecs_near > 0:
                #re-center
                np_temp = np.array(vc) / len(vc)
                sum_tmp = np.sum(np_temp, axis = 0)
                new_center = sum_tmp.tolist()

                new_cb[j] = new_center

                for i in dic_id[j]:
                    n_vec[i] = new_center


        #re-calculate distortion
        prev_distor = avg_distor if avg_distor > 0 else distor
        avg_distor = calc_distortion(n_vec,vec)

        #update error
        err = (prev_distor - avg_distor) / prev_distor
        #print(err)

    return new_cb
            
                
#######################################################################    
########     calculate the distortion of a vector l to data   #########
#######################################################################
def calc_distortion(l, data):
    size = len(data)
    distortion = np.sum( (np.array(l)-np.array(data)) **2 / size)

    return distortion

#########################################################################      
############        calculate mse        ################################
#########################################################################
def calc_mse(a,b):
    return(np.sum((np.array(a) - np.array(b))**2))

#########################################################################
######################      encode       ################################
#########################################################################
def encode(img, cb):
    #store the index of nearest vector
    cmpressed = np.zeros((img.shape[0] // 4, img.shape[1] // 4))
    id_x = 0
    for i in range(0,img.shape[0],4):
        id_y = 0
        for j in range(0,img.shape[1],4):
            src = img[i:i+4,j:j+4].reshape((4*4)).copy()
            k = find_nearest(src,cb)
            cmpressed[id_x,id_y] = k
            id_y += 1

        id_x += 1

    return cmpressed
            

#########################################################################
###################   find nearest vector in codebook  ##################
def find_nearest(src, cb):
    tmp = np.zeros((cb.shape[0],))
    for i in range(0, cb.shape[0]):
        tmp[i] = np.mean((np.subtract(src,cb[i])**2))
    nearest = np.argmin(tmp,axis=0)
    return nearest
    

#########################################################################
######################      decode       ################################
#########################################################################
def decode(cmpressed, cb):
    decoded = np.zeros((cmpressed.shape[0] * 4, cmpressed.shape[1] * 4))
    id_x = 0
    for i in range(0,cmpressed.shape[0]):
        id_y = 0
        for j in range(0, cmpressed.shape[1]):
            decoded[id_x:id_x+4,id_y:id_y+4] = cb[int(cmpressed[i,j])].reshape((4,4))
            id_y += 4
        id_x += 4
    return decoded

########################################################################
##################    vector quantization      #########################
########################################################################
def VQ(img_name_list, cb_size, img_to_quant):
    #img_name_list: list of images' names in string
    #cb_size: codebook size
    #img_to_quant:image to be quantized

    img = cv2.imread(img_to_quant, cv2.IMREAD_GRAYSCALE)
    dim = (512,512)
    img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("greyscale.png",img)

    length_l = len(img_name_list)

    img_list = [] #to store grey-scale imgs
    for n in img_name_list:
        img_list.append(cv2.imread(n, cv2.IMREAD_GRAYSCALE))

    train_vec = get_vector(img_list)
    print('generating codebook...(taking minutes)')
    codebook = get_codebook(train_vec, cb_size)
    cb = np.array(codebook)
    print('encoding')
    encoded = encode(img, cb)
    print('decoding')
    decoded = decode(encoded, cb)
    cv2.imwrite("decoded_" + str(length_l) + "_" + str(cb_size) + ".png",decoded)
    psnr_val = calc_psnr(img,decoded)
    print('psnr value with codebook of size ', cb_size, ": ", psnr_val)

    return 0
        

#########################################################################
##################      calculate PSNR     ##############################
#########################################################################
def calc_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    
##########################################################################
################      main        ########################################
##########################################################################
def main():
    codebook_size = 32
    img_to_quant = 'watch.png'
    img_name_list1 = ['watch.png']
    img_name_list10 = ['cat.png','airplane.png','arctichare.png', 'baboon.png', 'monarch.png', 'mountain.png', 'pool.png', 'tulips.png', 'watch.png', 'zelda.png']
    ret = VQ(img_name_list10, codebook_size, img_to_quant)

    return 0

if __name__ == "__main__":
    main()
    
