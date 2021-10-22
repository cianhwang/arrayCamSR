import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_exr(path):
    img = cv2.imread(path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.mean(img, axis=2)
    return img

def rot_mat(t):
    theta = np.arccos(t[2]/np.linalg.norm(t))
    phi = np.arctan2(t[1], t[0])
    R = np.array([[np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(phi), np.cos(phi), 0],
               [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]])
    return R

sensor_width = 36. # (mm)
res_w = 2048
res_h = 1536
pix_size = sensor_width/res_w

f = 64. # (mm)
f2 = 64. # (mm)

res_w1, res_h1 = 2048, 1536
t1 = np.array([7, -7, 5])
R1 = rot_mat(t1)
res_w2, res_h2 = 2048, 1536
t2 = np.array([7.5, -6.5, 5])
R2 = rot_mat(t2)

## sensor grid 1
image_coor = np.meshgrid(np.arange(res_h1), np.arange(res_w1))
image_coor = [image_coor[0].transpose(), image_coor[1].transpose()]
image_coor_in_mm = [image_coor[0] * pix_size - res_h1//2 * pix_size + 0.5 * pix_size, 
            image_coor[1] * pix_size - res_w1//2 * pix_size + 0.5 * pix_size]

def calc_corr(Z):
    ## load depth Z, from coor1 to world coor
    cam_coor = [image_coor_in_mm[0]/f * Z, 
                image_coor_in_mm[1]/f * Z,
                -Z]
    p = np.stack(cam_coor).reshape((3, -1))
    p_world = R1.transpose().dot(p) + t1.reshape((-1, 1))
    
    ## world to coor2
    p2 = R2.dot(p_world - t2.reshape((-1, 1)))
    cam2_coor = p2.reshape((3, res_h1, res_w1))
    image2_coor_in_mm = [-cam2_coor[0]/cam2_coor[2] * f2, -cam2_coor[1]/cam2_coor[2] * f2]
    image2_coor = [image2_coor_in_mm[0]/pix_size - 0.5 + res_h2//2, 
                   image2_coor_in_mm[1]/pix_size - 0.5 + res_w2//2]
    return image2_coor

def align_image(Z1, image2):
    image2_coor = calc_corr(Z1)
    x = np.floor(image2_coor[0]).astype(np.long)
    y = np.floor(image2_coor[1]).astype(np.long)
    dx = image2_coor[0] - x
    dy = image2_coor[1] - y
    assert len(image2.shape) == 2 or len(image2.shape) == 3
    if len(image2.shape) == 3:
        dx = dx[..., np.newaxis]
        dy = dy[..., np.newaxis]
    x = np.clip(x, 0, res_h2-2)
    y = np.clip(y, 0, res_w2-2)
    image2_warp = (1-dx)*(1-dy)*image2[x, y] + \
                    (dx)*(1-dy)*image2[x+1, y] + \
                    (1-dx)*(dy)*image2[x, y+1] + \
                    (dx)*(dy)*image2[x+1, y+1]
    return image2_warp

if __name__=='__main__':   
    image1 = load_exr('/home/qian/Desktop/scene0000/Image_550/Image0005_obj1.exr')/6.0
    image2 = load_exr('/home/qian/Desktop/scene0000/Image_550/Image0005_obj2.exr')/6.0
    Z = load_exr('/home/qian/Desktop/scene0000/Depth_550/Image0005_obj1.exr')
    print(image1.shape)
    image2_warp = align_image(Z, image2)
    plt.imshow(image1)
    plt.show()
    plt.imshow(image2)
    plt.show()
    plt.imshow(image2_warp)
    plt.show()
    
    
