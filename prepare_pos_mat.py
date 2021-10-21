import numpy as np
import matplotlib.pyplot as plt
import time

def calc_rot(t):
    theta = np.arccos(t[2]/np.linalg.norm(t))
    phi = np.arctan2(t[1], t[0])
    R = np.array([[np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(phi), np.cos(phi), 0],
               [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]])
    return R

def calc_tx(t):
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])

class Cam:
    def __init__(self, p, h, w, f, t):
        self.pix_size = p
        self.res_h = h
        self.res_w = w
        self.f = f
        self.coor = t
        self.R = calc_rot(t)
        
def calc_pix_3d_coor(cam):
    ## reference system: cam
    image_coor = np.meshgrid(np.arange(cam.res_h), np.arange(cam.res_w))
    image_coor = [image_coor[0].transpose(), image_coor[1].transpose()]
    image_coor_in_mm = [ (image_coor[0]-cam.res_h//2+ 0.5) * cam.pix_size, 
            (image_coor[1]-cam.res_w//2+ 0.5) * cam.pix_size]
    cam_coor = np.stack([image_coor_in_mm[0], 
            image_coor_in_mm[1],
            cam.f*np.ones_like(image_coor_in_mm[0])], axis=2)
    return cam_coor

def calc_pos_mat(cam, cam2, R, tx, is_test=False):
    E = R.dot(tx)
    cam_coor = calc_pix_3d_coor(cam)
    cam2_coor = calc_pix_3d_coor(cam2)
    all_lines = []
    for point_x in range(cam.res_h):
        ### for testing ###
        if is_test and point_x != 800//2:
            continue
        ###################
        print(f"process line {point_x}...", end='')
        start_time = time.time()
        line = []
        for point_y in range(cam.res_w):
            ### for testing ###
            if is_test and point_y != 580//2:
                continue
            ###################
            p = cam_coor[point_x, point_y].reshape(-1, 3).T
            lam = E.dot(p)
            dist = np.abs(cam2_coor.reshape(-1, 3).dot(lam))/np.expand_dims((lam[0, :]**2+lam[1, :]**2)**0.5, axis=0)

            k = cam2.res_h+cam2.res_w-1
            pps = np.argpartition(dist.flatten(), k)[:k]
            line.append(pps)
        line = np.stack(line, axis=0)
        all_lines.append(line)
        print(f"elapsed time: {time.time()-start_time}")

    all_lines = np.stack(all_lines, axis=0)
    xxs,yys = all_lines//cam2.res_w, all_lines%cam2.res_w
    return xxs.astype(np.uint16), yys.astype(np.uint16)

if __name__ == '__main__':
    cam = Cam(36./1024., 768, 1024, 64., np.array([7, -7, 5]))
    cam2 = Cam(36./1024., 768, 1024, 64., np.array([7.5, -6.5, 5]))
    R = cam2.R.T.dot(cam.R)
    tx = calc_tx(cam2.coor-cam.coor)
    is_test = False
    xxs, yys = calc_pos_mat(cam, cam2, R, tx, is_test)
    np.save("xxs.npy", xxs)
    np.save("yys.npy", yys)
    
    
    if is_test:
        print(xxs.shape)
        pos_mat = np.zeros((cam2.res_h, cam2.res_w))
        pos_mat[xxs[0, 0], yys[0, 0]] = 1
        plt.imshow(pos_mat, cmap='gray')
        plt.show()
        
        
        
#         p = cam_coor[point_x, :].reshape(-1, 3).T
#         lam = E.dot(p).astype(np.float16)
#         dist = np.abs(cam2_coor.reshape(-1, 3).dot(lam))/np.expand_dims((lam[0, :]**2+lam[1, :]**2)**0.5, axis=0)

#         k = cam2.res_h+cam2.res_w-1
#         pps = np.argpartition(dist, k, axis=0)[:k].T
#         all_lines.append(pps.astype(np.int32))