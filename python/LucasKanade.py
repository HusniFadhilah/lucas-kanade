import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    threshold = 0.1
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    rows_img, cols_img = It.shape
    rows_rect, cols_rect = x2 - x1, y2 - y1
    dp = [[cols_img], [rows_img]] #just an intial value to enforce the loop

    # template-related can be precomputed
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)     
    c = np.linspace(x1, x2, cols_rect)
    r = np.linspace(y1, y2, rows_rect)
    cc, rr = np.meshgrid(c, r)
    spline = RectBivariateSpline(y, x, It)
    T = spline.ev(rr, cc)
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)
    spline1 = RectBivariateSpline(y, x, It1)

    # in translation model jacobian is not related to coordinates
    jac = np.array([[1,0],[0,1]])
    pixel_diffs = []
    while np.square(dp).sum() > threshold:
            
        # warp image using translation motion model
        x1_w, y1_w, x2_w, y2_w = x1+p0[0], y1+p0[1], x2+p0[0], y2+p0[1]
        
        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)
        
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        pixel_diffs.append(err)
        #compute gradient
        Ix_w = spline_gx.ev(rrw, ccw)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #computer Hessian
        delta = I @ jac 
        #H is (2,2)
        H = delta.T @ delta
        
        #compute dp
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg
        
        #update parameters
        p0[0] += dp[0,0]
        p0[1] += dp[1,0]
    avg_pixel_diff = np.mean(pixel_diffs)
    return p0[0],p0[1],avg_pixel_diff

