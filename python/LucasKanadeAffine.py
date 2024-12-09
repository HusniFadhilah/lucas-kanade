import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    p = np.zeros((6, 1), dtype=np.float32)
    dp = np.zeros((6, 1), dtype=np.float32)
    thr = 0.1

    M = np.eye(3, dtype=np.float32) # (3, 3) get initial M from initial p 

    It1_h, It1_w = It1.shape[0:2]
    It_h, It_w = It.shape[0:2]

    It1_xs = np.arange(0, It1_w, 1, dtype=np.float32)
    It1_ys = np.arange(0, It1_h, 1, dtype=np.float32)
    It1_xs, It1_ys = np.meshgrid(It1_xs, It1_ys)
    It1_xs, It1_ys = It1_xs.reshape(-1), It1_ys.reshape(-1)
    It1_ones = np.ones((It1_xs.shape[0],), dtype=np.float32)

    all_It1_points = np.stack((It1_xs, It1_ys, It1_ones), axis=0) # get all point coordinate from It1, homogeneous, (3, 1)

    It_intp = get_intp_img(It)
    It1_intp = get_intp_img(It1)
    pixel_diffs = []
    while True:
        # update M
        M[0, 0:3] += dp[0:3, 0]
        M[1, 0:3] += dp[3:6, 0]
        
        # (3, N_It1)
        all_It1_points_back_project = np.linalg.inv(M) @ all_It1_points
        
        x_inbound = (0 <= all_It1_points_back_project[0, :]) * (all_It1_points_back_project[0, :] < It_w)
        y_inbound = (0 <= all_It1_points_back_project[1, :]) * (all_It1_points_back_project[1, :] < It_h)
        point_inbound = x_inbound * y_inbound

        # (3, N) filter out backprojected points that falls out It 
        all_It1_points_in_It = all_It1_points_back_project[:, point_inbound] 
        
        # (3, N)
        template_points = all_It1_points_in_It 
        search_points = M @ template_points

        # points to row and col ids
        template_row_ids, template_col_ids = template_points[1, :], template_points[0, :]
        search_row_ids, search_col_ids = search_points[1, :], search_points[0, :]

        # get search pixels and template pixels
        # (N, )
        template_pixels = It_intp.ev(template_row_ids, template_col_ids)
        search_pixels = It1_intp.ev(search_row_ids, search_col_ids)

        # evaluate difference and print
        pixel_diff = np.sum((template_pixels - search_pixels)**2)
        # print('difference with template: %f' % pixel_diff)
        pixel_diffs.append(pixel_diff)

        It1_dx = It1_intp.ev(search_row_ids, search_col_ids, dx=0, dy=1) # (N, )
        It1_dy = It1_intp.ev(search_row_ids, search_col_ids, dx=1, dy=0) # (N, )

        # (2N, 1) Construct from It1_dx and It1_dy
        A_part1 = np.stack((It1_dx, It1_dy), axis=1) # (N, 2)
        A_part1 = A_part1.reshape(-1, 1) # (2N, 1) 

        # (2N, 6) Construct from template points
        A_part2_zeros = np.zeros((template_points.shape[1], template_points.shape[0]), dtype=np.float32) # (N, 3)
        A_part2 = np.concatenate((template_points.transpose(), A_part2_zeros, A_part2_zeros, template_points.transpose()), axis=1) # (N, 12)
        A_part2 = A_part2.reshape(-1, 6) # (2N, 6)

        # assemble A (N, 6)
        A = A_part1 * A_part2 # (2N, 6)
        A = A[0::2, :] + A[1::2, :] # (N, 6)

        # (N, 1) It(template points) - It1(search points)
        b = template_pixels - search_pixels # (N, )
        b = b.reshape(-1, 1) # (N, 1)
        
        dp_LSE = np.linalg.inv((A.transpose() @ A)) @ A.transpose() @ b
        
        if np.sum((dp_LSE - dp)**2) < thr:
            break
        dp = dp_LSE
    avg_pixel_diff = np.mean(pixel_diffs)
    # remove last row from M
    return M[0:2, :], avg_pixel_diff


import numpy as np
from scipy.interpolate import RectBivariateSpline

def indice_from_rect(rect, row_sample_num=None, col_sample_num=None):
    # rect: tuple, (l, t, r, b)
    # return: np float arrays of shape (N, 1), row_ids, col_ids 
    # fractional coordinates will be handled by interpolation
    l, t, r, b = rect
    # inclusive
    row_min = t
    row_max = b
    col_min = l
    col_max = r
    if row_min > row_max or col_min > col_max:
        return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.int32)

    if row_sample_num is None:
        row_ids = np.arange(row_min, row_max+1, 1, dtype=np.float32)
    else:
        assert row_sample_num > 2
        step = (row_max - row_min) / (row_sample_num - 1.0)
        row_ids = np.arange(row_min, row_max+step/2.0, step, dtype=np.float32)

    if col_sample_num is None:
        col_ids = np.arange(col_min, col_max+1, 1, dtype=np.float32)
    else:
        assert col_sample_num > 2
        step = (col_max - col_min) / (col_sample_num - 1.0)
        col_ids = np.arange(col_min, col_max+step/2.0, step, dtype=np.float32)

    col_ids_all, row_ids_all = np.meshgrid(col_ids, row_ids)
    row_ids_all, col_ids_all = row_ids_all.reshape(-1), col_ids_all.reshape(-1)
    return row_ids_all, col_ids_all


def get_intp_img(img):
    # img: np array, (h, w)
    # return: RectBivariateSpline, able to access
    # pixel at fractional coordinate; Out-of-bound
    # pixel values equal to nearest boundary values.
    h, w = img.shape[:2]
    row_ids = np.arange(0, h, 1, dtype=np.float32)
    col_ids = np.arange(0, w, 1, dtype=np.float32)
    img_intp = RectBivariateSpline(row_ids, col_ids, img, kx=2, ky=2)
    return img_intp