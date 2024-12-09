import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeRobust(It, It1, rect, p0=np.zeros(2)):
    """
    Lucas-Kanade tracker with illumination robustness.
    """
    threshold = 0.1
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    rows_rect, cols_rect = y2 - y1, x2 - x1
    dp = np.array([np.inf, np.inf])  # Initialize with large values to start loop

    # Precompute template gradients
    Iy, Ix = np.gradient(It1)
    y = np.arange(It.shape[0])
    x = np.arange(It.shape[1])
    spline = RectBivariateSpline(y, x, It)
    T = spline.ev(np.linspace(y1, y2, rows_rect), np.linspace(x1, x2, cols_rect))
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)
    spline1 = RectBivariateSpline(y, x, It1)

    # Initialize weights
    weights = np.ones((rows_rect, cols_rect))

    while np.linalg.norm(dp) > threshold:
        x1_w, y1_w, x2_w, y2_w = x1 + p0[0], y1 + p0[1], x2 + p0[0], y2 + p0[1]
        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)
        warpImg = spline1.ev(rrw, ccw)

        # Compute error
        error = T - warpImg

        # Update weights using Tukey's M-estimator
        c = 4.685  # Tunable parameter for robustness
        weights = (np.abs(error) < c) * ((1 - (error / c)**2)**2)

        # Compute weighted Jacobian
        Ix_w = spline_gx.ev(rrw, ccw)
        Iy_w = spline_gy.ev(rrw, ccw)
        grad = np.vstack((Ix_w.ravel(), Iy_w.ravel())).T
        J = grad @ np.array([[1, 0], [0, 1]])

        # Weighted Hessian
        W = np.diag(weights.ravel())
        H = J.T @ W @ J

        # Solve for dp
        b = W @ error.ravel()
        dp = np.linalg.inv(H) @ J.T @ b

        # Update parameters
        p0 += dp

    return p0
