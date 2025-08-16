import numpy as np
import cv2

def compute_pix_to_meter(src_pts, real_distance_m):
    # src_pts: 2 điểm pixel có khoảng cách thực known
    p1, p2 = np.array(src_pts[0]), np.array(src_pts[1])
    pix_dist = np.linalg.norm(p1 - p2)
    return real_distance_m / max(pix_dist, 1e-6)  # m/pixel
