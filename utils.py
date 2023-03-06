import numpy as np
import cv2
from math import sin, cos


def anno2coords(d, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    ret = []
    for i in np.array(d.split()).reshape(-1, 7):
        ret.append(dict(zip(names, i.astype('float'))))
    return ret
def coords2imgcoords(coords, intrinsic_matrix):
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T    # (n,3)->(3,n)
    img_p = np.dot(intrinsic_matrix, P).T    # (3,3)@(3,n) -> (3,n)
    # Intrinsic matrix @ 3d point
     
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image
def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image
# def draw_text(image, points, data):
    # for (p_x, p_y, p_z) in zip(points, ):
# convert euler angle to rotation matrix
# product the pitch, raw, and roll maxtix
def euler_to_Rot(yaw, pitch, roll): 
    X_p = np.array([[1,            0,           0],
                    [0,   cos(pitch), -sin(pitch)],
                    [0,   sin(pitch),  cos(pitch)]])
    
    Y_y = np.array([[cos(yaw),    0,  sin(yaw)],
                    [        0,   1,         0],
                    [-sin(yaw),   0,  cos(yaw)]])

    Z_r = np.array([[cos(roll),   -sin(roll), 0],
                    [sin(roll),    cos(roll), 0],
                    [        0,            0, 1]])
    return np.dot(Y_y, np.dot(X_p, Z_r))
def visualize(img, coords, intrinsic_matrix):
    # Wolrd coordinate 상의 객체의 지점(mm)
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        # Extrinsic matrix, (4, 4)
        
        P = np.array([[x_l, -y_l, -z_l, 1],  
                      [x_l, -y_l, z_l, 1],   
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1], # four points for drawing box of a car
                      [0, 0, 0, 1]]).T       # No effective angle elements for drawing a center point of object 
        # wolrd position, (4, 1)
        
        # Intrinsic matrix @ Extrinsic matrix @ world potision
        img_cor_points = np.dot(intrinsic_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img
