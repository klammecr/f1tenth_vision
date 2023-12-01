# Third Party
import cv2
import os
import numpy as np

def get_corners(sheet_w, sheet_h, grid_size = (6,8)):
    img_pts = []
    objpts  = []
    for f in os.listdir("calibration"):
        img = cv2.imread(f"calibration/{f}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(gray, grid_size, None)
        img_pts.append(corners)

        # Visualization:
        #if retval:
        #     img = cv2.drawChessboardCorners(img, (6,8), grid_size, True)
        #     cv2.imshow("Corners", img)
        #     cv2.waitKey()
        # else:
        #     print("No Corners :(")

        # Find step sizes
        stepx = sheet_w/grid_size[0]
        stepy = sheet_h/grid_size[1]

        # Calculate the metric points
        objp = np.zeros((6*8,3), np.float32)
        y, x = np.mgrid[:grid_size[1], :grid_size[0]]
        objp[:,:2] = np.stack((x.flatten(), y.flatten())).T * np.array([stepx, stepy]).reshape(1,2)
        objpts.append(objp.astype("float32"))

    return img_pts, objpts, gray.shape[1], gray.shape[0]

def get_mount_h(K, pix_coord, depth):
    pt = np.array([pix_coord[0], pix_coord[1], 1])
    point3d = depth * (np.linalg.inv(K) @ pt)
    mount_h = point3d[1]
    return mount_h

def calc_dist(K, pix_coord, mount_h, other_depth):
    pt = np.array([pix_coord[0], pix_coord[1], 1])
    met = np.linalg.inv(K) @ pt
    depth = other_depth * (met[1]/mount_h)
    pt_depth_scale = met * depth
    car_x = pt_depth_scale[-1]
    car_y = -pt_depth_scale[0]
    return car_x, car_y

if __name__ == "__main__":
    # Get corners on the image
    grid_size = (6,8)

    # Get the 3D points
    sheet_h = .25
    sheet_w = sheet_h * (grid_size[0]/grid_size[1])

    # Get the corners
    corners, objpts, img_w, img_h = get_corners(sheet_w, sheet_h, grid_size)

    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[:8,:6].T.reshape(-1,2)
    objpoints = [objp] * len(corners)

    # Calibrate the camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, corners, (img_w, img_h), None, None)
    print(f"Camera Matrix: {K}")
    np.save("camera_matrix.npz", K)

    # Distance measurement
    # Calculate the extrinsic of the camera
    cone_img = cv2.imread("resource/cone_x40cm.png")
    cone_dist = 0.4 # meters
    extr_h = get_mount_h(K, (661,493), cone_dist)
    
    cone_img_unknown = cv2.imread("resource/cone_unknown.png")
    dist = calc_dist(K, (594, 415), extr_h, cone_dist)
    print(f"Distance to the Unknown Cone is: {dist} meters")
