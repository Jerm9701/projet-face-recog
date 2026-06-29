import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=9, height=7):
    """ Apply camera calibration operation for images in the given directory path. """
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)
    pattern_size = None
    objp = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Skipping unreadable image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if pattern_size is None:
            candidate_sizes = [(width, height), (width - 1, height - 1), (width - 1, height), (width, height - 1)]
            for candidate in candidate_sizes:
                if candidate[0] <= 0 or candidate[1] <= 0:
                    continue
                ret, corners = cv2.findChessboardCorners(gray, candidate, None)
                if ret:
                    pattern_size = candidate
                    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                    objp = objp * square_size
                    break

            if pattern_size is None:
                print(f"No chessboard pattern detected in {fname}")
                continue

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            display_img = img.copy()
            cv2.drawChessboardCorners(display_img, pattern_size, corners2, ret)
            cv2.imshow('Chessboard corners', display_img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if not imgpoints:
        raise ValueError("No chessboard corners were detected. Check the calibration images and board size.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def display_calibration_result(image_path, mtx, dist):
    """Show the first calibration image with detected corners and its undistorted version."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for pattern_size in [(9, 7), (8, 6), (7, 5), (6, 8)]:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            display_img = img.copy()
            cv2.drawChessboardCorners(display_img, pattern_size, corners2, ret)

            undistorted_img = cv2.undistort(img, mtx, dist)
            undistorted_with_corners = undistorted_img.copy()
            cv2.drawChessboardCorners(undistorted_with_corners, pattern_size, corners2, ret)
            combined_img = np.hstack((display_img, undistorted_with_corners))

            cv2.putText(combined_img, "Detected corners", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined_img, "Undistorted", (display_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.namedWindow("Calibration result", cv2.WINDOW_NORMAL)
            cv2.imshow("Calibration result", combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    raise ValueError("Could not detect chessboard corners in the first calibration image.")


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()

if __name__ == "__main__":
    dirpath = "F02-calibration-camera/calib_imgs"
    prefix = "calib_"
    image_format = "jpg"
    calibration = calibrate("F02-calibration-camera/calib_imgs", "calib_", "jpg", 0.025, 9, 7)
    save_coefficients(calibration[1], calibration[2], "F02-calibration-camera/camera_matrix.yml")

    calibration_images = sorted(glob.glob("F02-calibration-camera/calib_imgs/calib_*.jpg"))
    if calibration_images:
        display_calibration_result(calibration_images[0], calibration[1], calibration[2])