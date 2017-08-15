import numpy as np
import cv2

numBoards = 50  #how many boards would you like to find
board_w = 9
board_h = 6

board_sz = (9,6)
board_n = board_w*board_h

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = [] # 3d point in real world space
imagePoints1 = [] # 2d points in image plane.
imagePoints2 = [] # 2d points in image plane.

corners1 = []
corners2 = []

#obj = []
#for j in range(0,board_n):
    #obj.append(np.(j/board_w, j%board_w, 0.0))
obj = np.zeros((6*9,3), np.float32)
obj[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

cameraL = cv2.VideoCapture('C:/Users/peter/Desktop/ht_validation/calibration/intrinsic cal-left cam.avi')
if cameraL.isOpened() == False:
        print "Cannot open input video!"
        exit()
cameraL.set(1, 100) #CV_CAP_PROP_POS_FRAMES

cameraR = cv2.VideoCapture('C:/Users/peter/Desktop/ht_validation/calibration/intrinsic cal-right cam.avi')
if cameraR.isOpened() == False:
        print "Cannot open input video!"
        exit()
cameraR.set(1, 100) #CV_CAP_PROP_POS_FRAMES

success = 0
k = 0
found1 = False
found2 = False

while (success < numBoards):

   # grab the left frame
   (grabbed, img1) = cameraL.read()
   height, width, depth  = img1.shape
   # grab the right frame
   (grabbed, img2) = cameraR.read()
   gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
   gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

   found1, corners1 = cv2.findChessboardCorners(img1, board_sz)
   found2, corners2 = cv2.findChessboardCorners(img2, board_sz)

   if (found1):
       cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),criteria)
       cv2.drawChessboardCorners(gray1, board_sz, corners1, found1)

   if (found2):
       cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
       cv2.drawChessboardCorners(gray2, board_sz, corners2, found2)

   cv2.imshow('image1', gray1)
   cv2.imshow('image2', gray2)

   k = cv2.waitKey(10000)
   if (k == 27): #ESC key
       break
   if (k == 32 and found1 != 0 and found2 != 0):

       imagePoints1.append(corners1);
       imagePoints2.append(corners2);
       object_points.append(obj);
       print "Corners stored\n"
       success+=1

       if (success >= numBoards):
           break

cv2.destroyAllWindows()
cameraL.release()
cameraR.release()
print "Starting Calibration\n"

cameraMatrix1 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)
cameraMatrix2 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)

#cameraMatrix1 = np.array(
#                         [[610.07804118, 0, 314.48633303],
#                         [0, 609.62733791, 265.13152021],
#                         [0, 0, 1]], dtype = "double"
#                         )

#distCoeffs1 = np.array([-0.14480132, 0.28330372, 0.00185214, -0.00103649, -0.33967936]) # lens distortion

#cameraMatrix2 = np.array(
#                         [[602.3300926, 0, 314.00816007],
#                         [0, 601.89164578, 265.7839791],
#                         [0, 0, 1]], dtype = "double"
#                         )
 
#distCoeffs2 = np.array([-0.13277752, 0.24566243, 0.00257889, -0.00288082, -0.35018818]) # lens distortion


retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, (width, height),
                                                                                                 cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                                                                                 flags=cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_SAME_FOCAL_LENGTH,
                                                                                                 criteria = criteria)
## , cv2.cvTermCriteria(cv2.CV_TERMCRIT_ITER+cv2.CV_TERMCRIT_EPS, 100, 1e-5),   cv2.CV_CALIB_SAME_FOCAL_LENGTH | cv2.CV_CALIB_ZERO_TANGENT_DIST)
#cv2.cv.StereoCalibrate(object_points, imagePoints1, imagePoints2, pointCounts, cv.fromarray(K1), cv.fromarray(distcoeffs1), cv.fromarray(K2), cv.fromarray(distcoeffs2), imageSize, cv.fromarray(R), cv.fromarray(T), cv.fromarray(E), cv.fromarray(F), flags = cv.CV_CALIB_FIX_INTRINSIC)
#FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
# fs1 << "CM1" << CM1;
#fs1 << "CM2" << CM2;
# #fs1 << "D1" << D1;
#fs1 << "D2" << D2;
#fs1 << "R" << R;
#fs1 << "T" << T;
#fs1 << "E" << E;
#fs1 << "F" << F;
print "Done Calibration\n"
print "matrix1\n",cameraMatrix1,"\n"
print distCoeffs1,"\n"
print "matrix2\n",cameraMatrix2,"\n"
print distCoeffs2,"\n"
print "R\n",R,"\n"
print "T\n",T,"\n"
print "Starting Rectification\n"
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

print object_points

#(roi1, roi2) = cv2.cv.StereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))
cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))
#stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T)
print "R1\n",R1,"\n"
print "P1\n",P1,"\n"
print "R2\n",R2,"\n"
print "P2\n",P2,"\n"

#fs1 << "R1" << R1;
#fs1 << "R2" << R2;
#fs1 << "P1" << P1;
#fs1 << "P2" << P2;
#fs1 << "Q" << Q;

print "Done Rectification\n"
print "Applying Undistort\n"



map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)

print "Undistort complete\n"

cameraL = cv2.VideoCapture('C:/Users/peter/Desktop/ht_validation/calibration/intrinsic cal-left cam.avi')
if cameraL.isOpened() == False:
        print "Cannot open input video!"
        exit()
cameraL.set(1, 100) #CV_CAP_PROP_POS_FRAMES

cameraR = cv2.VideoCapture('C:/Users/peter/Desktop/ht_validation/calibration/intrinsic cal-right cam.avi')
if cameraR.isOpened() == False:
        print "Cannot open input video!"
        exit()
cameraR.set(1, 100) #CV_CAP_PROP_POS_FRAMES

while(True):
    # grab the left frame
    (grabbed, img1) = cameraL.read()
    # grab the right frame
    (grabbed, img2) = cameraR.read()

    imgU1 = np.zeros((height,width,3), np.uint8)
    imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imshow("imageL", img1);
    cv2.imshow("imageR", img2);
    cv2.imshow("image1L", imgU1);
    cv2.imshow("image2R", imgU2);
    k = cv2.waitKey(1000);
    if(k==27):
        break;

cv2.destroyAllWindows()
cameraL.release()
cameraR.release()
