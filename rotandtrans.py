
import cv2
import numpy as np
 
#try to map object points to image points based on physical measurements and pixel location in the frame
 
# Read Image
im = cv2.imread("c:/users/peter/desktop/ht_validation/aug3captures/rightCamtake1frame233.jpg");
size = im.shape
     
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (193,288),     # left frame reflector
                            (287,287),    # right frame reflector
                            (288, 282),    # back frame reflector
                            (258, 231),     # ball ctr
                            (252,231),     # ball left
                            (266, 231)     # ball right
                        ], dtype="double")
 
# 3D object points.
object_points = np.array([
                            (875.0, 762.0, 2621.4), # left frame reflector
                            (1275.0, 762.0, 2621.4), # right frame reflector
                            (1275.0, 762.0, 2921.4), # back frame reflector
                            (1422.4, 1041.4, 2884.57),   # ball ctr
                            (1385.57, 1041.4, 2884.57),  # ball left
                            (1459.23, 1041.4, 2884.57)  # ball right
                        ])
 
 
# Camera internals
 
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[602.3300926, 0, 314.00816007],
                         [0, 601.89164578, 265.7839791],
                         [0, 0, 1]], dtype = "double"
                         )
 
print "Camera Matrix :\n {0}".format(camera_matrix)
 
#dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
dist_coeffs = np.array([-0.13277752, 0.24566243, 0.00257889, -0.00288082, -0.35018818]) # lens distortion

(success, rotation_vector, translation_vector) = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
 
print "Rotation Vector:\n {0}".format(rotation_vector)
print "Translation Vector:\n {0}".format(translation_vector)
 
(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(1385.57, 1041.4, 2884.57)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
print int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])


p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
cv2.line(im, p2, p2, (255,0,0), 2)
 
# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
