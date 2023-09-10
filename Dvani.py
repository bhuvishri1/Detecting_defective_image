import cv2
import matplotlib.pyplot as plt
import numpy as np

#to read the image from the file cv2.imread() method is used
img = cv2.imread(r"C:\Users\Home\Desktop\Dhvani\good (1).png")

img_dtype = img.dtype
print(img_dtype)
#creating a GUI window to display the image
#first parameter is windows title and second parameter is image array
#cv2.imshow("image",img)
#cv2.waitKey(0)
#First Parameter is for holding screen for specified milliseconds.
# It should be positive integer. If 0 is passed as an parameter, then it will
# hold the screen until user close it.

#using pyplot we can display the image
plt.imshow(img)
#hold the window
plt.waitforbuttonpress()
print(img.shape)

# def centre_coordinates(img):
#     #seperate row and column values
#     total_row,total_col,layers = img.shape

#     #get the center value of the image
#     x,y = np.ogrid[:total_row, :total_col]
#     cen_x , cen_y = total_row/2 , total_col/2
#     return cen_x , cen_y

# cc = centre_coordinates(img) 
# print(cc)
# #cc is centre co-odinates of good image

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (9, 9), 2)



# #Using Hough Circle Transform to detect circles.
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT,1.5,100)

print(circles)
if circles is not None:
    # Convert circle coordinates and radius to integers
    circles = np.round(circles[0,:]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    # Draw the circles on the original image (for visualization)
    for (x, y, r) in circles:
         cv2.circle(img, (x, y),r, (0, 255, 0), 2)  # Green for outer circle
         #cv2.circle(img, (x, y),r, (0, 0, 255), 2)  # Red for inner circle


    # Display the image with detected circles
    cv2.imshow("Image with Detected Circles", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the coordinates and radii of the inner and outer circles
    print("Detected Circles:")
    for (x,y,r) in circles:
     print("Center:", (x, y), "Radius:", r)
    

else:
     print("NO circles Detected")

#we have detected outer circle now
#counting black pixels
no_of_black_pixels = np.sum(img ==0)
print(no_of_black_pixels)

outer_circle_centre = (587,549)
outer_circle_radius = 520


# Create a binary mask for the detected circle


circle_mask = np.zeros_like(img, dtype=np.uint8)
cv2.circle(circle_mask, outer_circle_centre, outer_circle_radius, 255, -1)  # Filled circle

# Convert the region outside the detected circle to black (pixel value 0)
img_outside_circle = img.copy()  # Create a copy of the original image
img_outside_circle[circle_mask != 255] = 0  # Set pixels outside the circle to 0

# Display or save the modified image
cv2.imshow("Image with Outside Circle Set to Black", img_outside_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()

white_pixels = np.sum(img_outside_circle==255)
print(white_pixels)

#now that we have calculated white pixels in the good image
#lets load the defect image and check for pixels

defect_img = cv2.imread(r"C:\Users\Home\Desktop\Dhvani\defect1.png")
defect_img_dtype = img.dtype
print(defect_img_dtype)
plt.imshow(defect_img)
#hold the window
plt.waitforbuttonpress()
print(defect_img.shape)

#getting difference between good image and defect image
difference_image = cv2.absdiff(img,defect_img)
print(difference_image)

threshold_value = 50  # Adjust this threshold as needed
thresholded_image = cv2.threshold(difference_image, threshold_value, 255, cv2.THRESH_BINARY)[1]

defective_pixel_count = np.sum(thresholded_image == 255)

#Define a threshold for classifying the image as defective (adjust as needed)
defective_threshold = 1000  # Adjust this threshold as needed

if defective_pixel_count > defective_threshold:
    print("Defective Image")
else:
    print("Good Image")

cv2.imshow("Difference Image", difference_image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#this way we can calculate the defects like flash and cuts in ann image
defect_img1= cv2.imread(r"C:\Users\Home\Desktop\Dhvani\defect2.png")
defect_img1_dtype = img.dtype
print(defect_img1_dtype)
plt.imshow(defect_img1)
#hold the window
plt.waitforbuttonpress()

#showing the difference of good image and cut image
difference_image1 = cv2.absdiff(img,defect_img1)
print(difference_image1)
cv2.imshow("Difference Image", difference_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()



































