imgThreshold
cv2.Canny(picture,x,y)
x seems to reduce the pickup of reflection the more it goes up, cranking it up to extreme values it makes it pick up the most obvious edges which is ideal for what im doing, 1000 seems to be the upper limit, past that the shape of the can becomes too hard to recognize, in fact with the rs4.jpg sample, I only seem to be able to get a decent result with x=300

y seems to reduce the sharpness of the edges of the image, 210 seems to be the ideal value when finding a balance between just finding the edges of the can and finding the edges of the text, a higher value of y results in unlegible edges, making the shape recognition useless.
