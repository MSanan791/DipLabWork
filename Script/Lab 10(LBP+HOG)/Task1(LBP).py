import numpy as np
import cv2
import matplotlib.pyplot as plt

uniform = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]

def calc_trans(vector):
    transitions = 0
    for i in range(1, len(vector)):
        transitions = transitions + 1 if vector[i-1] != vector[i] else transitions
    transitions = transitions + 1 if vector[0] != vector[len(vector)-1] else transitions

    return transitions

def calc_pattern(img):
    pattern = 0
    compared = [(img[0,0] > img[1,1]), (img[0,1] > img[1,1]), (img[0,2] > img[1,1]),
                (img[1,2] > img[1,1]), (img[2,2] > img[1,1]), (img[2,1] > img[1,1]),
                (img[2,0] > img[1,1]), (img[1,0] > img[1,1])]

    for i in range(len(compared)):
        pattern += compared[i]*np.power(2, i)

    return calc_trans(compared), pattern

def lbp(img):
    print("Inside LBP")
    hist = [0]*59
    [rows, cols] = np.shape(img)
    img_ret = np.zeros((rows, cols), np.uint8)
    img_mod = np.pad(img, ((1,1),(1,1)))

    for i in range(1, rows+1):
        for j in range(1, cols+1):
            [transitions, img_ret[i-1, j-1]] = calc_pattern(img_mod[i-1:i+2, j-1:j+2])
            if transitions <=2:
                hist[uniform.index(img_ret[i-1, j-1])] += 1
            else:
                hist[58] += 1

    return hist, img_ret

def calc_patternR(img):
    pattern = 0
    center = img[1,1]
    neighbors = [img[0,0], img[0,1], img[0,2], img[1,2], img[2,2], img[2,1], img[2,0], img[1,0]]
    compared = [int(n > center) for n in neighbors]

    d = np.argmax(neighbors)
    rotated = compared[d:] + compared[:d]

    for i in range(8):
        pattern += rotated[i] * np.power(2, 7 - i)

    return calc_trans(rotated), pattern

def lbpR(img):
    print("Inside LBPR")
    hist = [0]*59
    [rows, cols] = np.shape(img)
    img_ret = np.zeros((rows, cols), np.uint8)
    img_mod = np.pad(img, ((1,1),(1,1)))

    for i in range(1, rows+1):
        for j in range(1, cols+1):
            [transitions, img_ret[i-1, j-1]] = calc_patternR(img_mod[i-1:i+2, j-1:j+2])
            if transitions <=2:
                hist[uniform.index(img_ret[i-1, j-1])] += 1
            else:
                hist[58] += 1

    return hist, img_ret

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, img_lbp = lbp(frame)
    histR, img_lbpR = lbpR(frame)

    plt.subplot(2,1,1)
    plt.plot(hist)
    plt.subplot(2, 1, 2)
    plt.plot(histR)

    cv2.imshow("Original", frame)
    cv2.imshow("Rotated", img_lbpR)
    cv2.imshow('LBP', img_lbp)
    plt.show()
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from picamera import Picamera2
# import time
#
# uniform = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
#
# def calc_trans(vector):
#     transitions = 0
#     for i in range(1, len(vector)):
#         transitions = transitions + 1 if vector[i-1] != vector[i] else transitions
#     transitions = transitions + 1 if vector[0] != vector[-1] else transitions
#     return transitions
#
# def calc_pattern(img):
#     pattern = 0
#     compared = [(img[0,0] > img[1,1]), (img[0,1] > img[1,1]), (img[0,2] > img[1,1]),
#                 (img[1,2] > img[1,1]), (img[2,2] > img[1,1]), (img[2,1] > img[1,1]),
#                 (img[2,0] > img[1,1]), (img[1,0] > img[1,1])]
#     for i in range(len(compared)):
#         pattern += compared[i]*np.power(2, i)
#     return calc_trans(compared), pattern
#
# def lbp(img):
#     hist = [0]*59
#     [rows, cols] = np.shape(img)
#     img_ret = np.zeros((rows, cols), np.uint8)
#     img_mod = np.pad(img, ((1,1),(1,1)))
#     for i in range(1, rows+1):
#         for j in range(1, cols+1):
#             [transitions, img_ret[i-1, j-1]] = calc_pattern(img_mod[i-1:i+2, j-1:j+2])
#             if transitions <=2:
#                 hist[uniform.index(img_ret[i-1, j-1])] += 1
#             else:
#                 hist[58] += 1
#     return hist, img_ret
#
# def calc_patternR(img):
#     pattern = 0
#     center = img[1,1]
#     neighbors = [img[0,0], img[0,1], img[0,2], img[1,2], img[2,2], img[2,1], img[2,0], img[1,0]]
#     compared = [int(n > center) for n in neighbors]
#     d = np.argmax(neighbors)
#     rotated = compared[d:] + compared[:d]
#     for i in range(8):
#         pattern += rotated[i] * np.power(2, 7 - i)
#     return calc_trans(rotated), pattern
#
# def lbpR(img):
#     hist = [0]*59
#     [rows, cols] = np.shape(img)
#     img_ret = np.zeros((rows, cols), np.uint8)
#     img_mod = np.pad(img, ((1,1),(1,1)))
#     for i in range(1, rows+1):
#         for j in range(1, cols+1):
#             [transitions, img_ret[i-1, j-1]] = calc_patternR(img_mod[i-1:i+2, j-1:j+2])
#             if transitions <=2:
#                 hist[uniform.index(img_ret[i-1, j-1])] += 1
#             else:
#                 hist[58] += 1
#     return hist, img_ret
#
# # Initialize Pi Camera
# picam2 = Picamera2()
# picam2.configure(picam2.preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
# picam2.start()
#
# time.sleep(2)
#
# try:
#     while True:
#         frame = picam2.capture_array()
#         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#
#         hist, img_lbp = lbp(gray)
#         histR, img_lbpR = lbpR(gray)
#
#         # Plot LBP and Rotated LBP histograms
#         plt.figure(figsize=(8, 4))
#         plt.subplot(2,1,1)
#         plt.title("LBP Histogram")
#         plt.plot(hist)
#         plt.subplot(2, 1, 2)
#         plt.title("Rotated LBP Histogram")
#         plt.plot(histR)
#         plt.tight_layout()
#         plt.pause(0.001)
#         plt.clf()
#
#         # Show images
#         cv2.imshow("Original", gray)
#         cv2.imshow("LBP", img_lbp)
#         cv2.imshow("Rotated LBP", img_lbpR)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# except KeyboardInterrupt:
#     pass
#
# finally:
#     cv2.destroyAllWindows()
#     picam2.close()
#
#
