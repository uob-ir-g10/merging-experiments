import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

R_RESOLUTION = 1 # in px
THETA_RESOLUTION = np.pi/180 # in rads
CROP = 1500 # how many px's to crop off the edges


def houghSpectrum(img):
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(img, 206, 255, cv2.THRESH_BINARY)[1]

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, R_RESOLUTION, THETA_RESOLUTION, 50)

    counts = [0] * int(2 * np.pi / THETA_RESOLUTION)
    bin_edges = [x for x in np.arange(0, 2* np.pi, THETA_RESOLUTION)]

    assert(len(counts) == len(bin_edges))

    for r_theta in lines:
        r, theta = np.array(r_theta[0], dtype=np.float64)
        i = int(theta / THETA_RESOLUTION)
        i2 = int((np.pi + theta)/ THETA_RESOLUTION)
        counts[i] += 1
        counts[i2] += 1

    #plt.bar(bin_edges, counts, width=0.3)
    # plt.plot(bin_edges, counts)
    # plt.show()

    return counts

def circularCrossCorrelation(hs1, hs2):
    assert(len(hs1) == len(hs2))
    theta_s = int(2 * np.pi / THETA_RESOLUTION)
    ccc = [0] * theta_s
    for k in range(theta_s):
        for i in range(theta_s):
            ccc[k] += hs1[i] * hs2[(i + k) % theta_s]
    return ccc

def xyCrossCorrelation(spec1, spec2):
    assert(len(spec1) == len(spec2))
    corr = scipy.signal.correlate(spec1, spec2, mode='same')
    return np.argmax(corr) - (len(spec1) // 2)


def localMaxima(arr, n):
    arr = list(enumerate(arr))
    arr.sort(key=lambda x: x[1], reverse=True)
    return [np.pi + x[0] * THETA_RESOLUTION for x in arr[:n]]

def xySpec(occ_map):
    # slow
    h, w, _ = occ_map.shape
    yspec = [0] * h
    xspec = [0] * w

    for x in range(w):
        for y in range(h):
            # can remove this line if working with occupancy maps
            if not (occ_map[y, x, 0] in [0, 205, 254]):
                print(occ_map[y, x, 0])
            occupied = int(occ_map[y, x, 0] < 205)
            xspec[x] += occupied  
            yspec[y] += occupied  
    
    return xspec, yspec

def moveImg(img, angle, dx, dy):
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle*180/np.pi, scale=1)
    rotate_matrix = np.vstack([rotate_matrix, [0.0, 0.0, 1.0]])
    trans_matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    matrix = trans_matrix @ rotate_matrix
    rotated_image = cv2.warpPerspective(src=img, M=matrix, dsize=(width, height), flags=cv2.INTER_NEAREST)
    return rotated_image, matrix

def acceptanceIndex(img1, img2):
    # slow
    h, w, _ = img1.shape
    agreement = 0.0
    disagreement = 0.0

    for x in range(w):
        for y in range(h):
            # can remove this line if working with occupancy maps
            if img1[y, x, 0] == 205 or img2[y, x, 0] == 205:
                continue
            if img1[y, x, 0] == img2[y, x, 0]: 
                agreement += 1
            else:
                disagreement += 1
    
    return 0 if agreement == 0 else agreement / (agreement + disagreement)



data = []

for imgs in [('all_1.pgm', 'all_2.pgm'), ('almost_1.pgm', 'almost_2.pgm'),('split_1.pgm', 'split_2.pgm')]:
    print(imgs)
    img1 = cv2.imread(imgs[0])[CROP:-CROP, CROP:-CROP]
    img2 = cv2.imread(imgs[1])[CROP:-CROP, CROP:-CROP]
    hs1 = houghSpectrum(img1)

    # align coords for better performance
    best_rotation = localMaxima(hs1, 1)[0]
    img1, _ = moveImg(img1, best_rotation, 0, 0)

    hs1 = houghSpectrum(img1)
    hs2 = houghSpectrum(img2)

    ccc = circularCrossCorrelation(hs1, hs2)

    ## PARAMETERS (n, eta)
    N = 21
    eta = None
    ##    
    print(eta)
    maxima = localMaxima(ccc, N)
    xspec1, yspec1 = xySpec(img1)

    if eta != None:
        temp = []
        for angle in maxima:
            temp.append(angle)
            temp.append(angle + eta)
            temp.append(angle - eta)
        maxima = temp

    best_w = 0
    for i, angle in enumerate(maxima):
        t1 = time.time()
        img3, t = moveImg(img2, angle, 0, 0)
        xspec3, yspec3 = xySpec(img3)
        dx = xyCrossCorrelation(xspec1, xspec3)
        dy = xyCrossCorrelation(yspec1, yspec3)
        tm2, t = moveImg(img2, angle, dx, dy)
        w = acceptanceIndex(img1, tm2)
        if w > best_w:
            best_w = w
        print([i, best_w])
        data.append([i, best_w])

np.savetxt('results_no_eta.csv', data, delimiter=',', fmt='%.10f', header='n, w')