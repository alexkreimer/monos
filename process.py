from __future__ import print_function
import os
import cv2
import matplotlib as mpl
print("using MPL version:", mpl.__version__)
mpl.use('TkAgg')

import numpy as np
from numpy import linalg as LA

import pylab
from matplotlib import pyplot as plt

from collections import namedtuple
from PIL import Image
import pickle
import multiprocessing as mp

class FeatureWarehouse(object):
    def __init__(self, corners):
        self.corners = corners

    def ft1(self, bins=5):
        '''floating bins followed by their edges '''

        deltas = self.corners[0].astype('int32') - self.corners[1].astype('int32')
        deltas = LA.norm(deltas, axis=1)
        
        h, edges = np.histogram(deltas, bins=5, density=True)
        #pylab.subplot(2,4,j*4+i+1)
        #plt.bar(edges[:-1], h, width = 1)
        #plt.xlim(min(edges), max(edges))
        
        # since the bins flat, we add edges
        h = np.hstack([np.nan_to_num(h), edges])
        return (h, '5bins_edges')

    def ft2(self, bins=200):
        '''uniform bins 0-200 '''

        deltas = self.corners[0].astype('int32') - self.corners[1].astype('int32')
        deltas = LA.norm(deltas, axis=1)
        
        h, edges = np.histogram(deltas, bins=range(bins), density=True)
        h = np.array(np.nan_to_num(h))
        return (h, '200bins')
    
    def generate(self):
        return [self.ft1(), self.ft2()]

class MatchDisplay(object):
    blend, montage = range(2)

    @staticmethod
    def showFeatures(im, kp):
        pylab.figure(figsize=(14,8))
        pylab.imshow(im, cmap=pylab.gray())
        pylab.scatter(kp[:,0], kp[:,1])
        pylab.show()

    @staticmethod
    def showMatchedFeatures(im1, im2, kp1, kp2, method = montage):
        im1 = Image.fromarray(im1)
        im2 = Image.fromarray(im2)
        
        w1, h1 = im1.size[:2]
        w2, h2 = im2.size[:2]
        
        w = max(w1,w2)
        h = h1 + h2
        im = Image.new('RGB', (w,h))

        im.paste(im1, (0, 0))
        im.paste(im2, (0, h1))

        fig = pylab.figure(figsize=(14,8))
        pylab.imshow(np.array(im))
        pylab.scatter(kp1[:,0], kp1[:,1])
        pylab.scatter(kp2[:,0], h1+kp2[:,1])

        F = pylab.gcf()
        DPI = F.get_dpi()
        print("DPI:", DPI)
        DefaultSize = F.get_size_inches()
        print("Default size in Inches", DefaultSize)
        print("Which should result in a %i x %i Image"%(DPI*DefaultSize[0], DPI*DefaultSize[1]))
        pylab.show()
        

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0]))
        p2.append(np.int32(np.array(kpp[1]) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)
    cv2.waitKey()

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(p1 - (x, y)) < r) | (anorm(p2 - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                 (x1, y1), (x2, y2) = p1[i], p2[i]
                 col = (red, green)[status[i]]
                 cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                 kp1, kp2 = kp_pairs[i]
                 kp1s.append(kp1)
                 kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv2.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    return vis

def func(a, img, win_size):
    ''' extract a patch from the image '''
    row_min, row_max = a[1]-win_size, a[1]+win_size
    col_min, col_max = a[0]-win_size, a[0]+win_size
    val = img[np.ix_(range(row_min, row_max+1), range(col_min, col_max+1))]
    return val.ravel()

def harris_gen(path, mask, first, last):
    for i in range(first, last+1):
        file_path = os.path.join(path, mask % i)
        #print('reading ', file_path)

        pil_image = Image.open(file_path).convert('RGB') 
        cv_image = np.array(pil_image) 
        cv_image = cv_image[:, :, ::-1].copy()

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # image size
        rows, cols = gray.shape

        # gray is currently 'uint8', convert it to 'float32'
        gray = np.float32(gray)

        # apply the detector
        #dst = cv2.cornerHarris(gray, blockSize=7, ksize=3, k=0.04)
        kp = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=.01, minDistance=3)

        kp = np.reshape(kp,(kp.shape[0],2))
        # compute corner indices
        #kp = np.transpose(np.nonzero(dst>0.01*dst.max()))

        # strip features too close to the edges, since we need to extract
        # patches
        thresh = 5
        good_x = np.logical_and(kp[:,0] > thresh, kp[:,0] < cols-thresh)
        good_y = np.logical_and(kp[:,1] > thresh, kp[:,1] < rows-thresh)

        # compute the keypoints
        kp = kp[np.logical_and(good_x, good_y)].astype('uint32')

        # compute the descriptors
        des = np.apply_along_axis(lambda x: func(x, gray, thresh), 1, kp)
        
        # return
        yield (kp, des, gray)

def read_scales(file):
    scales = []
    with open(file, 'r') as f:
        for line in f:
            pose = np.array([float(val) for val in line.split()]).reshape(3,4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            try:
                delta = np.dot(LA.inv(ppose), pose)
            except NameError:
                ppose = pose
                continue

            scale = LA.norm(delta[:3,3])
            scales.append(scale)

            ppose = pose
    return scales

def main(seq):
    KITTI_HOME = '/home/kreimer/KITTI'
    seq_dir = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq)
    image_0 = os.path.join(seq_dir, 'image_0')

    calib_file = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'calib.txt')
    
    scales = read_scales('data/%s.txt' % seq)
    scales = np.array(scales)

    grid   = (5,3)
    imsize = (1226, 370)
    # edges contain boundaries, i.e., the number of cells will be #-2
    xedges = np.linspace(0, imsize[0], num = grid[0])
    yedges = np.linspace(0, imsize[1], num = grid[1])

    with open(calib_file, 'r') as f:
        for line in f:
            line = [float(v) for v in line.split()[1:] ]
            K = np.array(line).reshape(3,4)
            break

    Intrinsic = namedtuple('intrinsic', ['f', 'pp'])
    intr = Intrinsic(f = K[0,0], pp=K[range(2),2])

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    HH = []
    num_frames = scales.shape[0]

    data = {}
    for key in data.keys():
        data[key].append(np.array([]))
    
    for frame, (kp2, des2, img2) in enumerate(harris_gen(image_0, '%06d.png', 0, num_frames)):
        print('seq %s: frame %d of %d' % (seq, frame, num_frames))
        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            m1 = np.array([m.queryIdx for m in matches])
            m2 = np.array([m.trainIdx for m in matches])

            # slice inliers
            kp1_matched = kp1[m1,:]
            kp2_matched = kp2[m2,:]

            des1_matched = des1[m1,:]
            des2_matched = des2[m2,:]

            #explore_match('win', img1, img2, zip(kp1, kp2))
            E, inliers = cv2.findEssentialMat(kp1_matched.astype('float32'),
                                              kp2_matched.astype('float32'),
                                              intr.f, tuple(intr.pp))
            # retval, R, t, mask = cv2.recoverPose(E, kp1, kp2)

            inliers = inliers.ravel().view('bool')
            kp1_matched = kp1_matched[inliers,:]
            kp2_matched = kp2_matched[inliers,:]

            des1_matched = des1_matched[inliers,:]
            des2_matched = des2_matched[inliers,:]

            #MatchDisplay.showFeatures(img2, kp2_matched)
            xv, yv = np.meshgrid(xedges, yedges, indexing='ij')
            H = None

            xbins = len(xedges)-1
            ybins = len(yedges)-1

            for k in data.keys():
                data[k].append(np.array([]))
                
            for i in range(xbins):
                for j in range(ybins):
                    xmin, xmax = xv[i,j], xv[i+1,j]
                    ymin, ymax = yv[i,j], yv[i,j+1]

                    xbin = np.logical_and(kp1_matched[:,0] > xmin, kp1_matched[:,0] <= xmax)
                    ybin = np.logical_and(kp1_matched[:,1] > ymin, kp1_matched[:,1] <= ymax)

                    val = np.logical_and(xbin, ybin)

                    # binned feature points
                    kp1_bin = kp1_matched[val,:]
                    kp2_bin = kp2_matched[val,:]

                    f = FeatureWarehouse((kp1_bin, kp2_bin)).generate()

                    for k, val in enumerate(f):
                        name = val[1]
                        value = val[0]

                        cur_val = data.setdefault(name, [np.array([])])
                        data[name][-1] = np.hstack([cur_val[-1], value])

            kp1, des1, img1 = kp2, des2, img2
        except NameError:
            kp1, des1, img1 = kp2, des2, img2

    scales = scales[:frame].reshape((frame,1))

    for k in data.keys():
        with open('data/%s_%s.pickle' % (seq, k), 'wb') as fd:
            pickle.dump({'X': np.array(data[k]), 'y': scales.ravel()}, fd)

if __name__ == '__main__':
    n = mp.cpu_count()
    pool = mp.Pool(processes = n-1)
    seqs = ['%02d' % s for s in range(11)]
    pool.map(main, seqs)
