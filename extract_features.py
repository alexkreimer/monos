
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib tk')


# In[2]:

import os, sys, math, errno, glob, warnings, traceback, pickle
import numpy as np
from numpy import linalg as LA
from collections import namedtuple
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker
from matplotlib import cm
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

import multiprocessing as mp
from progressbar import Percentage, Bar, ProgressBar
from time import sleep

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

KITTI_HOME = '/home/kreimer/KITTI'

Intrinsic = namedtuple('Intrinsic', ['f', 'pp'])


# In[3]:

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

def read_poses(file):
    poses = []
    with open(file, 'r') as f:
        for line in f:
            pose = np.array([float(val) for val in line.split()]).reshape(3,4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses


# In[4]:

def read_image(seq, frame_num, mono=True):
    global KITTI_HOME
    
    file1_path = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'image_0', '%06d.png' % frame_num)
    
    pil_image = Image.open(file1_path).convert('RGB') 
    cv_image = np.array(pil_image) 
    cv_image = cv_image[:, :, ::-1].copy()
    gray1 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if mono:
        return gray1
    
    file2_path = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'image_1', '%06d.png' % frame_num)
    
    pil_image = Image.open(file2_path).convert('RGB') 
    cv_image = np.array(pil_image) 
    cv_image = cv_image[:, :, ::-1].copy()
    gray2 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return (gray1, gray2)


# In[5]:

def patch_extractor(a, img, win_size):
    ''' extract a patch from the image '''
    row_min, row_max = a[1]-win_size, a[1]+win_size
    col_min, col_max = a[0]-win_size, a[0]+win_size
    val = img[np.ix_(range(row_min, row_max+1), range(col_min, col_max+1))]
    return val.ravel()


# In[6]:

def harris(gray):
    # image size
    rows, cols = gray.shape

    # gray is currently 'uint8', convert it to 'float32'
    gray = np.float32(gray)

    # apply the detector
    kp = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=.01, minDistance=3)
    kp = np.reshape(kp,(kp.shape[0],2))

    # strip features too close to the edges, since we need to extract
    # patches
    thresh = 5
    good_x = np.logical_and(kp[:,0] > thresh, kp[:,0] < cols-thresh)
    good_y = np.logical_and(kp[:,1] > thresh, kp[:,1] < rows-thresh)

    # compute the keypoints
    kp = kp[np.logical_and(good_x, good_y)].astype('uint32')

    # compute the descriptors
    des = np.apply_along_axis(lambda x: patch_extractor(x, gray, thresh), 1, kp)

    return (kp, des)


# In[7]:

def match(des1, des2):
    # brute force L1 matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches


# In[8]:

def prune_matches(matches, kp1, kp2, des1, des2, intr):
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
    
    return (kp1_matched, des1_matched, kp2_matched, des2_matched)


# In[9]:

def compute_hist(corners, nbins):
    '''uniform bins 0-200 '''
    deltas = corners[0].astype('int32') - corners[1].astype('int32')
    deltas = LA.norm(deltas, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h, edges = np.histogram(deltas, bins=range(nbins+1), density=True)
    h = np.array(np.nan_to_num(h))
    return (h, edges)

def compute_features(kp1, kp2, imsize=(1226, 370), grid=(4,3), nbins=50):
    # edges contain boundaries, i.e., the number of cells will be #-2
    xedges = np.linspace(0, imsize[0], num = grid[0]+1)
    yedges = np.linspace(0, imsize[1], num = grid[1]+1)

    xv, yv = np.meshgrid(xedges, yedges, indexing='ij')
    
    k = 0
    desc = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            xmin, xmax = xv[i,j], xv[i+1,j]
            ymin, ymax = yv[i,j], yv[i,j+1]

            xbin = np.logical_and(kp1[:,0] > xmin, kp1[:,0] <= xmax)
            ybin = np.logical_and(kp1[:,1] > ymin, kp1[:,1] <= ymax)

            val = np.logical_and(xbin, ybin)

            # binned feature points
            kp1_bin = kp1[val,:]
            kp2_bin = kp2[val,:]
            (h, edges) = compute_hist((kp1_bin, kp2_bin), nbins=nbins)
            desc.append(h)
    return desc


# In[10]:

def process_pair(image_0, pair, nbins=150, grid=(6,4), imsize=(1226,370), debug=False):
    ''' compute optical flow features for a pair of images '''
    dpi = 20.
    stride = (float(imsize[0])/grid[0], float(imsize[1])/grid[1])

    locx = plticker.MultipleLocator(base=stride[0])
    locy = plticker.MultipleLocator(base=stride[1])
    cmap = plt.get_cmap('jet')

    image1 = read_image(image_0, '%06d.png', pair[0])
    image2 = read_image(image_0, '%06d.png', pair[1])
    (kp1, des1) = harris(image1)
    (kp2, des2) = harris(image2)

    matches = match(des1, des2)
    
    kp1, des1, kp2, des2 = prune_matches(matches, kp1, kp2, des1, des2, intr)
    N = len(kp1)
    
    h = compute_features(kp1, kp2, imsize, grid, nbins=nbins)

    if debug:
        (hist, edges) = compute_hist((kp1, kp2), nbins=200)
        fig = plt.figure()
        plt.bar(range(200), hist)
    
        fig = plt.figure(figsize=(float(imsize[0])/dpi,float(imsize[1])/dpi), dpi=dpi)
        ax = fig.add_subplot(111, xlim=(0,imsize[0]), ylim=(imsize[1],0))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
        ax.xaxis.set_major_locator(locx)
        ax.yaxis.set_major_locator(locy)

        # Add the grid
        ax.grid(which='major', axis='both', linestyle='-', linewidth=3)

        plt.imshow((image1.astype('int32')+image2.astype('int32'))/2, cmap='gray')
        d = N*[0]
        for i in range(N):
            x1, x2 = kp1[i,0], kp2[i,0]
            y1, y2 = kp1[i,1], kp2[i,1]
            dx, dy = x1-x2, y1-y2
        
            dist = math.sqrt(dx*dx + dy*dy)
            if dist>nbins:
                dist = nbins
            d[i] = dist

        maxd, mind = max(d), min(d)

        a = 1.0/(maxd-mind)
        b = -a*mind

        for i in range(N):
            plt.plot([kp1[i,0], kp2[i,0]], [kp1[i,1], kp2[i,1]], linewidth=2, color=cm.jet(a*d[i]+b))
        plt.figure()
        f, axarr = plt.subplots(grid[1], grid[0], sharex='col', sharey='row')
        for i in range(grid[0]):
            for j in range(grid[1]):
                axarr[j, i].bar(range(nbins), h[i*grid[1]+j])

    return np.array(h).ravel()


# In[11]:

def compute_dt(pairs, poses):
    dts = []
    for pair in pairs:
        pose1 = poses[pair[1]]
        pose0 = poses[pair[0]]
        delta = np.dot(LA.inv(pose0), pose1)
        dt = delta[0:3,3]
        dts.append(dt)
    return dts


# In[12]:

def do_work(x):
    #print('pair: (%d, %d), grid=(%d, %d), nb=%d' % (x[1][0], x[1][1], x[2][0], x[2][1], x[3]))
    with np.errstate(divide='ignore'):
        return process_pair(x[0], x[1], grid=x[2], nbins=x[3])

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# In[13]:

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback.print_stack()
    log = file if hasattr(file,'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


# In[22]:

def extract_and_match(seq, pair, intr):
    ''' compute optical flow features for a pair of images '''
    
    image1 = read_image(seq, pair[0])
    image2 = read_image(seq, pair[1])
    
    (kp1, des1) = harris(image1)
    (kp2, des2) = harris(image2)
    
    matches = match(des1, des2)
    
    kp1, des1, kp2, des2 = prune_matches(matches, kp1, kp2, des1, des2, intr)
    return (kp1, kp2)


# In[33]:

def seq_len(seq):
    global KITTI_HOME
    
    return(len(glob.glob(os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'image_0', '*.png'))))

def read_intrinsics(seq):
    global KITTI_HOME

    calib_file = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'calib.txt')
    with open(calib_file, 'r') as f:
        for line in f:
            line = [float(v) for v in line.split()[1:] ]
            K = np.array(line).reshape(3,4)
            break
    intr = Intrinsic(f = K[0,0], pp=K[range(2),2])
    return intr

def extract_and_match_wrapper(task):
    val = extract_and_match(task[0], task[1], task[2])
    return 1

def create_feat_db(sequences):
    for seq in sequences:
        intr = read_intrinsics(seq)
        feat = []
        N = seq_len(seq)
        pairs = [(i, i+1) for i in range(N)] + [(i, i+2) for i in range(N-1)] + [(i, i+3) for i in range(N-2)]
        NN = len(pairs)
        print('sequence:', seq, 'tasks:', NN)
        pbar = ProgressBar(widgets=[Percentage(), Bar()], max_value=NN).start()
        tasks = [(seq, pair, intr) for pair in pairs]
        with mp.Pool(mp.cpu_count()-1) as p:
            p.map_async(extract_and_match_wrapper, tasks, callback=feat.append)
    
        while len(feat) < NN:
            print(len(feat))
            pbar.update(len(feat))
            sleep(0.5)
        pbar.finish()
        
        with open(os.path.join('data', 'features', '%s.pkl' % seq), 'wb+') as fd:
                pickle.dump([pairs, feat], fd)


# In[34]:

if __name__ == '__main__':
    print('process id:', os.getpid())
    create_feat_db(['04'])


# In[ ]:

# grids = list(zip([6,5,4], [4,3,2]))
nbins = [100, 150, 200]
#p = mp.Pool(mp.cpu_count()-1)

for seq in ['%02d' % v for v in range(12)]:
    print('processing sequence %s' % seq)
    seq_dir = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq)
    calib_file = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'calib.txt')

    with open(calib_file, 'r') as f:
        for line in f:
            line = [float(v) for v in line.split()[1:] ]
            K = np.array(line).reshape(3,4)
            break

    Intrinsic = namedtuple('intrinsic', ['f', 'pp'])
    intr = Intrinsic(f = K[0,0], pp=K[range(2),2])
    image_0 = os.path.join(seq_dir, 'image_0')
    
    poses = read_poses(os.path.join('data', 'paths', '%s.txt' % seq))
    nposes = len(poses)
    
    pairs1 = [(i,i+1) for i in range(nposes-1)]
    pairs2 = [(i,i+2) for i in range(nposes-2)]
    
    dts1 = compute_dt(pairs1, poses)
    dts2 = compute_dt(pairs2, poses)
    
    scales1 = [LA.norm(val) for val in dts1]
    scales2 = [LA.norm(val) for val in dts2]
    with np.errstate(divide='ignore'):
        h1, edges1 = np.histogram(scales1, bins=10)
        h2, edges2 = np.histogram(scales2, bins=10)
    fig, ax = plt.subplots()
    bar1 = ax.bar(edges1[:-1], h1, width=edges1[1]-edges1[0])
    bar2 = ax.bar(edges2[:-1], h2, width=edges2[1]-edges2[0], color='g')
    ax.legend((bar1[0], bar2[0]), ('scale1', 'scale2'))
    plt.title('sequence %s' % seq)
    pairs = pairs1 + pairs2
    scales = scales1 + scales2
    dts = dts1 + dts2
    N = len(pairs)
    
    for grid in grids:
        for nb in nbins:
            print('sequence: %s, grid: (%d, %d), nbins: %d' % (seq, grid[0], grid[1], nb))
            dirname = os.path.join('data','%d_%dx%d' % (nb, grid[0], grid[1]))
            mkdir_p(dirname)
            feat = []
            pbar = ProgressBar(widgets=[Percentage(), Bar()], max_value=N).start()
            tasks = [(image_0, pair, grid, nb) for pair in pairs]
            r = [p.map_async(do_work, (x,), callback=feat.append) for x in tasks]
            while len(feat) < N:
                pbar.update(len(feat))
                sleep(0.5)
            pbar.finish()

            #feat = p.map(do_work, [(image_0, pair, grid, nb) for pair in pairs])
            with open(os.path.join(dirname, '%s.pkl' % seq), 'wb+') as fd:
                pickle.dump([pairs, feat, dts, scales], fd)


# In[ ]:



