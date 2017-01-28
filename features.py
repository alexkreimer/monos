import os, sys, math, errno, glob, warnings, traceback, pickle, gzip, cv2, multiprocessing as mp
import matplotlib, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as plticker
from collections import namedtuple
from matplotlib import cm
from PIL import Image
from progressbar import Percentage, Bar, ProgressBar
from time import sleep
import argparse

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

KITTI_HOME = '/home/kreimer/KITTI'

Intrinsic = namedtuple('Intrinsic', ['f', 'pp'])

def read_scales(file):
    scales = []
    with open(file, 'r') as f:
        for line in f:
            pose = np.array([float(val) for val in line.split()]).reshape(3,4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            try:
                delta = np.dot(np.linalg.inv(ppose), pose)
            except NameError:
                ppose = pose
                continue

            scale = np.linalg.norm(delta[:3,3])
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
    deltas = np.linalg.norm(deltas, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h, edges = np.histogram(deltas, bins=range(nbins+1), density=True)
    h = np.array(np.nan_to_num(h))
    return (h, edges)

def compute_feature_vector(kp1, kp2, grid, nbins, imsize=(1226, 370)):
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
    return np.array(desc).ravel()


# In[10]:

def this_needs_a_new_life(seq, nbins, grid, imsize=(1226,370)):
    ''' compute optical flow features for a pair of images '''
    dpi = 20.
    stride = (float(imsize[0])/grid[0], float(imsize[1])/grid[1])

    locx = plticker.MultipleLocator(base=stride[0])
    locy = plticker.MultipleLocator(base=stride[1])
    cmap = plt.get_cmap('jet')
    
    image1 = read_image(seq, pair[0])
    image2 = read_image(seq, pair[1])
    
    pairs, (kp1, kp2) = load_tracks(seq)
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
        #plt.figure()
        #f, axarr = plt.subplots(grid[1], grid[0], sharex='col', sharey='row')
        #for i in range(grid[0]):
        #    for j in range(grid[1]):
        #        axarr[j, i].bar(range(nbins), h[i*grid[1]+j])


# In[11]:

def compute_dt(pairs, poses):
    dts = []
    for pair in pairs:
        pose1 = poses[pair[1]]
        pose0 = poses[pair[0]]
        delta = np.dot(np.linalg.inv(pose0), pose1)
        dt = delta[0:3,3]
        dts.append(dt)
    return dts


# In[12]:

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


# In[14]:

def extract_and_match(seq, pair, intr):
    ''' compute optical flow features for a pair of images '''
    
    image1 = read_image(seq, pair[0])
    image2 = read_image(seq, pair[1])
    
    (kp1, des1) = harris(image1)
    (kp2, des2) = harris(image2)
    
    matches = match(des1, des2)
    kp1, des1, kp2, des2 = prune_matches(matches, kp1, kp2, des1, des2, intr)
    return (kp1, kp2)


# In[15]:

def seq_len(seq):
    ''' computes KITTI sequence length (by counting the number of .png files)'''
    global KITTI_HOME
    
    return(len(glob.glob(os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'image_0', '*.png'))))


# In[16]:

def read_intrinsics(seq):
    ''' read KITTI camera intrinsics for a sequence '''
    global KITTI_HOME

    calib_file = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq, 'calib.txt')
    with open(calib_file, 'r') as f:
        for line in f:
            line = [float(v) for v in line.split()[1:] ]
            K = np.array(line).reshape(3,4)
            break
    intr = Intrinsic(f = K[0,0], pp=K[range(2),2])
    return intr


# In[17]:

def extract_and_match_wrapper(task):
    seq, pair, intr = task
    val = extract_and_match(seq, pair, intr)
    return val

def save_tracks(seq, pairs, feat):
    with gzip.open(os.path.join('data', 'tracks', '%s.pklz' % seq), 'wb+') as fd:
        pickle.dump([pairs, feat], fd)

def load_tracks(seq):
    with gzip.open(os.path.join('data', 'tracks', '%s.pklz' % seq), 'rb') as fd:
        pairs, feat = pickle.load(fd)
    return pairs, feat

def compute_save_tracks(sequences, strides):
    ''' extract harris corners and match them between image pairs
        save intermediate results to speed up things
    '''
    for seq in sequences:
        intr = read_intrinsics(seq)
        feat = []
        N = seq_len(seq)
        
        pairs = []
        for stride in strides:
            pairs = pairs + [(i, i+stride) for i in range(N-stride)]

        NN = len(pairs)
        
        pbar = ProgressBar(widgets=[Percentage(), Bar()], max_value=NN).start()
        tasks = [(seq, pair, intr) for pair in pairs]
        
        p = mp.Pool(mp.cpu_count()-1)
        r = [p.map_async(extract_and_match_wrapper, (t,), callback=feat.extend) for t in tasks]
        p.close()
        while len(feat) < NN:
            pbar.update(len(feat))
            sleep(0.5)
        pbar.update(len(feat))
        pbar.finish()
        p.join()
        save_tracks(seq, pairs, feat)


# In[18]:

def plot_seq_scale(seq, strides):
    poses = read_poses(os.path.join('data', 'paths', '%s.txt' % seq))
    nposes = len(poses)
    
    for stride in strides:
        pairs = [(i, i+stride) for i in range(nposes-stride)]
    
        dts = compute_dt(pairs, poses)
        scales = [np.linalg.norm(val) for val in dts]

        with np.errstate(divide='ignore'):
            h, edges = np.histogram(scales, bins=10)

        fig, ax = plt.subplots()
        bar = ax.bar(edges[:-1], h, width=edges[1]-edges[0])
        ax.legend((bar[0],), ('stride: %d' % stride,))
        plt.title('sequence %s' % seq)


# In[73]:

def compute_feature_vector_wrapper(args):
    ''' this is here because multiprocessing uses pickle which does not pickle lambdas '''
    kp1, kp2, grid, nbins = args
    return compute_feature_vector(kp1, kp2, grid, nbins)

def save_features(seq, stride, grid, nb, pairs, X, dts):
    ds_name = '_'.join([seq, str(stride), str(grid[0]), str(grid[1]), str(nb)])
    with gzip.open(os.path.join('data', 'features', '%s.pklz' % ds_name), 'wb+') as fd:
        pickle.dump([pairs, X, dts], fd)

def load_features(seq, stride, grid, nb):
    ds_name = '_'.join([seq, str(stride), str(grid[0]), str(grid[1]), str(nb)])
    with gzip.open(os.path.join('data', 'features', '%s.pklz' % ds_name), 'rb') as fd:
        return pickle.load(fd)

def compute_save_features(sequences, strides):
    for seq in sequences:
        print('processing sequence %s' % seq)
        intr = read_intrinsics(seq)
        
        poses = read_poses(os.path.join('data', 'paths', '%s.txt' % seq))
        nposes = len(poses)
        pairs, matches = load_tracks(seq)
        tracks = dict(zip(pairs, matches))
        
        for stride in strides:
            pairs = [(i, i+stride) for i in range(nposes-stride)]
            print(pairs[-1])
            
            dts = compute_dt(pairs, poses)
            scales = [np.linalg.norm(val) for val in dts]
 
            N = len(pairs)
            for grid in grids:
                for nb in nbins:
                    print('stride:', stride, 'grid: (%d, %d), nbins: %d' % (grid[0], grid[1], nb))
                    X = []
                    
                    pbar = ProgressBar(widgets=[Percentage(), Bar()], max_value=N).start()
                    
                    tasks = [(tracks[pair][0], tracks[pair][1], grid, nb) for pair in pairs]
                    p = mp.Pool(mp.cpu_count()-1)
                    
                    r = [p.map_async(compute_feature_vector_wrapper, (x,), callback=X.extend) for x in tasks]
                    p.close()
                    
                    while len(X) < N:
                        pbar.update(len(X))
                        sleep(0.1)

                    pbar.update(len(X))
                    pbar.finish()
                    p.join()
                    save_features(seq, stride, grid, nb, pairs, X, dts)


# In[20]:

def load_images(seq, pair):
    im1 = read_image(seq, pair[0])
    im2 = read_image(seq, pair[1])
    return (im1, im2)

def plot_feature_vector(seq, pair, grid, nb, stride):
    data = load_features(seq, stride, grid, nb)
    im1, im2 = load_images(seq, pair)
    feature_vector = data[1][pair[0]]
    print(len(feature_vector))

    plt.imshow(im2, cmap="gray")
    
    ax = plt.subplot2grid((grid[0], 1), (0, 0), rowspan=grid[0])
    ax.imshow(im1, cmap="gray")    


    bin_len = (int)(len(feature_vector)/grid[1])
    print(bin_len)    

    parts = [feature_vector[i*bin_len:(i+1)*bin_len] for i in range(grid[1])]
    
    #f, ax1 = plt.subplots(grid[1], 1, sharex=True)
    for row in range(grid[1]):
        ax = plt.subplot2grid((grid[0], 1), (row, 0), rowspan=1)
        ax.bar(range(bin_len), parts[row])


# In[21]:

#print('process id:', os.getpid())
#sequences = ['%02d' % seq_num for seq_num in range(11)]
#compute_save_tracks(sequences, strides)
#compute_save_features(sequences, strides)


# In[74]:

def plt_features(seq, stride, grid, nb, i1, i2):
    pairs, X, dts = load_features(seq, stride, grid, nb)
    show_feature_vector(seq, (0, 1), grid, nb, stride)
def make_arrays(num_samples, num_features):
  if num_samples:
    X = np.ndarray((num_samples, num_features), dtype=np.float32)
    y = np.ndarray(num_samples, dtype=np.float32)
  else:
    X, y = None, None
  return X, y

def merge_datasets(pickle_files):
    num_parts = len(pickle_files)
    X, y, Pairs = None, None, None
    for label, pickle_file in enumerate(pickle_files):
        seq = int(os.path.split(pickle_file)[1].split('_')[0])
        try:
            with gzip.open(pickle_file, 'rb') as f:
                pairs, feat, dts = pickle.load(f)
                pairs = np.array([(seq, pair[0], pair[1]) for pair in pairs])
                feat = np.array(feat)
                scales = np.array([np.linalg.norm(val) for val in dts])
        except Exception as e:
          print('Unable to process data from', pickle_file, ':', e)
          raise

        if X is not None:
            X = np.vstack((X, feat))
            y = np.hstack((y, scales))
            Pairs = np.vstack((Pairs, pairs))
        else:
            X = feat
            y = scales
            Pairs = pairs
    return X, y, Pairs

def shuffle_in_unison(X, y, pairs):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p,:], y[p], pairs[p,:]

def split_data(X, y, pairs, train_size, valid_size, test_size):
    train_X, train_y, train_pairs = X[:train_size,:], y[:train_size], pairs[:train_size,:]
    
    k = train_size + valid_size
    valid_X, valid_y, valid_pairs = X[train_size:k,:], y[train_size:k], pairs[train_size:k,:]
    
    test_X, test_y, test_pairs = X[k:(k+test_size),:], y[k:(k+test_size)], pairs[k:(k+test_size),:]
    
    return train_X, train_y, train_pairs, valid_X, valid_y, valid_pairs, test_X, test_y, test_pairs

def create_dataset(train_sequences, grids, nbins, strides):
    labels = []
    for grid in grids:
        for nb in nbins:
            if len(strides) > 1:
                labels.append(['%d_%d_%d_%d' % (stride, grid[0], grid[1], nb) for stride in strides])
            labels.extend(['%d_%d_%d_%d' % (stride, grid[0], grid[1], nb) for stride in strides])
    for label in labels:
        print('create dataset: ', label)
        if isinstance(label, str):
            tag = label
            files = [os.path.join('data', 'features', '%s_%s.pklz' % (seq, label)) for seq in train_sequences]
        else:
            tag = '_'.join(label)
            files = []
            for sub_label in label:
                files.extend([os.path.join('data', 'features', '%s_%s.pklz' % (seq, sub_label)) for seq in train_sequences])

        X, y, pairs = merge_datasets(files)
        X, y, pairs = shuffle_in_unison(X,y,pairs)

        pickle.dump({'X': X, 'y': y, 'pairs': pairs}, gzip.open(os.path.join('data', 'datasets', 'data_%s.pklz' % tag), 'wb'))

        size = len(X)
        train_size = int(.7*size)
        valid_size = int(.15*size)
        test_size = size-train_size-valid_size

        train_X, train_y, train_pairs, valid_X, valid_y, valid_pairs, test_X, test_y, test_pairs = split_data(X, y, pairs, train_size, valid_size, test_size)
        print('#train: ', len(train_y), '#val: ', len(valid_y), '#test: ', len(test_y))
        pickle.dump({'X': train_X, 'y': train_y, 'pairs': train_pairs}, gzip.open('data/datasets/train_%s.pklz' % tag, 'wb'))
        pickle.dump({'X': valid_X, 'y': valid_y, 'pairs': valid_pairs}, gzip.open('data/datasets/valid_%s.pklz' % tag, 'wb'))
        pickle.dump({'X': test_X, 'y': test_y, 'pairs': test_pairs}, gzip.open('data/datasets/test_%s.pklz' % tag, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_corners', action='store_true')
    parser.add_argument('--compute_features', action='store_true')
    parser.add_argument('--create_dataset', action='store_true')
    
    args = parser.parse_args()

    # sequences = ['%02d' % seq for seq in range(22)]
    sequences = ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']
    
    grids = [(6,3)]
    nbins = [300]
    strides = [1]
    prefixes = ['']
    
    if args.extract_corners:
        compute_save_tracks(sequences, strides)
    if args.compute_features:
        compute_save_features(sequences, strides)
    if args.create_dataset:
        create_dataset(sequences, grids, nbins, strides)
