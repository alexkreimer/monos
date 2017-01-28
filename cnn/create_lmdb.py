from __future__ import print_function
import numpy as np, lmdb, os, sys, math, caffe, argparse, os.path as path, kitti
import glob, progressbar
from caffe.proto import caffe_pb2

def shuffle_in_unison(list1, list2):
    p = np.random.permutation(len(list1))
    list1 = list1[p]
    list2 = list2[p,:]
    return list1, list2

def array_to_datum(arr, label=None):
    """Converts a 3-dimensional array to datum. If the array has dtype uint8,
    the output data will be encoded as a string. Otherwise, the output data
    will be stored in float format.
    """
    #if arr.ndim != 3:
    #    raise ValueError('Incorrect array shape.')
    datum = caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = arr.shape
    if arr.dtype == np.uint8:
        datum.data = arr.tostring()
    else:
        datum.float_data.extend(arr.flat)
    if label is not None:
        # change the units to mm (the label may only hold integer)
        datum.label = int(label*1e3)
    return datum

class CreateLmdb:
    def __init__(self, args):
        scales = []
        image_files = []
        for sequence in range(22):
            images_path = os.path.join(args.kitti_path, 'dataset', 'sequences', '%02d' % sequence, 'image_0', '*.png')
            poses_file_path = '/home/kreimer/prj/monos/data/paths/%02d.txt' % sequence
            image_files_path = sorted(glob.glob(images_path))
            seq_scales = kitti.Util.read_scales(poses_file_path)
            seq_image_files = zip(image_files_path[:-1], image_files_path[1:])
            print('%02d: %g %d' % (sequence, np.sum(seq_scales), len(seq_image_files)))
            scales.extend(seq_scales)
            image_files.extend(seq_image_files)

        scales = np.array(scales)
        image_files = np.array(image_files)
        print('total: %g %d' % (np.sum(scales), image_files.shape[0]))
        train_size = int(image_files.shape[0]*.9)
        scales, image_files = shuffle_in_unison(scales, image_files)
        self.create_lmdb_file(path.join(args.out_dir, 'train'), image_files[:train_size,:], scales[:train_size])
        self.create_lmdb_file(path.join(args.out_dir, 'test'), image_files[train_size:,:], scales[train_size:])
   
    def create_lmdb_file(self, lmdb_path, image_files, scales):
        if not path.isdir(lmdb_path):
            os.makedirs(lmdb_path)

        lmdb_env = lmdb.open(lmdb_path, map_size=int(1e12))
        bar = progressbar.ProgressBar()
        for idx in bar(range(len(scales))):
            with lmdb_env.begin(write=True) as txn:
                image1 = kitti.Util.read_image(image_files[idx,0])
                image2 = kitti.Util.read_image(image_files[idx,1])

                scale = scales[idx]
                image = np.concatenate([image1[:370,:1226,np.newaxis], image2[:370,:1226,np.newaxis]], axis=2)
                datum = array_to_datum(image, scale)
                txn.put('{:0>10d}'.format(idx), datum.SerializeToString())
        lmdb_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create the LMDB for the monocular scale estimation over the KITTI dataset')
    parser.add_argument('--kitti_path', default='/home/kreimer/KITTI')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    
    create_lmdb = CreateLmdb(args)
