import os
import torch
import argparse
import numpy as np
import deepdish as dd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


from utils.change_ckpt_dict_name import rename_var

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    #版本原因，下面一句替换
    #reader = tf.train.NewCheckpointReader(ckpt)
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("--infile", type=str,
                        default='weight_tf/wgts_epochs_10000.ckpt',
                        help="Path to the ckpt.")
    parser.add_argument("--outfile", type=str, nargs='?',default='weight_out_keras/wgts_epochs_10000.hdf5',
                        help="Output file (inferred if missing).")
    args = parser.parse_args()
    
    if args.outfile == '':
        args.outfile = os.path.splitext(args.infile)[0] + '.hdf5'
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    new_infile = os.path.splitext(args.infile)[0] + '_fix.ckpt'
    print(new_infile)
    rename_var(args.infile, new_infile)
    weights = read_ckpt(new_infile)
    dd.io.save(args.outfile, weights)
    weights2 = dd.io.load(args.outfile)