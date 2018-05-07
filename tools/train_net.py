#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--rpn_anchor_scale', dest='rpn_anchor_scale',
                        help='scale of anchors to creat',
                        default='8,16,32', type=str)
    parser.add_argument('--rpn_anchor_ratio', dest='rpn_anchor_ratio',
                        help='ratio of anchor to creat',
                        default='0.5,1,2', type=str)
    parser.add_argument('--rpn_min_size', dest='rpn_min_size',
                        help='minimun size of an anchor choosed by rpn',
                        default=16, type=int)
    parser.add_argument('--rpn_positive_overlap', dest='rpn_positive_overlap',
                        help='rpn positive overlap',
                        default=0.7)
    parser.add_argument('--rpn_negative_overlap', dest='rpn_nagetive_overlap',
                        help='rpn negative overlap',
                        default=0.3)
    parser.add_argument('--bg_thresh_hi', dest='bg_thresh_hi',
                        help='Overlap threshold for a ROI to be considered background',
                        default=0.5)
    parser.add_argument('--fg_thresh', dest='fg_thresh',
                        help='Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)',
                        default=0.5)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    #il faut ajouter et changer les portfolio_id
    if args.rpn_anchor_scale != '8,16,32' :
        list_scale = [int(x) for x in args.rpn_anchor_scale.split(",")]
        cfg.RPN_ANCHOR_SCALES = np.array(list_scale)
    if args.rpn_anchor_ratio != '0.5,1,2' :
        list_ratio = [float(x) for x in args.rpn_anchor_ratio.split(",")]
        cfg.RPN_ANCHOR_RATIOS = list_ratio
    if args.rpn_min_size != 16 :
        cfg.TRAIN.RPN_MIN_SIZE = args.rpn_min_size
        cfg.TEST.RPN_MIN_SIZE = args.rpn_min_size
    if float(args.rpn_positive_overlap) != 0.7 :
        cfg.TRAIN.POSITIVE_OVERLAP = float(args.rpn_positive_overlap)
    if float(args.rpn_negative_overlap) != 0.3 :
        cfg.TRAIN.NEGATIVE_OVERLAP = float(args.rpn_negative_overlap)
    if float(args.bg_thresh_hi) != 0.5 :
        cfg.TRAIN.BG_THRESH_HI = float(args.bg_thresh_hi)
    if float(args.fg_thresh) != 0.5 :
        cfg.TRAIN.FG_THRESH = float(args.fg_thresh)




    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
