"""
Mix dataset
support reading images from lmdb
"""
import logging
import random

import data.util as util
import lmdb
import numpy as np
import torch
import torch.utils.data as data


logger = logging.getLogger("base")


class MixDataset(data.Dataset):
    """
    Reading the training REDS dataset
    key example: 000_00000000
    HQ: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    """

    def __init__(self, opt):
        super(MixDataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        self.HQ_roots = opt["dataroots_HQ"]
        self.LQ_roots = opt["dataroots_LQ"]
        self.use_identical = opt["identical_loss"]
        dataset_weights = opt["dataset_weights"]
        self.data_type = "lmdb"
        # directly load image keys
        self.HQ_envs, self.LQ_envs = None, None
        self.paths_HQ = []
        for idx, (HQ_root, LQ_root) in enumerate(zip(self.HQ_roots, self.LQ_roots)):
            paths_HQ, _ = util.get_image_paths(self.data_type, HQ_root)
            self.paths_HQ += list(zip([idx] * len(paths_HQ), paths_HQ)) * dataset_weights[idx]
        random.shuffle(self.paths_HQ)
        logger.info("Using lmdb meta info for cache keys.")

    def _init_lmdb(self):
        self.HQ_envs, self.LQ_envs = [], []
        for HQ_root, LQ_root in zip(self.HQ_roots, self.LQ_roots):
            self.HQ_envs.append(lmdb.open(HQ_root, readonly=True, lock=False, readahead=False, meminit=False))
            self.LQ_envs.append(lmdb.open(LQ_root, readonly=True, lock=False, readahead=False, meminit=False))

    def __getitem__(self, index):
        if self.HQ_envs is None:
            self._init_lmdb()

        HQ_size = self.opt["HQ_size"]
        env_idx, key = self.paths_HQ[index]
        name_a, name_b = key.split("_")
        target_frame_idx = int(name_b)

        # determine the neighbor frames
        # ensure not exceeding the borders
        neighbor_list = [target_frame_idx]
        name_b = "{:08d}".format(neighbor_list[0])

        # get the HQ image (as the center frame)
        img_HQ_l = []
        for v in neighbor_list:
            img_HQ = util.read_img(self.HQ_envs[env_idx], "{}_{:08d}".format(name_a, v), (3, 720, 1280))
            img_HQ_l.append(img_HQ)

        # get LQ images
        img_LQ = util.read_img(self.LQ_envs[env_idx], "{}_{:08d}".format(name_a, neighbor_list[-1]), (3, 720, 1280))
        if self.opt["phase"] == "train":
            _, H, W = 3, 720, 1280  # LQ size
            # randomly crop
            rnd_h = random.randint(0, max(0, H - HQ_size))
            rnd_w = random.randint(0, max(0, W - HQ_size))
            img_LQ = img_LQ[rnd_h : rnd_h + HQ_size, rnd_w : rnd_w + HQ_size, :]
            img_HQ_l = [v[rnd_h : rnd_h + HQ_size, rnd_w : rnd_w + HQ_size, :] for v in img_HQ_l]

            # augmentation - flip, rotate
            img_HQ_l.append(img_LQ)
            rlt = util.augment(img_HQ_l, self.opt["use_flip"], self.opt["use_rot"])
            img_HQ_l = rlt[0:-1]
            img_LQ = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_HQs = np.stack(img_HQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_HQs = img_HQs[:, :, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_HQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQs, (0, 3, 1, 2)))).float()
        # print(img_LQ.shape, img_HQs.shape)

        if self.use_identical and np.random.randint(0, 10) == 0:
            img_LQ = img_HQs[-1, :, :, :]
            return {"LQ": img_LQ, "HQs": img_HQs, "identical_w": 10}

        return {"LQ": img_LQ, "HQs": img_HQs, "identical_w": 0}

    def __len__(self):
        return len(self.paths_HQ)
