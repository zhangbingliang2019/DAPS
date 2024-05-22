"""
REDS dataset
support reading images from lmdb, image folder and memcached
"""
import logging
import os.path as osp
import pickle
import random

import cv2
import data.util as util
import lmdb
import numpy as np
import torch
import torch.utils.data as data


try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger("base")


class REDSDataset(data.Dataset):
    """
    Reading the training REDS dataset
    key example: 000_00000000
    HQ: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        self.HQ_root, self.LQ_root = opt["dataroot_HQ"], opt["dataroot_LQ"]
        self.N_frames = opt["N_frames"]
        self.data_type = self.opt["data_type"]
        # directly load image keys
        if self.data_type == "lmdb":
            self.paths_HQ, _ = util.get_image_paths(self.data_type, opt["dataroot_HQ"])
            logger.info("Using lmdb meta info for cache keys.")
        elif opt["cache_keys"]:
            logger.info("Using cache keys: {}".format(opt["cache_keys"]))
            self.paths_HQ = pickle.load(open(opt["cache_keys"], "rb"))["keys"]
        else:
            raise ValueError(
                "Need to create cache keys (meta_info.pkl) \
                        by running [create_lmdb.py]"
            )

        # remove the REDS4 for testing
        self.paths_HQ = [v for v in self.paths_HQ if v.split("_")[0] not in ["000", "011", "015", "020"]]
        assert self.paths_HQ, "Error: HQ path is empty."

        if self.data_type == "lmdb":
            self.HQ_env, self.LQ_env = None, None
        elif self.data_type == "mc":  # memcached
            self.mclient = None
        elif self.data_type == "img":
            pass
        else:
            raise ValueError("Wrong data type: {}".format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.HQ_env = lmdb.open(self.opt["dataroot_HQ"], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt["dataroot_LQ"], readonly=True, lock=False, readahead=False, meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)

    def _read_img_mc(self, path):
        """ Return BGR, HWC, [0, 255], uint8"""
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        """
        Read BGR channels separately and then combine for 1M limits in cluster
        """
        img_B = self._read_img_mc(osp.join(path + "_B", name_a, name_b + ".png"))
        img_G = self._read_img_mc(osp.join(path + "_G", name_a, name_b + ".png"))
        img_R = self._read_img_mc(osp.join(path + "_R", name_a, name_b + ".png"))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == "mc":
            self._ensure_memcached()
        elif self.data_type == "lmdb" and (self.HQ_env is None or self.LQ_env is None):
            self._init_lmdb()

        HQ_size = self.opt["HQ_size"]
        key = self.paths_HQ[index]
        name_a, name_b = key.split("_")

        # get the HQ image
        img_HQ = util.read_img(self.HQ_env, key, (3, 720, 1280))

        # get the LQ image
        img_LQ = util.read_img(self.LQ_env, key, (3, 720, 1280))

        if self.opt["phase"] == "train":
            _, H, W = 3, 720, 1280  # LQ size
            # randomly crop
            rnd_h = random.randint(0, max(0, H - HQ_size))
            rnd_w = random.randint(0, max(0, W - HQ_size))
            img_LQ = img_LQ[rnd_h : rnd_h + HQ_size, rnd_w : rnd_w + HQ_size, :]
            img_HQ = img_HQ[rnd_h : rnd_h + HQ_size, rnd_w : rnd_w + HQ_size, :]

            # augmentation - flip, rotate
            imgs = [img_HQ, img_LQ]
            rlt = util.augment(imgs, self.opt["use_flip"], self.opt["use_rot"])
            img_HQ = rlt[0]
            img_LQ = rlt[1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_HQ = img_HQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_HQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ, (2, 0, 1)))).float()

        return {"LQ": img_LQ, "HQ": img_HQ}

    def __len__(self):
        return len(self.paths_HQ)
