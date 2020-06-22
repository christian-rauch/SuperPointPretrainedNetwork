from torch.utils import data
import os
import glob
import parse
from PIL import Image
import numpy as np
from skimage.transform import AffineTransform, warp
from torchvision import transforms


class ImagePairLoader(data.Dataset):
    img_file_fmt = "s{S}-cam{C}-t{T}.{I}.png"
    map_file_fmt = "s{S}-cam{C}.map.npz"

    def __init__(self, data_path, config, seed=None):
        self.local_seed(seed)

        self.data_path = data_path
        self.map_path_list = glob.glob(os.path.join(data_path, self.map_file_fmt.format(S='*', C='*')))
        self.img_path_fmt = os.path.join(self.data_path, self.img_file_fmt.format(S="{S}", C="{C}", T="{T}", I="colour"))
        self.config = config

        self.bkgr_file_paths = list()
        if 'background' in config and config['background']['mode'] is not None and os.path.isdir(config['background']['mode']):
            types = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'gif', 'bmp', 'pbm', 'pgm', 'ppm']
            for t in types:
                self.bkgr_file_paths.extend(glob.glob(os.path.join(config['background']['mode'], "**", "*."+t), recursive=True))
            # image colour transforms
            self.augmentation_colour = list()
            self.augmentation_colour.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))

    def local_seed(self, seed):
        '''
        seed random number generators with local scope to this class instance
        '''
        from random import Random
        from numpy.random import Generator, MT19937
        self.random_py = Random(seed)
        self.random_np = Generator(MT19937(seed))

    def __len__(self):
        return len(self.map_path_list)

    def get_pair(self, index):
        map_path = self.map_path_list[index]

        # read sample and camera ID from filename
        parsed_result = parse.parse(self.map_file_fmt, os.path.basename(map_path))
        i_sample = parsed_result['S']
        i_cam = parsed_result['C']

        # load mapping between coordinates and mask
        npz_file = np.load(map_path)
        map_xy = npz_file['map_xy']
        mask = npz_file['mask']
        segments = npz_file['segments'] if 'segments' in npz_file else [None] * 2
        uv = npz_file['uv'] if 'uv' in npz_file else [None] * 2
        npz_file.close()

        I0 = np.array(Image.open(self.img_path_fmt.format(S=i_sample, C=i_cam, T=0)))
        I1 = np.array(Image.open(self.img_path_fmt.format(S=i_sample, C=i_cam, T=1)))

        return [I0, I1], map_xy, mask, segments, uv

    def augmentation_foreground_affine_random(self, shape, conf):
        '''
        2D translation and rotation
        Rotations are only applied to the reference image (0), but not the target image (1)
        since this might change the target coordinate.
        '''
        shape_xy = np.roll(np.array(shape), shift=1)
        t_max = conf.get('translation', 0)
        r_max = conf.get('rotation', 0)

        # 2D transformation on image 0
        tf0 = AffineTransform(
            translation = (self.random_np.uniform() * t_max * shape_xy).astype(np.int),
            rotation = self.random_np.uniform() * r_max)

        # 2D translation on image 1
        t1 = (self.random_np.uniform() * t_max * shape_xy).astype(np.int)
        tf1 = AffineTransform(translation = t1)

        # transform the reference space
        tf0_f = lambda x : warp(x, tf0, mode='wrap', order=0, preserve_range=True)
        tf1_f = lambda x : warp(x, tf1, mode='wrap', order=0, preserve_range=True)
        # transform the mapped coordinates
        tf1c_f = lambda x : np.mod(x+(-1)*t1, shape_xy)

        return tf0_f, tf1_f, tf1c_f

    def augmentation_background_affine_random(self, shape, conf):
        '''
        2D affine transformation (translation, rotation, scale, shear)
        '''
        shape_xy = np.roll(np.array(shape), shift=1)
        t_max = conf.get('translation', 0)
        r_max = conf.get('rotation', 0)
        s_max = conf.get('scale', 0)
        h_max = conf.get('shear', 0)

        # 2D transformation on background 0
        tf0 = AffineTransform(
            translation = (self.random_np.uniform() * t_max * shape_xy).astype(np.int),
            rotation = self.random_np.uniform() * r_max,
            scale = [1 + self.random_np.uniform() * s_max] * 2,
            shear = self.random_np.uniform() * h_max)

        tf1 = AffineTransform(
            translation = (self.random_np.uniform() * t_max * shape_xy).astype(np.int),
            rotation = self.random_np.uniform() * r_max,
            scale = [1 + self.random_np.uniform() * s_max] * 2,
            shear = self.random_np.uniform() * h_max)

        # transform the reference space
        tf0_f = lambda x : warp(x, tf0, mode='wrap', order=0, preserve_range=True)
        tf1_f = lambda x : warp(x, tf1, mode='wrap', order=0, preserve_range=True)

        return tf0_f, tf1_f

    def replace_background_random(self, I0, I1, conf):
        # mask invalid area by alpha channel
        M0 = (I0[..., 3]<0.5)
        M1 = (I1[..., 3]<0.5)

        background_mode = conf.get('mode', None)

        # replace invalid pixels
        if background_mode is None or background_mode == "random":
            # random colour
            I0[M0, :3] = self.random_np.uniform(size=(M0.sum(), 3))
            I1[M1, :3] = self.random_np.uniform(size=(M1.sum(), 3))
        elif background_mode == "mean":
            # mean colour of valid pixels
            I0[M0, :] = np.mean(I0[~M0, :], axis=0)
            I1[M1, :] = np.mean(I1[~M1, :], axis=0)
        elif self.bkgr_file_paths:
            # select a random background image
            background = Image.open(self.random_py.choice(self.bkgr_file_paths)).convert('RGB')
            # apply colour transforms
            if 'augmentation' in conf and conf['augmentation'].get('colour', False):
                background = transforms.Compose(self.augmentation_colour)(background)
            background = transforms.Resize(M0.shape)(background)

            background = np.array(background) / 255

            # apply two different spatial transforms on the same colour image
            tf0, tf1 = self.augmentation_background_affine_random(background.shape[:2], conf['augmentation'])
            b0 = tf0(background)
            b1 = tf1(background)

            I0[M0, :3] = b0[M0, :]
            I1[M1, :3] = b1[M1, :]
        else:
            raise RuntimeError("the background has to be one of the supported modes or a directory with images")

        return I0, I1

    def __getitem__(self, index):
        I, map_xy, mask, segm, uv = self.get_pair(index)

        for i in range(2):
            I[i] = I[i].astype(np.float) / 255

        if 'augmentation' in self.config:
            tf0, tf1, tf1c = self.augmentation_foreground_affine_random(mask.shape[:2], self.config['augmentation'])
            # apply spatial transforms to pairs
            for i, tf in zip(range(2), (tf0,tf1)):
                I[i] = tf(I[i])
                if uv[i] is not None:
                    uv[i] = tf(uv[i])
                if segm[i] is not None:
                    segm[i] = tf(segm[i])
            # apply spatial transforms to mapping
            mask = tf0(mask).astype(np.uint)
            map_xy = tf0(map_xy)
            map_xy = tf1c(map_xy)

            # rotation of the binary mask can lead to a mismatching overlap with the map
            # remove pixels from the mask that point to invalid (nan) correspondences
            mask[~np.isfinite(map_xy).all(axis=-1)] = 0

        if 'background' in self.config:
            I[0], I[1] = self.replace_background_random(I[0], I[1], self.config['background'])

        # remove alpha channel
        for i in range(2):
            I[i] = I[i][..., :3]

        return I[0], I[1], map_xy, mask, segm[0], segm[1], uv[0], uv[1]