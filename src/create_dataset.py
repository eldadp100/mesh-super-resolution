"""
    Here we take a dataset of meshes.
    For each mesh we downscale and save the low resolution and high resolution tuple.
    Then we make a pytorch Dataset and make it available to use in pytorch.
"""

from models.layers.mesh import Mesh
from models.layers.mesh_pool import MeshPool
import shutil
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob

from options.train_options import TrainOptions
from subdivide_mesh import do_subdivide

opt22 = TrainOptions().parse()


#
# def create_dataset(init_dataset='shrec_16', opts=None):
#     downscale_edges = 500
#     mesh_pool = MeshPool(downscale_edges)
#
#     output_name = f'sr_{init_dataset}'
#     output_folder = f'../datasets/{output_name}'
#     if os.path.exists(output_folder):
#         x = input("Are you sure you want to delete previous folder? [Y/N]  ")
#         if not x.lower() == 'y':
#             return
#
#         shutil.rmtree(output_folder)
#     os.mkdir(output_folder)
#     mesh_paths = glob.glob(f'../datasets/{init_dataset}' + '/**/*.obj', recursive=True)
#     for i, path in enumerate(mesh_paths):
#         ith_dir = f'{output_folder}/{i}'
#         os.mkdir(ith_dir)
#         shutil.copy2(path, f'{ith_dir}/HR.obj')
#         with open(f'{ith_dir}/original_path.txt', 'w') as f:
#             f.write(path)
#
#     min_nedges = 10000
#     for i, path in enumerate(mesh_paths):
#         mesh = Mesh(file=path, opt=opts, hold_history=True, export_folder=None)
#
#         from models.networks import UNETMeshSuperResolutionNetwork
#         import torch
#         net = UNETMeshSuperResolutionNetwork(opt22).cuda()
#         net(torch.tensor([mesh.extract_features()]).cuda().float(), [mesh])
#
#         mesh_pool.forward(None, [mesh], random_pool=True)
#         mesh.export(f'{output_folder}/{i}/LR_before_subdivide.obj')
#         subdivided_mesh = Mesh.subdivide(f'{output_folder}/{i}/LR_before_subdivide.obj', f'{output_folder}/{i}/LR.obj',
#                                          opts)
#         min_nedges = min(min_nedges, len(subdivided_mesh.edges))
#
#     """
#         For Genus 0: V - E + F = 2 (EULER)
#                      faces are triangles -> 2E=3F
#                      E = 3V - 6
#         For Genus 1:
#                     E = 3V - 12
#         ...
#         so we need to do pooling again
#     """
#     meshes = [Mesh(file=f'{output_folder}/{i}/LR.obj', opt=opts, hold_history=True, export_folder=None) for i in
#               range(len(mesh_paths))]
#     A = sorted(set([len(x.edges) for x in meshes]))
#     max_A = max(A) if A[-1] - A[-2] > 1 else max(A) + 2
#     B = [max_A - x for x in A]
#     B = [(b // 3, 0) if b % 3 == 0 else (b // 3, 1) if b % 3 == 2 else (b // 3 - 1, 2) for b in B]
#     C = {A[i]: B[i] for i in range(len(B))}
#
#
#
#     meshes_2 = []
#     for i, mesh in enumerate(meshes):
#         edges_count = len(mesh.edges)
#         mesh.subidivide_3(C[edges_count][0])
#         mesh.subdivide_2(C[edges_count][1])
#
#         os.remove(f'{output_folder}/{i}/LR.obj')
#         mesh.export_2(f'{output_folder}/{i}/LR.obj')
#         meshes_2.append(mesh)
#     a = 1


def create_dataset(init_dataset='shrec_16_full', opts=None):
    downscale_edges = 500
    mesh_pool = MeshPool(downscale_edges)

    output_name = f'sr_{init_dataset}'
    output_folder = f'../datasets/{output_name}'
    if os.path.exists(output_folder):
        x = input("Are you sure you want to delete previous folder? [Y/N]  ")
        if not x.lower() == 'y':
            return

        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    mesh_paths = glob.glob(f'../datasets/{init_dataset}' + '/**/*.obj', recursive=True)
    for i, path in enumerate(mesh_paths):
        ith_dir = f'{output_folder}/{i}'
        os.mkdir(ith_dir)
        shutil.copy2(path, f'{ith_dir}/HR.obj')
        with open(f'{ith_dir}/original_path.txt', 'w') as f:
            f.write(path)

    nedges = []
    for i, path in enumerate(mesh_paths):
        mesh = Mesh(file=path, opt=opts, hold_history=True, export_folder=None)

        mesh_pool.forward(None, [mesh], random_pool=True)
        mesh.export(f'{output_folder}/{i}/LR_before_subdivide.obj')
        mesh.calculate_faces()
        nedges.append(len(mesh.edges) + len(mesh.faces) * 3)

    max_nedges = max(nedges)
    for i, path in enumerate(mesh_paths):
        path = f'{output_folder}/{i}/LR_before_subdivide.obj'
        do_subdivide(path, f'{output_folder}/{i}/LR.obj', max_nedges)


class MeshSuperResolutionDataset(Dataset):

    def __init__(self, opts=None):
        super(MeshSuperResolutionDataset, self).__init__()
        self.dataset_root = opts.dataroot

        self.paths = os.listdir(self.dataset_root)
        self.opts = opts

        self.lr_meshes = [
            Mesh(file=f'{self.dataset_root}/{path}/LR.obj', opt=self.opts, hold_history=True, export_folder=None) for
            path in self.paths]

        self.hr_meshes = [
            Mesh(file=f'{self.dataset_root}/{path}/HR.obj', opt=self.opts, hold_history=True, export_folder=None) for
            path in self.paths]

        self.lr_features = [x.extract_features() for x in self.lr_meshes]
        self.hr_features = [x.extract_features() for x in self.hr_meshes]

    def __getitem__(self, item):
        return item

    def __len__(self):
        return len(self.paths)


def get_dataloaders(dataset, train_ratio=0.7, batch_size=32, shuffle=True):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    from collections import namedtuple

    NT = namedtuple('A', 'num_aug')
    opts = NT(num_aug=1)
    print((opts.num_aug))

    create_dataset(opts=opts)

# import os
# import torch
# from data.base_dataset import BaseDataset
# from util.util import is_mesh_file, pad
# import numpy as np
# from models.layers.mesh import Mesh

# class SegmentationData(BaseDataset):

#     def __init__(self, opt):
#         BaseDataset.__init__(self, opt)
#         self.opt = opt
#         self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
#         self.root = opt.dataroot
#         self.dir = os.path.join(opt.dataroot, opt.phase)
#         self.paths = self.make_dataset(self.dir)
#         self.seg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg')
#         self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
#         self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
#         self.nclasses = len(self.classes)
#         self.size = len(self.paths)
#         self.get_mean_std()
#         # # modify for network later.
#         opt.nclasses = self.nclasses
#         opt.input_nc = self.ninput_channels

#     def __getitem__(self, index):
#         path = self.paths[index]
#         mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
#         meta = {}
#         meta['mesh'] = mesh
#         label = read_seg(self.seg_paths[index]) - self.offset
#         label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
#         meta['label'] = label
#         soft_label = read_sseg(self.sseg_paths[index])
#         meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
#         # get edge features
#         edge_features = mesh.extract_features()
#         edge_features = pad(edge_features, self.opt.ninput_edges)
#         meta['edge_features'] = (edge_features - self.mean) / self.std
#         return meta

#     def __len__(self):
#         return self.size

#     @staticmethod
#     def get_seg_files(paths, seg_dir, seg_ext='.seg'):
#         segs = []
#         for path in paths:
#             segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
#             assert(os.path.isfile(segfile))
#             segs.append(segfile)
#         return segs

#     @staticmethod
#     def get_n_segs(classes_file, seg_files):
#         if not os.path.isfile(classes_file):
#             all_segs = np.array([], dtype='float64')
#             for seg in seg_files:
#                 all_segs = np.concatenate((all_segs, read_seg(seg)))
#             segnames = np.unique(all_segs)
#             np.savetxt(classes_file, segnames, fmt='%d')
#         classes = np.loadtxt(classes_file)
#         offset = classes[0]
#         classes = classes - offset
#         return classes, offset

#     @staticmethod
#     def make_dataset(path):
#         meshes = []
#         assert os.path.isdir(path), '%s is not a valid directory' % path

#         for root, _, fnames in sorted(os.walk(path)):
#             for fname in fnames:
#                 if is_mesh_file(fname):
#                     path = os.path.join(root, fname)
#                     meshes.append(path)

#         return meshes


# def read_seg(seg):
#     seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
#     return seg_labels


# def read_sseg(sseg_file):
#     sseg_labels = read_seg(sseg_file)
#     sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
#     return sseg_labels
