import os

import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
from .layers.mesh import Mesh


def loss(pred_meshes, real_meshes):
    chamfer_loss = Mesh.chamfer_distance(pred_meshes, real_meshes).sum()
    return chamfer_loss


class SRModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt, net):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.gt = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        # load/define networks
        self.net = net
        self.net.train(self.is_train)
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        A = max([max([len(y) for y in x]) for x in data['lr_features']])
        data['lr_features'] = list(data['lr_features'])
        for x in range(len(data['lr_features'])):
            data['lr_features'][x] = list(data['lr_features'][x])
            for y in range(len(data['lr_features'][x])):
                data['lr_features'][x][y] = list(data['lr_features'][x][y]) + [0.] * (
                            A - len(data['lr_features'][x][y]))

        input_edge_features = torch.tensor(data['lr_features']).float()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.gt = [x.to(self.device) for x in data['hr_mesh']]
        self.mesh = [x.to(self.device) for x in data['lr_mesh']]

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = loss(out, self.gt)
        print(self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, curr, start_path):
        """tests model """
        with torch.no_grad():
            out = self.forward()
            _loss = loss(self.mesh, out)
        for i, m in enumerate(self.mesh):
            m.export(os.path.join(start_path, f"{curr+i}.obj"))
        return _loss, len(self.mesh), curr +len(self.mesh)
    #
    # def get_accuracy(self, pred, labels):
    #     """computes accuracy for classification / segmentation """
    #     if self.opt.dataset_mode == 'classification':
    #         correct = pred.eq(labels).sum()
    #     elif self.opt.dataset_mode == 'segmentation':
    #         correct = seg_accuracy(pred, self.soft_label, self.mesh)
    #     return correct
    #
    # def export_segmentation(self, pred_seg):
    #     if self.opt.dataset_mode == 'segmentation':
    #         for meshi, mesh in enumerate(self.mesh):
    #             mesh.export_segments(pred_seg[meshi, :])
