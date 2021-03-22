import time
import copy

import torch

from models.mesh_classifier import SRModel
from options.train_options import TrainOptions

from models import create_model
from util.writer import Writer
from create_dataset import get_dataloaders, MeshSuperResolutionDataset
from models.networks import UNETMeshSuperResolutionNetwork

import shutil
import os

results_path = '../results'
if os.path.exists(results_path):
    shutil.rmtree(results_path)
os.mkdir(results_path)


def run_test(epoch, model, test_dl):
    epoch_results_path = os.path.join(results_path, f"{epoch}")
    os.mkdir(epoch_results_path)

    total_steps = 0
    _loss = torch.tensor(0.)
    curr = 0
    for i, data in enumerate(test_dl):
        data = {
            "lr_mesh": [copy.deepcopy(dataset.lr_meshes[x]) for x in data],
            "lr_features": [copy.deepcopy(dataset.lr_features[x]) for x in data],
            "hr_mesh": [copy.deepcopy(dataset.hr_meshes[x]) for x in data],
            "hr_features": [copy.deepcopy(dataset.hr_features[x]) for x in data]
        }
        total_steps += opt.batch_size
        model.set_input(data)
        l, _, curr = model.test(results_path, curr, epoch_results_path)
        _loss += l.cpu()

    print(f"Test: {_loss / total_steps}")


def choose_model(opt):
    net = UNETMeshSuperResolutionNetwork(opt).cuda()  # TODO...
    return SRModel(opt, net)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = MeshSuperResolutionDataset(opt)
    train_dl, test_dl = get_dataloaders(dataset)
    dataset_size = len(dataset)
    print('#meshes in dataset = %d' % dataset_size)

    model = choose_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if epoch % opt.save_epoch_freq == 0:
            for i, data in enumerate(train_dl):
                data = {
                    "lr_mesh": [copy.deepcopy(dataset.lr_meshes[x]) for x in data],
                    "lr_features": [copy.deepcopy(dataset.lr_features[x]) for x in data],
                    "hr_mesh": [copy.deepcopy(dataset.hr_meshes[x]) for x in data],
                    "hr_features": [copy.deepcopy(dataset.hr_features[x]) for x in data]
                }
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.print_freq == 0:
                    loss = model.loss
                    t = (time.time() - iter_start_time) / opt.batch_size
                    writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                    writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

                if i % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.save_network('latest')

                iter_data_time = time.time()
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            run_test(epoch, model, test_dl)

    writer.close()
