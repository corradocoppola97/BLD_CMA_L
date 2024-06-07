import time

import numpy as np
from cmalight_BD import get_w
from network import get_pretrained_net
import torch
import torchvision
from torch.utils.data import Subset
from warnings import filterwarnings
from utils_BD import set_optimizer, closure, accuracy, count_parameters

filterwarnings('ignore')

transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomRotation(10),
                                            torchvision.transforms.RandomAffine(0, shear = 10, scale = (0.8, 1.2)),
                                            torchvision.transforms.ColorJitter(brightness = 0.2, contrast = 0.2,
                                                                               saturation = 0.2),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (
                                                0.5, 0.5, 0.5))])


def train_model(sm_root: str,
                opt: str,
                ep: int,
                ds: str,
                net_name: str,
                history_ID: str,
                dts_train: torch.utils.data.DataLoader,
                dts_test: torch.utils.data.DataLoader,
                verbose_train: bool,
                importance: list,
                beta=1.1,
                patience=3,
                ro=1.005,
                n_ep_cmal=5,
                seed=1,
                *args,
                **kwargs) -> dict:
    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    torch.manual_seed(42)
    if ds == 'CIFAR100':
        model = get_pretrained_net(net_name, num_classes = 100, seed = seed, pretrained = False).to(device)
    else:
        model = get_pretrained_net(net_name, seed = seed, pretrained = False).to(device)
    if verbose_train: print('\n The model has: {} trainable parameters'.format(count_parameters(model)))

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = set_optimizer(opt, model, *args, **kwargs)

    # Initial Setup
    P = (len(dts_train) - 1) * (len(dts_train[0][0])) + len(dts_train[-1][0])
    min_acc = 0
    t1 = time.time()
    fw0 = closure(dts_train, model, criterion, device)
    # print("fw0", fw0) ##
    t2 = time.time()
    time_compute_fw0 = t2 - t1  # To be added to the elapsed time in case we are using CMA Light (information used)
    initial_val_loss = closure(dts_test, model, criterion, device)
    train_accuracy = accuracy(dts_train, model, device)
    test_accuracy = accuracy(dts_test, model, device)
    f_tilde = fw0
    if opt == 'cmal' or opt == 'cmalbd':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)
    history = {'blocks': [], 'update': [], 'train_loss': [fw0], 'val_loss': [initial_val_loss],
               'train_acc': [train_accuracy],
               'val_acc': [test_accuracy], 'step_size': [],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted': [], 'Exit': [], 'comments': [],
               'elapsed_time_noVAL': [0.0]}

    blocks_to_update_list = importance.copy()
    block_before_ep = [[p.data.clone() for p in eval(f'model.layer{block}.parameters()', {'model': model})] for block in
                       importance]  # lasciare se si prevede di disattivare dall'epoca 1

    L = len(importance)  # num. blocks
    val_accuracy_threshold = 0
    count = 0  # number of consecutive epochs without val. accuracy improvement

    # Train
    for epoch in range(ep):
        print(blocks_to_update_list)
        # epoch_update_list = []
        # real_loss = closure(dts_train, model, criterion, device)
        # print(f'Real Loss Before Epoch =  {real_loss}') #
        start_time = time.time()
        if verbose_train: print(f'-------------Epoch {epoch + 1} di {ep}-------------')
        # print(f'Real Loss Before Epoch =  {real_loss}')
        model.train()
        f_tilde = 0
        if opt == 'cmal' or opt == 'cmalbd':
            w_before = get_w(model)

        for x, y in dts_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            f_tilde += loss.item() * (len(x) / P)
            loss.backward()
            optimizer.step()

        if epoch + 1 <= n_ep_cmal:  # first n epochs --> cmal: all the layers are updated
            pass

        else:  # Selection of blocks to freeze or unfreeze
            block_list = [[p for p in eval(f'model.layer{block}.parameters()', {'model': model})] for block in
                          blocks_to_update_list]
            difference = [torch.linalg.norm(torch.cat([p.data.view(-1) for p in block_list[k]]) - torch.cat(
                [p.data.view(-1) for p in block_before_ep[k]])).item() for k in range(len(blocks_to_update_list))]

            deactivation_threshold = beta * np.mean(difference)

            # We froze the least important block whose update is small compared to the average update:
            if difference[0] < deactivation_threshold:
                for p in eval('model.layer' + str(blocks_to_update_list[0]) + '.parameters()'):
                    p.requires_grad = False
                blocks_to_update_list.pop(0)

            # We update layers closer to the output if not sufficient improvement of val. acc for a number of consecutive epochs = patience
            val_accuracy = accuracy(dts_test, model, device, val_check = True, ep = epoch)
            if val_accuracy <= ro * val_accuracy_threshold:
                count += 1
                #print(f'no val acc improvement for {count} consecutive epochs')
                if count == patience:
                    count = 0
                    for block in reversed(importance):
                        if block not in blocks_to_update_list:
                            blocks_to_update_list.insert(0, block)
                            for p in eval('model.layer' + str(block) + '.parameters()'):
                                p.requires_grad = True
                            break

            else:  # val acc sufficiently improved, we update the val acc threshold
                count = 0
                val_accuracy_threshold = val_accuracy

        # epoch_update_list.append(difference)
        if epoch + 1 == n_ep_cmal:
            val_accuracy_threshold = accuracy(dts_test, model, device, val_check = True, ep = epoch)

        if len(blocks_to_update_list) == 0:
            blocks_to_update_list.append(importance[-1])
            for p in eval('model.layer' + str(importance[-1]) + '.parameters()'):
                p.requires_grad = True

        block_before_ep = [[p.data.clone() for p in eval(f'model.layer{block}.parameters()', {'model': model})] for
                           block in blocks_to_update_list]

        # CMAL support functions
        if opt == 'cmal' or opt == 'cmalbd':
            optimizer.set_f_tilde(f_tilde)
            phi = optimizer.phi
            print(f' f_tilde = {f_tilde}   ')
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure,
                                                                   dts_train, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after, phi))
        else:
            f_after = f_tilde

        elapsed_time_4_epoch_noVAL = time.time() - start_time

        # Validation
        model.eval()
        val_loss = closure(dts_test, model, criterion, device)
        train_accuracy = accuracy(dts_train, model, device)
        test_accuracy = accuracy(dts_test, model, device)
        elapsed_time_4_epoch = time.time() - start_time

        # history['update'].append(epoch_update_list)
        history['blocks'].append(blocks_to_update_list.copy())
        history['train_loss'].append(f_after)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(test_accuracy)
        history['time_4_epoch'].append(history['time_4_epoch'][-1] + elapsed_time_4_epoch)
        history['elapsed_time_noVAL'].append(history['elapsed_time_noVAL'][-1] + elapsed_time_4_epoch_noVAL)
        if epoch == 0 and (opt == 'cmal' or opt == 'cmalbd'):
            history['time_4_epoch'][-1] += time_compute_fw0
            history['elapsed_time_noVAL'][-1] += time_compute_fw0

        # Save data during training
        if min_acc < test_accuracy:
            torch.save(model, sm_root + 'model_best.pth')
            min_acc = test_accuracy
            if verbose_train: print('\n - New best Val-ACC: {:.3f} at epoch {} - \n'.format(min_acc, epoch + 1))

        if history['step_size'][-1] <= 1e-15:
            history['comments'] = f'Forced stopping at epoch {epoch}'
            print("Forced stopping")
            break

        torch.save(history, 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '_seed_' + str(seed) +'.txt')
        print('\n')

    if verbose_train: print('\n - Finished Training - \n')
    torch.save(history, 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '_seed_' + str(seed) + '.txt')

    return history


if __name__ == '__main__':
    dts_root = './data'
    bs = 128
    nw = 6
    # inds = range(6000)
    # inds_test = range(600)
    nets = ['resnet18', 'resnet152']
    datasets = ['CIFAR10', 'CIFAR100']
    importance = [3, 4, 1, 2]  # importance of the blocks, ascending order
    for net in nets:
        for dataset in datasets:
            for seed in [1,10,100,1000,10000]:
                # trainset = Subset(eval("torchvision.datasets."+ dataset)(root = dts_root, train = True, download = True, transform = transform), inds)
                trainset = eval("torchvision.datasets." + dataset)(root = dts_root, train = True, download = True,
                                                                   transform = transform)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs,
                                                          shuffle = True)  # , pin_memory = True,num_workers = nw)
                # testset = Subset(eval("torchvision.datasets."+dataset)(root = dts_root, train = False, download = True, transform = transform), inds_test)
                testset = eval("torchvision.datasets." + dataset)(root = dts_root, train = False, download = True,
                                                                  transform = transform)
                testloader = torch.utils.data.DataLoader(testset, batch_size = bs,
                                                         shuffle = False)  # , pin_memory = True,num_workers = nw)

                trainloader = [(x, y) for x, y in trainloader]
                testloader = [(x, y) for x, y in testloader]

                history = train_model(sm_root = '', opt = 'cmalbd', ep = 50, ds = dataset, net_name = net,
                                      history_ID = 'prova', dts_train = trainloader,
                                      dts_test = testloader, verbose_train = True,
                                      zeta = 0.05, eps = 1e-3, theta = 0.5, delta = 0.9, tau = 1e-2, gamma = 1e-6,
                                      verbose = True, verbose_EDFL = True, beta = 1.25, patience = 2, n_ep_cmal = 5,
                                      ro = 1.005, importance = importance, seed = seed)
