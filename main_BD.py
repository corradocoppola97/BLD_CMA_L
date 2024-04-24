import time

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
                *args,
                **kwargs) -> dict:
    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    torch.manual_seed(42)
    model = get_pretrained_net(net_name).to(device)
    if verbose_train: print('\n The model has: {} trainable parameters'.format(count_parameters(model)))

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = set_optimizer(opt, model, *args, **kwargs)

    # Initial Setup
    min_acc = 0
    t1 = time.time()
    fw0 = closure(dts_train, model, criterion, device)
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
    history = {'update': [], 'train_loss': [fw0], 'val_loss': [initial_val_loss], 'train_acc': [train_accuracy],
               'val_acc': [test_accuracy], 'step_size': [],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted': [], 'Exit': [], 'comments': [],
               'elapsed_time_noVAL': [0.0]}

    # All the req.grad = False, we freeze all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Train
    for epoch in range(ep):
        epoch_update_list = []
        real_loss = closure(dts_train, model, criterion, device)
        start_time = time.time()
        if verbose_train: print(f'-------------Epoch {epoch + 1} di {ep}-------------')
        print(f'Real Loss Before Epoch =  {real_loss}')
        model.train()
        f_tilde = 0
        if opt == 'cmal' or opt == 'cmalbd':
            w_before = get_w(model)
        for x, y in dts_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            count = 0
            batch_update_list = []
            # we update the layers one by one, in a backward order
            for p in reversed(list(model.parameters())):
                count += 1
                p_before = p.data.clone()
                p.requires_grad = True
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if count == 1: # we compute the estimate of the loss before when processing a new batch
                    f_tilde += loss.item() * (len(x) / len(dts_train.dataset))
                loss.backward()
                optimizer.step()  # only params with req.grad = True will be updated
                p.requires_grad = False
                p_after = p.data
                batch_update_list.append(torch.linalg.norm(p_after - p_before).item()) #we save the update for every layer
            epoch_update_list.append(batch_update_list)

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

        # print("train accuracy",train_accuracy)
        # print("test accuracy",test_accuracy)

        history['update'].append(epoch_update_list)
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

        torch.save(history, 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')
        print('\n')

    if verbose_train: print('\n - Finished Training - \n')
    torch.save(history, 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')

    return history


if __name__ == '__main__':
    dts_root = './data'
    bs = 128
    nw = 6
    inds = range(6000)
    inds_test = range(600)
    trainset = Subset(
        torchvision.datasets.CIFAR10(root = dts_root, train = True, download = True, transform = transform), inds)
    # trainset = torchvision.datasets.CIFAR10(root=dts_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True, pin_memory = True,
                                              num_workers = nw)
    testset = Subset(
        torchvision.datasets.CIFAR10(root = dts_root, train = False, download = True, transform = transform), inds_test)
    # testset = torchvision.datasets.CIFAR10(root=dts_root, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = False, pin_memory = True,
                                             num_workers = nw)
    history = train_model(sm_root = '', opt = 'cmalbd', ep = 100, ds = 'CIFAR10', net_name = 'resnet18',
                          history_ID = 'prova', dts_train = trainloader, dts_test = testloader, verbose_train = True,
                          zeta = 0.05, eps = 1e-3, theta = 0.5, delta = 0.9, tau = 1e-2, gamma = 1e-6,
                          verbose = True, verbose_EDFL = True)
