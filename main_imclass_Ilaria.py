import torch
import numpy as np
from utility_Ilaria import get_pretrained_net, set_optimizer, get_w, set_dataset, closure, accuracy
import os, time
from warnings import filterwarnings
filterwarnings('ignore')

def freeze_layers(model, blocks_to_update_list):
    layer = getattr(model, f'layer{blocks_to_update_list[0]}', None)
    if layer is not None:
        for p in layer.parameters():
            p.requires_grad = False
    return blocks_to_update_list[1:]  # Alternatively, blocks_to_update_list.pop(0)

def train_model(dtl_train,
                dtl_test,
                arch: str,
                sm_root: str,
                opt: str,
                ep: int,
                time_limit: int,
                seed: int,
                importance: list,
                ID_history = '',
                verbose_train = False,
                device = 'cpu',
                criterion = torch.nn.CrossEntropyLoss(),
                ds = 'cifar10',
                savemodel = False,
                modelpth = None,
                beta=1.25,
                patience=3,
                ro=1.005,
                n_ep_cmal=5,
                *args,
                **kwargs):

    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    P = (len(dtl_train)-1) * (len(dtl_train[0][0])) + len(dtl_train[-1][0])
    if device is None: device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
    device = torch.device(device)
    torch.cuda.empty_cache()
    num_classes = 10 if ds != 'cifar100' else 100
    if modelpth is not None:
        model = torch.load(modelpth,map_location='cpu').to(device)
    else:
        model = get_pretrained_net(arch,seed,num_classes).to(device)

    #if ds == 'mnist':
    #    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
    #if ds == 'tinyimagenet':
    #    num_ftrs = model.fc.in_features
    #    model.fc = torch.nn.Linear(num_ftrs, 200).to(device)

    optimizer = set_optimizer(opt,model,*args,**kwargs)
    print(f"ds: {ds}   arch: {arch}   n: {sum(p.numel() for p in model.parameters())}")


    model.eval()
    elapsed_time = 0.0
    best_accuracy = 0.0
    t1 = time.time()
    fw0 = closure(dataset=dtl_train, mod=model, loss_fun=criterion, device=device)
    t2 = time.time()
    time_compute_fw0 = t2 - t1  # To be added to the elapsed time in case we are using CMA Light (information used)
    #if opt == 'sgd':
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    initial_val_loss = closure(dataset=dtl_test, mod=model, loss_fun=criterion, device=device)
    train_accuracy = accuracy(dataset=dtl_train, mod=model, device=device)
    val_acc = accuracy(dataset=dtl_test, mod=model, device=device)
    f_tilde = fw0
    if verbose_train: print(f'Initial stats. fw0 ={fw0}    val_acc={val_acc}    time_to_compute_fw0={time_compute_fw0}')
    if opt == 'cmal' or opt == 'cmalbd':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)
    #if opt == 'cma':
    #    optimizer.set_fw0(fw0)
    #    optimizer.set_reference(fw0)

    history = {'blocks': [], 'train_loss': [fw0], 'val_loss': [initial_val_loss], 'time_4_epoch': [0.0], 'step_size': [],
               'accepted': [], 'nfev': 1, 'Exit':[], 'val_accuracy':[val_acc], 'to_compute_obj':[time_compute_fw0],
               'train_accuracy':[train_accuracy],'f_tilde':[fw0]}

    if optimizer in {'cma','ig','cmal','cmalbd'}:
        history['step_size'].append(optimizer.param_groups[0]['zeta'])

    blocks_to_update_list = importance.copy()
    block_before_ep = [[p.data.clone() for p in getattr(model, f'layer{block}').parameters()] for block in importance]

    val_accuracy_threshold = 0
    count = 0  # number of consecutive epochs without val. accuracy improvement

    # Training cycle - Epochs Level
    if verbose_train: print(f"\n Start Train. Dataset:{ds}  Net:{arch}   Opt:{opt}   Seed:{seed}  N:{num_classes}\n")
    for epoch in range(ep):
        model.train()
        start_epoch_time = time.time()
        if verbose_train: print(f'Epoch n. {epoch+1}/{ep}')
        print(blocks_to_update_list)
        if elapsed_time >= time_limit:
            break
        if opt == 'cmal' or opt=='cmalbd':
            w_before = get_w(model)
        # Training cycle - Single epoch level
        f_tilde = 0
        j = 0
        for x,y in dtl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            f_tilde += loss.item() * (len(x) / P)
            if verbose_train and j%25 == 0: print(f'batch {j+1} batch_loss = {loss.item()}')
            loss.backward()
            optimizer.step()
            j += 1

        history['f_tilde'].append(f_tilde)
        #if opt == 'sgd':
        #    scheduler.step()
        #print(f'Loss after IC = {closure(dataset=dtl_train, mod=model, loss_fun=criterion, device=device)}')

        # CMA support functions
        #if opt == 'cma':
        #    model, history, f_before, f_after, exit_type = optimizer.control_step(model,w_before,closure,
        #                                            dtl_train,device,criterion,history,epoch)
        #    optimizer.set_reference(f_before=f_after)
        #    if history['step_size'][-1] <= 1e-15:
        #        history['comments'] = f'Forced stopping at epoch {epoch}'
        #        break

        if epoch + 1 <= n_ep_cmal:  # first n epochs --> cmal: all the layers are updated
            pass
        else:  # Selection of blocks to freeze or unfreeze
            block_list = [[p for p in getattr(model, f'layer{block}').parameters()] for block in blocks_to_update_list]
            if len(blocks_to_update_list) > 1:
                difference = [torch.linalg.norm(torch.cat([p.data.view(-1) for p in block_list[k]]) -
                                                torch.cat([p.data.view(-1) for p in block_before_ep[k]])).item()
                              for k in range(len(blocks_to_update_list))]
                deactivation_threshold = beta * np.mean(difference)
                if difference[0] < deactivation_threshold:
                    blocks_to_update_list = freeze_layers(model, blocks_to_update_list)

            val_accuracy = accuracy(dtl_test, model, device, val_check=True, ep=epoch)
            if val_accuracy <= ro * val_accuracy_threshold:
                count += 1
                if count == patience:
                    count = 0
                    for block in reversed(importance):
                        if block not in blocks_to_update_list:
                            blocks_to_update_list.insert(0, block)
                            for p in getattr(model, f'layer{block}').parameters():
                                p.requires_grad = True
                            break
            else:
                count = 0
                val_accuracy_threshold = val_accuracy

        if epoch + 1 == n_ep_cmal:
            val_accuracy_threshold = accuracy(dtl_test, model, device, val_check=True, ep=epoch)

        block_before_ep = [[p.data.clone() for p in getattr(model, f'layer{block}').parameters()] for block in blocks_to_update_list]


        #CMAL support functions
        if opt=='cmal' or opt == 'cmalbd':
            optimizer.set_f_tilde(f_tilde)
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure,
                                                                   dtl_train, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after, optimizer.phi))

            #if history['step_size'][-1] <= 1e-15:
            #    history['comments'] = f'Forced stopping at epoch {epoch}'
            #    break
        else:
            pass

        # Update history
        model.eval()
        val_loss = closure(dataset=dtl_test, mod=model, loss_fun=criterion, device=device)
        val_acc = accuracy(dataset=dtl_test, mod=model, device=device)
        elapsed_time += time.time()-start_epoch_time
        real_train_loss = closure(dataset=dtl_train, mod=model, loss_fun=criterion, device=device)
        train_accuracy = accuracy(dataset=dtl_train,mod=model,device=device)
        if epoch == 0 and opt in {'cmal','cmalbd'}:
            elapsed_time += time_compute_fw0

        history['blocks'].append(blocks_to_update_list.copy())
        history['time_4_epoch'].append(elapsed_time)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['train_accuracy'].append(train_accuracy)
        history['train_loss'].append(real_train_loss)

        if val_acc > best_accuracy and savemodel==True:
            torch.save(model,'best_model_'+opt+'_'+arch+'_'+ds+'_'+ID_history+'.pth')
            best_accuracy = val_acc
            if verbose_train: print(f'New best model with {best_accuracy} validation accuracy :)')
        #if opt == 'ig':
        #    history['step_size'].append(optimizer.param_groups[0]['zeta'])
        #    optimizer.update_zeta()
            if history['step_size'][-1] <= 1e-15: # ?
                history['comments'] = f'Forced stopping at epoch {epoch}'
                break

        if verbose_train: print(f'End Epoch {epoch}   Train Loss:{history["train_loss"][-1]:3e}  Time:{elapsed_time:2f}   Train_acc:{train_accuracy:2f}    Val_acc: {val_acc:2f}   f_tilde: {f_tilde} \n ')

        #torch.save(history, sm_root + 'history_' + opt + '_' + arch + '_' + ds + '_' + ID_history + '.txt')
        # Empty CUDA cache
        torch.cuda.empty_cache()


    # Operations after training
    torch.save(history,sm_root + 'history_'+opt+'_'+arch+'_'+ds+'_'+ID_history+'.txt')
    if verbose_train: print('\n - Finished Training - \n')
    return history


if __name__ == "__main__":
    seeds = [1,10,100,1000,10000]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smroot = ''
    verbose_train = True
    verbose = True
    verbose_EDFL = True
    os.chdir('prove_Ilaria')
    ds = 'cifar100'
    n_ep = 50
    importance = [3, 4, 1, 2]  # importance of the blocks, ascending order
    for rete in ['resnet152']:
        print('rete----- ', rete)

        for seed in seeds:
            print('CMAL with seed = ', seed)
            #subset = (range(2048), range(128))
            torch.manual_seed(seed)

            train_loader, test_loader = set_dataset(ds=ds, bs=32, RR=False, subset=None, seed=seed, to_list=True)

            history_cmal = train_model(dtl_train=train_loader,dtl_test=test_loader,
            ds=ds, arch=rete, sm_root=smroot, opt='cmalbd', ep=n_ep, time_limit=5000000,
            max_it_EDFL=100, ID_history='seed_' + str(seed), zeta=1e-2,
            theta=0.75,delta=0.9, gamma=1e-6, verbose=verbose, tau=1e-2, verbose_EDFL=verbose_EDFL,
            verbose_train=verbose_train, seed=seed,device=device,savemodel=True,beta = 1.4, patience = 3, n_ep_cmal = 5,
                              ro = 1.005, importance = importance)


        print('FINE RETE --- \n')


