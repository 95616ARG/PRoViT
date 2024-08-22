import os, sys
sys.path.append(os.getcwd())
from timeit import default_timer as timer
from tqdm.auto import tqdm
import gurobipy as gp
from gurobipy import GRB
from torch.utils.data import DataLoader
from . import models
import torch
import torchvision
from copy import deepcopy

def accuracy_rec(net, datasets, topk=(1, 5)):
    """Compute the accuracy of {net} on the given datasets.
    If {datasets} is a single dataset, return its top1 and top5 accuracy
    as well as a tuple containing the length of the dataset and the
    (top1, top5) accuracy tuple.
    If {datasets} is a list, return the aggregate top1 and top5
    accuracy for all of the datasets in the list, as well as a tuple
    containing the number of datasets in the list and a list of their
    accuracies.
    If {datasets} is a dictionary, save the top1 and top5 accuracies
    of the datasets in a new dictionary with matching keys. Also return
    the aggregate top1 and top5 accuracy for all datasets and a tuple
    containing the number of datasets in the dictionary and the new
    dictionary with all the accuracies saved.
    """
    if hasattr(datasets, 'accuracy'):
        acc = datasets.accuracy(net, topk=topk)
        return *acc, (len(datasets), acc)

    elif isinstance(datasets, (list, tuple)):
        a1, a5, total_num = 0, 0, 0
        acc_dicts = []
        for d in tqdm(datasets, leave=False, desc='evaluating dataset list'):
            top1, top5, (num, acc_dict) = accuracy_rec(net, d, topk=topk)
            a1 += top1 * num
            a5 += top5 * num
            total_num += num
            acc_dicts.append(acc_dict)
        return a1 / total_num, a5 / total_num, (total_num, acc_dicts)

    elif isinstance(datasets, dict):
        if len(datasets) == 0:
            return 0, 0, {}
        a1, a5, total_num = 0, 0, 0
        acc_dict = {}
        for key, val in tqdm(datasets.items(), total=len(datasets), leave=False, desc='evaluating dataset dict'):
            top1, top5, (num, val_acc_dict) = accuracy_rec(net, val, topk=topk)
            a1 += top1 * num
            a5 += top5 * num
            total_num += num
            acc_dict[key] = val_acc_dict
        return a1 / total_num, a5 / total_num, (total_num, acc_dict)


def repair(net, repair_datasets, eps=0.01, dtype=torch.float32, device='cpu'):
    """Repair {net} according to the images and labels in {repair_datasets}
    using the PRoViT_LP strategy. This strategy only repairs weights incoming
    to the labels that are explicitly present in the repair set. Restricting the
    weights in this manner allows us to conserve memory by reducing the number of
    variables in the model as well as the number of constraints necessary to represent
    the argmax of the symbolic output of the Vision Transformer.

    Return the original network, the repaired last layer of the network,
    the encoding time (seconds) and solve time
    (seconds). If the repair failed, the returned last layer is None.
    """

    if isinstance(net, torchvision.models.VisionTransformer):
        last_layer = net.heads[-1]
    elif net.netname == 'resnet152':
        last_layer = net[10]
    elif net.netname == 'vgg19':
        last_layer = net[3][6]
    else:
        last_layer = net.head

    # Determine the subset of labels present in the repair set.
    set_of_labels = []
    if not isinstance(repair_datasets, list):
        repair_datasets = [repair_datasets]
    for d in repair_datasets:
        for img, label in tqdm(DataLoader(d, batch_size=1), leave=False, desc='unique labels'):
            if label not in set_of_labels:
                set_of_labels.append(int(label))

    # Determine the reduced weight matrix and bias matrix.
    weight_matrix = last_layer.weight.data[set_of_labels,:].to(dtype=dtype).detach().cpu().numpy()
    bias_matrix = last_layer.bias.data[set_of_labels].to(dtype=dtype).detach().cpu().numpy()

    # Initialize Gurobi model and add weights and bias delta variables.
    model = gp.Model()
    model.Params.Crossover = 0
    model.Params.Method = 2
    import os
    model.Params.Threads = os.cpu_count()
    model.Params.Presolve = 1
    model.Params.BarConvTol = 1e-4

    weight_delta_matrix = model.addMVar(shape=tuple(weight_matrix.shape), lb=-10., ub=10.)
    bias_delta_matrix = model.addMVar(shape=tuple(bias_matrix.shape), lb=-10., ub=10.)
    symbolic_biases = bias_matrix + bias_delta_matrix

    start_encoding = timer()
    for d in tqdm(repair_datasets, leave=False, desc='encode datasets'):
        for img, lbl in tqdm(DataLoader(d, batch_size=1), leave=False, desc='encode inputs'):
            if net.netname == 'resnet152':
                concrete_input = net[:10](img).detach().cpu().numpy()[0]
            elif net.netname == 'vgg19':
                concrete_input = net[3][:6](net[:3](img)).detach().cpu().numpy()[0]
            else:
                concrete_input = models.get_encoder_output(net, img).detach().cpu().numpy()[0]
            concrete_out_max = torch.max(net(img)[0]).item()

            # Compute the symbolic output via matrix multiplication with the inputs to the final
            # layer of the Vision Transformer.
            sym_out = (weight_matrix @ concrete_input + (weight_delta_matrix @ concrete_input)) + symbolic_biases

            # Convert the true label to its index in the subset.
            label = set_of_labels.index(lbl)

            # Add constraints such that the output at the label index is greater than other
            # indices of the symbolic output.
            for i in range(sym_out.shape[0]):
                if i != label:
                    model.addConstr(sym_out[label] >= sym_out[i] + eps)

            # Add a constraint that ensures the symbolic output at the label index is greater
            # than the original concrete max of the Transformer output.
            model.addConstr(sym_out[label] >= concrete_out_max + eps)

    # Minimize the l1 norm and linf norm of the weight deltas and bias deltas.
    model.setObjective(norm_ub(weight_delta_matrix, model) + norm_ub(bias_delta_matrix, model), GRB.MINIMIZE)
    end_encoding = timer() - start_encoding

    # Solve.
    start_solve = timer()
    model.optimize()
    end_solve = timer() - start_solve

    succeed = (model.status == GRB.OPTIMAL)
    if succeed:
        # Update the appropriate weights and biases based on the model solution.
        last_layer.weight.data[set_of_labels,:] = torch.from_numpy(weight_delta_matrix.X + weight_matrix).to(dtype=dtype, device=device)
        last_layer.bias.data[set_of_labels] = torch.from_numpy(bias_delta_matrix.X + bias_matrix).to(dtype=dtype, device=device)
        return net, last_layer, end_encoding, end_solve

    else:
        return net, None, end_encoding, end_solve


def norm_ub(mvar, model):
    """Compute the upper bound of the l1 norm + linf norm of the variables in {mvar}.
    Constraints representing these norms are added to {model}.
    """

    def abs_ub(mvar, model):
        """Compute the upper bound of the absolute value of the variables in {mvar}.
        Return the variable matrix representing the absolute values.
        """
        assert mvar.size > 0
        mvar_flat = mvar.reshape(-1)
        Z = model.addMVar(shape=(mvar.size), lb=0.)

        for m, z in zip(mvar_flat, Z):
            model.addConstr(m <= z)
            model.addConstr((-1)*m <= z)
        return Z

    def max_ub(mvar, model):
        """Compute the upper bound of the maximum of the variables in {mvar}.
        Return the variable representing the maximum.
        """
        z = model.addVar()
        model.addConstr(z >= mvar)
        return z

    abs = abs_ub(mvar, model)
    return (abs.sum() / mvar.size) + max_ub(abs, model)

def fine_tune(
    net, repair_datasets,
    dtype=torch.float32, device='cpu',
    batch_size=10,
    max_epoches=100000,
    optimizer='Adam',
    lr=1e-3,
    gamma=0.995,
    validation_interval = 5,
    only_last_layer=True
):
    """Fine tune the network {net} on the images in {repair_datasets}.
    {batch_size} controls the number of inputs computed in each iteration.
    {max_epoches} indicates how many epochs to run before timing out.
    {optimizer} selects the optimizer from torch.optim.
    {lr} controls the learning rate for gradient descent.
    {gamma} is the multiplicative factor of learning rate decay in the
    learning rate scheduler.
    {validation_interval} selects how often to check for 100% accuracy on the
    repair datasets.
    {only_last_layer} selects whether to fine tune just the last layer of the
    network or the entire network.
    """
    net_og = net
    net = deepcopy(net_og)

    for n, p in net.named_parameters():
        p.requires_grad_(False)

    if only_last_layer:
        # Determine the last layer for the network to fine tune.
        if isinstance(net, torchvision.models.VisionTransformer):
            last_layer = net.heads[-1]
        elif net.netname == 'resnet152':
            last_layer = net[10]
        elif net.netname == 'vgg19':
            last_layer = net[3][6]
        elif net.netname == 'deit':
            last_layer = net.head
        else:
            raise NotImplementedError

        ft_params = []
        ft_params_og = []
        for n, p in last_layer.named_parameters():
            print(n)
            p.requires_grad_(True)
            ft_params.append(p)
            ft_params_og.append(p.detach().clone())

    else:
        last_layer = net
        not_ft_params_name = ('class_token', 'norm', 'embeddings', 'classifier', 'pooler', 'shared', 'embed', 'positions')
        ft_params = [p for n, p in net.named_parameters() if all(e not in n for e in not_ft_params_name)]
        for n, p in net.named_parameters():
            p.requires_grad_(all(e not in n for e in not_ft_params_name))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, optimizer)(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Concatenate the datasets in the repair_datasets list.
    concatenated_dataset = torch.utils.data.ConcatDataset(repair_datasets)
    dataloader = torch.utils.data.DataLoader(concatenated_dataset,
        batch_size=batch_size, shuffle=True, drop_last=False)

    net.train()
    pbar = tqdm(range(max_epoches), total=max_epoches)
    epoch_acc = 0.
    efficacy = -1.
    start_time = timer()
    for epoch in pbar:  # loop over the dataset multiple times
        loss_sum = 0.
        num_batches = 0

        for inputs, targets in tqdm(dataloader, leave=False):

            inputs = inputs.to(device,dtype)
            targets = targets.to(device).long().squeeze(-1)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()
            num_batches +=1

        scheduler.step()

        # Check if 100% accuracy has been achieved on the repair datasets. If so, stop fine tuning.
        if epoch > 0 and epoch % validation_interval == 0:
            with torch.no_grad():
                efficacy = accuracy_rec(net, repair_datasets)[0]
                if efficacy == 1.:
                    break

        # Print statistics.
        running_loss = loss_sum / num_batches
        desc = f'[{epoch + 1}] ft acc:{epoch_acc:.3f}, loss: {running_loss:.3f}, efficacy: {efficacy:.2%}, {timer() - start_time:.1f}s'
        print(desc, flush=True)
        pbar.set_description(desc)

    ft_time = timer() - start_time

    return net.eval(), last_layer.eval(), 0, ft_time


def stdlp(net, repair_datasets, eps=0.01, dtype=torch.float32, device='cpu'):
    """Repair {net} according to the images and labels in {repair_datasets}
    using the STD_LP strategy. This strategy is the standard last layer linear programming
    method used by APRNN and Minimal Modifications of Deep Neural Networks.

    Return the original network, the repaired last layer of the network,
    the encoding time (seconds) and solve time
    (seconds). If the repair failed, the returned last layer is None.
    """

    if isinstance(net, torchvision.models.VisionTransformer):
        last_layer = net.heads[-1]
    elif net.netname == 'resnet152':
        last_layer = net[10]
    elif net.netname == 'vgg19':
        last_layer = net[3][6]
    else:
        last_layer = net.head

    # Determine the reduced weight matrix and bias matrix.
    weight_matrix = last_layer.weight.data.to(dtype=dtype).detach().cpu().numpy()
    bias_matrix = last_layer.bias.data.to(dtype=dtype).detach().cpu().numpy()

    # Initialize Gurobi model and add weights and bias delta variables.
    model = gp.Model()
    model.Params.Crossover = 0
    model.Params.Method = 2
    import os
    model.Params.Threads = os.cpu_count()
    model.Params.Presolve = 1
    model.Params.BarConvTol = 1e-4
    weight_delta_matrix = model.addMVar(shape=tuple(weight_matrix.shape), lb=-10., ub=10.)
    bias_delta_matrix = model.addMVar(shape=tuple(bias_matrix.shape), lb=-10., ub=10.)
    symbolic_biases = bias_matrix + bias_delta_matrix

    start_encoding = timer()
    for d in tqdm(repair_datasets, leave=False, desc='encode datasets'):
        for img, lbl in tqdm(DataLoader(d, batch_size=1), leave=False, desc='encode inputs'):
            if net.netname == 'resnet152':
                concrete_input = net[:10](img).detach().cpu().numpy()[0]
            elif net.netname == 'vgg19':
                concrete_input = net[3][:6](net[:3](img)).detach().cpu().numpy()[0]
            else:
                concrete_input = models.get_encoder_output(net, img).detach().cpu().numpy()[0]
            concrete_out_max = torch.max(net(img)[0]).item()

            # Compute the symbolic output via matrix multiplication with the inputs to the final
            # layer of the Vision Transformer.
            sym_out = (weight_matrix @ concrete_input + (weight_delta_matrix @ concrete_input)) + symbolic_biases

            # Add constraints such that the output at the label index is greater than other
            # indices of the symbolic output.
            for i in range(sym_out.shape[0]):
                if i != lbl:
                    model.addConstr(sym_out[lbl] >= sym_out[i] + eps)

            # Add a constraint that ensures the symbolic output at the label index is greater
            # than the original concrete max of the Transformer output.
            model.addConstr(sym_out[lbl] >= concrete_out_max + eps)

    # Minimize the l1 norm and linf norm of the weight deltas and bias deltas.
    model.setObjective(norm_ub(weight_delta_matrix, model) + norm_ub(bias_delta_matrix, model), GRB.MINIMIZE)
    end_encoding = timer() - start_encoding

    # Solve.
    start_solve = timer()
    model.optimize()
    end_solve = timer() - start_solve

    succeed = (model.status == GRB.OPTIMAL)
    if succeed:
        # Update the appropriate weights and biases based on the model solution.
        last_layer.weight.data = torch.from_numpy(weight_delta_matrix.X + weight_matrix).to(dtype=dtype, device=device)
        last_layer.bias.data = torch.from_numpy(bias_delta_matrix.X + bias_matrix).to(dtype=dtype, device=device)
        return net, last_layer, end_encoding, end_solve

    else:
        return net, None, end_encoding, end_solve
