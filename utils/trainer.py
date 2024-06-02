import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from timeit import default_timer
import sys
import torch.nn.functional as F
from .logger import Logger


# fomat the record info with standard process
def str_rec(names, data, unit=None, sep=', ', presets='{}'):
    if unit is None:
        unit = [''] * len(names)
    data = [str(i)[:6] for i in data]
    out_str = "{}: {{}} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str


# calculate the relative error about the predict value and target value
def calculate_mre(pred_y, test_y):
    # calculate batch error
    return sum((pred_y[i] - test_y[i]).norm().item() / test_y[i].norm().item() for i in range(pred_y.shape[0]))


# calculate the mae metric for the test data
def calculate_mae(pred_y, test_y):
    return F.l1_loss(pred_y, test_y).item() * pred_y.shape[0]


# calculate the max_mae metric for the test data
def calculate_max_mae(pred_y, test_y):
    return torch.sum(torch.max(torch.abs(test_y - pred_y).flatten(1), dim=1)[0]).item()


@torch.no_grad()
def evaluation(config, loss_fn, model, tt_loader, logger, epoch):
    dsize = 0
    test_loss, test_mre, test_mae, test_max_mae = 0.0, 0.0, 0.0, 0.0
    itrcnt = 0
    start_time = default_timer()
    for test_x, test_y in tqdm(tt_loader):
        itrcnt += 1
        dsize += test_y.shape[0]
        if isinstance(test_x, list):
            for i in range(len(test_x)):
                test_x[i] = torch.tensor(test_x[i]).to(config['device'])
        else:
            test_x = torch.tensor(test_x).to(config['device'])
        test_y = test_y.to(config['device'])
        pred_y = model(test_x)

        test_loss += loss_fn(pred_y, test_y).detach().cpu().numpy()
        test_mre += calculate_mre(pred_y, test_y)
        test_max_mae += calculate_max_mae(pred_y, test_y)
        test_mae += calculate_mae(pred_y, test_y)
    end_time = default_timer()
    test_loss /= itrcnt
    test_mre /= dsize
    test_mae /= dsize
    test_max_mae /= dsize

    print(f'the test loss is {test_loss}')
    print(f'the test relative mean error is {test_mre}')
    print(f'the test mae is {test_mae}')
    print(f'the test max mae is {test_max_mae}')
    print(f'testing time is: {end_time - start_time:.2f}.seconds')
    print_info = ['test', epoch, test_loss, end_time - start_time, test_mre, test_mae, test_max_mae]
    logger.append(print_info)


# train and record corresponding data for the cylinder
def train(model, optimizer, scheduler, tr_loader, tt_loader, config):
    itrcnt = 0
    loss_fn = nn.MSELoss()
    model_path = config.get('model_path')
    logger = Logger(config['result_file'])
    logger.set_names(['Train/Test', 'epoch', 'Loss', 'Time', 'MRE', 'MAE', 'MAX_MAE'])
    start_time = default_timer()
    for epoch in range(config['epochs']):
        # train the model for each epoch and record the history
        train_size = 0
        train_loss, train_mre, train_mae, train_max_mae = 0.0, 0.0, 0.0, 0.0
        for x, y in tqdm(tr_loader):
            train_size += y.shape[0]
            optimizer.zero_grad()
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = torch.tensor(x[i]).to(config['device'])
            else:
                x = torch.tensor(x).to(config['device'])
            y = y.to(config['device'])
            pred_y = model(x)
            loss = loss_fn(pred_y, y)
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            optimizer.step()
            itrcnt += 1
            with torch.no_grad():
                train_mre += calculate_mre(pred_y, y)
                train_mae += calculate_mae(pred_y, y)
                train_max_mae += calculate_max_mae(pred_y, y)
        scheduler.step()
        end_time = default_timer()
        train_loss /= itrcnt
        train_mre /= train_size
        train_mae /= train_size
        train_max_mae /= train_size

        if epoch % 100 == 0:
            print(f'training {epoch} th epoch time: {end_time - start_time:.2f}.seconds')
            print(f'the train loss is {train_loss}')
            print(f'the train relative mean error is {train_mre}')
            print(f'the train mae is {train_mae}')
            print(f'the train max mae is {train_max_mae}')

            print_info = ['train', epoch, train_loss, end_time - start_time, train_mre, train_mae, train_max_mae]
            # record the train loss into the result file
            logger.append(print_info)
            # test the model loss and record the result into the csv files
            evaluation(config, loss_fn, model, tt_loader, logger, epoch)

            # save the voronoi-model
            if epoch % 500 == 0 and model_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_fn,
                    'lr_state_dict': scheduler.state_dict(),
                }, model_path)

    logger.close()



if __name__ == '__main__':
    # test mre function
    y1 = torch.Tensor([[1, 2, 3], [1, 2, 3]])
    y2 = torch.Tensor([[2, 2, 4], [2, 2, 4]])
    print(calculate_mre(y1, y2))
    print('test')






