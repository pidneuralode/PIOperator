# construct the work flow for burgers1d dataset
import argparse
import torch
import sys
from model.deepONet.deepONet1d import DeepONet
from builders.burgers1d import Burgers1dBuilder
from utils import trainer
from utils.optim import instantiate_optimizer, instantiate_scheduler

parser = argparse.ArgumentParser('build the train-test flow for burgers1d with deeponet')

# parse the argument from the terminal
parser.add_argument('--gpu',
                    default=0,
                    help='choice of gpu'
                    )
parser.add_argument('--batch-size',
                    default=16,
                    help='number of batch size to train the model'
                    )
parser.add_argument('--weight-decay',
                    default=0,
                    help='weight decay to regular the weight matrix of network'
                    )
parser.add_argument('--lr',
                    default=0.001,
                    help='the learning rate for this model'
                    )
parser.add_argument('--epochs',
                    default=500000,
                    help='the number of inters in process'
                    )
parser.add_argument('--result-file',
                    default='E:\\dfno\\PIOperator\\results\\burgers1d\\loss_deeponet.txt',
                    help='the path file to store the result of this experiment'
                    )
parser.add_argument('--model-path',
                    default='E:\\dfno\\PIOperator\\results\\burgers1d\\model_deeponet.pth',
                    type=str,
                    help='the path to save the neural model'
                    )
parser.add_argument('--n-train',
                    default=1000,
                    type=int,
                    help='the numbers of trainset'
                    )
parser.add_argument('--n-test',
                    default=200,
                    type=int,
                    help='the numbers of testset'
                    )
parser.add_argument('--resolution',
                    default=128,
                    type=int,
                    help='grid size')
# resolve the args
args = parser.parse_args()
config = {}
for key, value in vars(args).items():
    config[key] = value


def main():
    # define the device and update the information of data config
    device = f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')
    config['device'] = device

    # model
    burgers_model = DeepONet([128, 128, 128, 128, 128],
                             [1, 128, 128, 128],
                             "tanh",).to(device=device)

    # load data from cylinder pickle ans csv files
    path = 'E:\\dfno\\PIOperator\\data\\burgers\\1d\\burgers_1d.mat'
    burgers_dataloader = Burgers1dBuilder(config['n_train'],
                                          config['n_test'],
                                          config['resolution'],
                                          path,
                                          'DeepONet',
                                          batch_size=config['batch_size']
                                          )

    # optimizer
    optimizer = instantiate_optimizer(burgers_model, config)

    # scheduler
    scheduler = instantiate_scheduler(optimizer, config)

    # train the whole process for cylinder
    trainer.train(model=burgers_model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  tr_loader=burgers_dataloader.train_dataloader(),
                  tt_loader=burgers_dataloader.test_dataloader(),
                  config=config
                  )


if __name__ == '__main__':
    main()
