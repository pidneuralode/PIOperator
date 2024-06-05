# construct the work flow for burgers1d dataset
import argparse
import torch
import sys
from model.deepONet.PI_deepONet1d import PIDeepONet
from builders.burgers2d_physics import Burgers2dBuilder
from utils import trainer
from utils.optim import instantiate_optimizer, instantiate_scheduler
from utils.physics_loss import Burgers2dPhyLoss

parser = argparse.ArgumentParser('build the train-test flow for burgers2d with pi-deeponet')

# parse the argument from the terminal
parser.add_argument('--gpu',
                    default=0,
                    help='choice of gpu'
                    )
parser.add_argument('--batch-size',
                    default=50000,
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
                    default='E:\\dfno\\PIOperator\\results\\burgers2d\\loss_pideeponet.txt',
                    help='the path file to store the result of this experiment'
                    )
parser.add_argument('--model-path',
                    default='E:\\dfno\\PIOperator\\results\\burgers2d\\model_pideeponet.pth',
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
    burgers_model = PIDeepONet([101, 128, 128, 128, 128],
                             [2, 128, 128, 128],
                             "tanh",).to(device=device)

    # load data from cylinder pickle ans csv files
    path = 'E:\\dfno\\PIOperator\\data\\burgers\\2d\\burgers_2d.mat'
    burgers_dataloader = Burgers2dBuilder(config['n_train'],
                                          config['n_test'],
                                          path,
                                          config['device'],
                                          batch_size=config['batch_size']
                                          )

    # optimizer
    optimizer = instantiate_optimizer(burgers_model, config)

    # scheduler
    scheduler = instantiate_scheduler(optimizer, config)

    # physical loss
    burgers_loss = Burgers2dPhyLoss(burgers_model)

    # train the whole process for cylinder
    trainer.train_PI_DeepONet(model=burgers_model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  tr_loader=burgers_dataloader.train_dataloader(),
                  tt_loader=burgers_dataloader.test_dataloader(),
                  config=config,
                  physic_loss=burgers_loss
                  )


if __name__ == '__main__':
    main()
