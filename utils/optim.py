import torch


# choose the instance of a scheduler
def instantiate_scheduler(optimizer, config):
    # define the default properties
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.95)

    if config.get('sheduler') == 'ex':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    elif config.get('sheduler') == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    elif config.get('sheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    elif config.get('sheduler') == 'multi-step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 1000], gamma=0.1)
    elif config.get('sheduler') == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=500)
    elif config.get('sheduler') == 'cos-warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.01)

    return scheduler


# choose the instance of an optimizer
def instantiate_optimizer(model, config):
    # use adam as the default optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if config.get('optim') == 'sgd-m':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                                    momentum=0.9)
    elif config.get('optim') == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    return optimizer