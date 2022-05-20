import torch
from . import dataset
from ray import tune
from functools import partial

class NeuralNet(torch.nn.Module):
    def __init__(self, l1, l2):
        super(NeuralNet, self).__init__()

        self.block1 = torch.nn.Sequential(
                    torch.nn.Linear(40,l1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(l1,l2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(l2,1),
                    torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.block1(x.float())
        return y

def train_loop(model, train_dataloader, loss_fn, optimizer, device):

    no_correct = 0
    total = 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        #make prediction
        prediction = model.forward(X)
        no_correct += (torch.round(torch.squeeze(prediction)) == y).sum().item()
        total += len(y)
        loss = loss_fn(torch.squeeze(prediction).float(), y.float())

        #backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = loss.item()
    train_accuracy = no_correct / total
    return train_loss, train_accuracy

def test_loop(model, test_dataloader, loss_fn, device):
    no_correct = 0
    total = 0
    validation_loss = 0
    with torch.no_grad():
        for X,y in test_dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            no_correct += (torch.round(torch.squeeze(prediction)) == y).sum().item()
            total += len(y)

            validation_loss += loss_fn(torch.squeeze(prediction).float(), y.float()).item()

    validation_loss /= len(test_dataloader)
    validation_accuracy = no_correct / total
    return validation_loss, validation_accuracy

def train_model(config: dict, seed=int):

    #load data
    torch.manual_seed(seed)
    df, feature_list = dataset.load_data()
    train_dataloader, val_dataloader, test_dataloader = dataset.split_data(df, feature_list, config['batch_size'])

    #instantiate model, optimizer, and loss functions
    model = NeuralNet(config['l1'], config['l2'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.BCELoss()

    for epoch in range(0, config['epochs']):
        
        #for each epohc, train model, validate performance, and communicate to tune to track performance
        train_loss, train_acc = train_loop(model, train_dataloader, loss_fn, optimizer, device) 
        test_loss, test_acc = test_loop(model, val_dataloader, loss_fn, device)

        tune.report(test_loss=test_loss, test_accuracy=test_acc, train_loss=train_loss, train_accuracy=train_acc) 

def tune_model(path2output: str, seed: int, samples: int, config: dict, cpus=1, gpus=0, max_epochs=2000):
    """Wrapper function for train model"""

    scheduler = tune.schedulers.ASHAScheduler(metric='test_loss', mode='min', max_t=max_epochs, grace_period=1, reduction_factor=2)
    result = tune.run(
        partial(train_model, seed=seed),
        resources_per_trial={'cpu': cpus, "gpu": gpus},
        config=config,
        num_samples=samples,
        scheduler=scheduler,
        local_dir=path2output)



    
