import torch

# torch.utils.data.ChainDataset
# torch.utils.data.ConcatDataset
# all above is for dataset combiner!
# we want data loader combiner!

class DataloaderCombiner(object):
    '''
    uniform sampling for each loader
    '''

    def __init__(self, loaders):

        self.loaders = loaders
        self.num_loader = len(loaders)


    def __iter__(self):

        idx = int(torch.randint(0, self.num_loader, size=(1))[0])
        loader = self.loaders[idx]

        batch = next(loader)

        yield batch