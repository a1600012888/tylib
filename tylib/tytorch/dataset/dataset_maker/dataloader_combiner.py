import torch

# torch.utils.data.ChainDataset
# torch.utils.data.ConcatDataset
# all above is for dataset combiner!
# we want data loader combiner!

class _DataloaderCombiner(object):
    '''
    uniform sampling for each loader
    '''

    def __init__(self, loaders):

        self.loaders = []
        for loader in loaders:
            self.loaders.append(enumerate(loader))
        #self.loaders = loaders
        self.num_loader = len(loaders)
        self.lens = [len(loader) for loader in loaders]
        total_len = 0
        for l in self.lens:
            total_len += l
        self.total_len = total_len
        self.start_idx = 0


    def __next__(self):

        idx = self.start_idx % 3
        self.start_idx += 1
        loader = self.loaders[idx]

        i, batch = next(loader)
        #print(i, idx)
        #print(self.lens)

        if i == self.lens[idx]-1:
            self.loader.pop(idx)
            self.num_loader -= 1
            self.lens.pop(idx)
        return batch

    def __len__(self):
        return self.total_len

class DataloaderCombiner(object):
    '''
    uniform sampling for each loader
    '''

    def __init__(self, loaders):

        self.num_loader = len(loaders)
        self.lens = [len(loader) for loader in loaders]
        total_len = 0
        for l in self.lens:
            total_len += l
        self.total_len = total_len
        self.dataloader_combiner = _DataloaderCombiner(loaders)

    def __iter__(self):
        return self.dataloader_combiner

    def __len__(self):
        return self.total_len

