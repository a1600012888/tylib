import torch
import numpy as np

class CropDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, num, ratio_range=[0.3, 0.9]):

        if isinstance(dataset, tuple):
            print('using pre-defined partition')
            label_data, data = dataset
        else:
            print('random sampling test set!')
            label_data, data = torch.utils.data.random_split(dataset,
                                                             [num, len(dataset)-num])
        self.num_img = len(data)
        self.num_classes = len(label_data)
        self.label_data = label_data
        self.data = data
        self.ratio_range = ratio_range

    def __len__(self):
        return self.num_img

    def __getitem__(self, item):
        ratio_h = np.random.uniform(*self.ratio_range)
        ratio_w = np.random.uniform(*self.ratio_range)

        start_h = np.random.uniform(low=0, high=ratio_h)
        start_w = np.random.uniform(low=0, high=ratio_w)

        label = np.random.randint(low=0, high=self.num_classes)
        label_img, _ = self.label_data[label]

        img, _ = self.data[item]

        height_of_img, width_of_img = img.size(1), img.size(2)

        #print(img.size(), type(img))
        x1, x2 = int(start_h * height_of_img), int(ratio_h * height_of_img)
        y1, y2 = int(start_w * height_of_img), int(ratio_w * height_of_img)

        #print(img.size(), type(img), x1, x2, y1, y2)

        img[:,x1:x2, y1:y2] = label_img[:,x1:x2, y1:y2]

        return img, label
