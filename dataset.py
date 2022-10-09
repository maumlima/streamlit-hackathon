from torch.utils.data import Dataset
import pandas as pd
from matplotlib import image
from collections import defaultdict

class HackathonDataset(Dataset):
    def __init__(self, type: str == 'train'):
        super(HackathonDataset, self).__init__()

        self.type = type

        data = pd.read_csv('ai_ready/x-ai_data.csv')
        temp = data[data['split'] == type].T.to_dict()

        i = 0
        self.data = defaultdict()
        for key in temp:
            self.data[i] = defaultdict()
            file = temp[key]['filename']
            self.data[i]['filename'] = file
            self.data[i]['class'] = temp[key]['class']
            self.data[i]['split'] = temp[key]['split']
            img = image.imread(f'ai_ready/images/{file}')
            mask = image.imread(f'ai_ready/masks/{file}')

            self.data[i]['img'] = img
            self.data[i]['mask'] = mask
            i += 1

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
