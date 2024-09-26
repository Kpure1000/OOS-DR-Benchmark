import h5py
import numpy as np

class DatasetBase:

    def __init__(self, name, path, has_labels):
        self.name = name
        self.path = path
        self.has_labels = has_labels

    def load(self):
        with h5py.File(self.path, 'r') as f:
            
            E = f['E'][:] if 'E' in f else None
            O = f['O'][:] if 'O' in f else None

            if len(E) != len(O):
                raise ValueError('E and O must have the same size')
            
            data = [{'E': np.array(e), 'O': np.array(o)} for e, o in zip(E, O)]
            labels = np.array(f['labels']) if (self.has_labels and 'labels' in f) else None

            self.data = data
            self.labels = labels

            return data, labels
        
    def loaded(self):
        return self.data is not None

    def group_size(self):
        return len(self.data)


class TruthDataset(DatasetBase):
    def __init__(self, name, path, has_labels):
        super().__init__(name, path, has_labels)


class SyntheticDataset(DatasetBase):
    def __init__(self, name, path, has_labels):
        super().__init__(name, path, has_labels)


class DatasetManager:
    def __init__(self):
        self.datasets = {}

    def register(self, dataset):
        self.datasets[dataset.name] = dataset

    def get(self, name):
        return self.datasets[name]
