from torch.utils.data.sampler import Sampler
import numpy as np
import random
import copy
import torch

from collections import defaultdict

# three Samplers:BalancedSampler\ClassMiningSampler\RandomIdentitySampler

class BalancedSampler(Sampler):

    def __init__(self, dataset, batch_size, n_instance):
        n_classes = batch_size // n_instance
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.label_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }
        for label in self.label_indices:
            np.random.shuffle(self.label_indices[label])
        self.used_ind_pos = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_instance
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        print("batch_size:",self.batch_size,"n_samples:",self.n_samples, "n_classes:",self.n_classes)
        super(BalancedSampler, self).__init__(dataset)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for c in classes:
                c_indices = self.label_indices[c]
                pos = self.used_ind_pos[c]
                indices.extend(c_indices[pos:pos+self.n_samples])
                self.used_ind_pos[c] += self.n_samples
                if self.used_ind_pos[c] + self.n_samples > len(self.label_indices[c]):
                    np.random.shuffle(self.label_indices[c])
                    self.used_ind_pos[c] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


class ClassMiningSampler(Sampler):

    def __init__(self, dataset, batch_size, n_instance, balanced=False):
        n_classes = batch_size // n_instance
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.label_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }
        for label in self.label_indices:
            np.random.shuffle(self.label_indices[label])
        self.used_ind_pos = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_instance
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        self.centers = []
        self.dist = [0]
        self.dist_rank = [0]
        self.balanced = balanced
        super(ClassMiningSampler, self).__init__(dataset)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            if len(self.centers) == 0:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            else:
                # label set start from 1 while dist_rank start from 0
                c = np.random.choice(self.labels_set, 1)[0]
                classes = self.dist_rank[c-1][:self.n_classes]+1
            indices = []
            if self.balanced:
                for c in classes:
                    c_indices = self.label_indices[c]
                    pos = self.used_ind_pos[c]
                    indices.extend(c_indices[pos:pos + self.n_samples])
                    self.used_ind_pos[c] += self.n_samples
                    if self.used_ind_pos[c] > len(self.label_indices[c]):
                        np.random.shuffle(self.label_indices[c])
                        self.used_ind_pos[c] = 0
                yield indices
                self.count += self.batch_size
            else:
                indices = np.random.choice(np.where(np.isin(self.labels, classes))[0], self.batch_size, replace=False)
                yield indices
                self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def update_centers(self, centers, epoch):
        print('update centers in epoch %d' % epoch)
        self.centers = centers.cpu().numpy()
        self.dist = np.sum((np.expand_dims(self.centers, 1)-np.expand_dims(self.centers, 0))**2, axis=2)
        self.dist_rank = np.argsort(self.dist)


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        batch_idxs_dict, avai_labels = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch
