import argparse
import os
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TestData(Dataset):
    """Yield user IDs for full-catalog evaluation (Section V-A.3)."""

    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorData(Dataset):
    """Sample one BPR triplet per user and behavior (Eq. 13)."""

    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        total = []
        user = idx + 1
        key = str(user)
        history = set(self.behavior_dict['all'].get(key) or ())
        has_negative = len(history) < self.item_count or any(
            item not in history for item in range(1, self.item_count + 1)
        )

        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(key)
            if not items:
                total.append([0, 0, 0])
                continue

            pos = random.sample(items, 1)[0]
            neg = 0
            if has_negative:
                # Draw negatives outside the user's training history.
                neg = random.randint(1, self.item_count)
                while neg in history:
                    neg = random.randint(1, self.item_count)
            total.append([user, pos, neg])

        return np.array(total)

    def __len__(self):
        return self.user_count


class NegativeSampleData(Dataset):
    """Sample ten negatives per positive for the alternate loader."""

    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.item_list = [x + 1 for x in range(self.item_count)]

    def __getitem__(self, idx):
        total = []

        for behavior in self.behaviors:
            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0] * 12
            else:
                pos = random.sample(items, 1)[0]
                neg = np.random.choice(np.setdiff1d(self.item_list, [pos]), 10).tolist()
                signal = [idx + 1, pos]
                signal.extend(neg)

            total.append(signal)

        all_inter = self.behavior_dict['all'].get(str(idx + 1), None)
        if all_inter is None:
            total.append([0] * 12)
        else:
            pos = random.sample(all_inter, 1)[0]
            neg = np.random.choice(np.setdiff1d(self.item_list, [pos]), 10).tolist()
            signal = [idx + 1, pos]
            signal.extend(neg)

            total.append(signal)

        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):
    """Load behavior graphs and precomputed splits (Sections III-A, V-A.3)."""

    def __init__(self, args):
        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type

        self.__get_count()
        self.item_offset = self.user_count + 1
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()
        self.__build_offline_popularity_pool()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    def __get_test_dict(self):
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """Build bipartite graphs and global behavior degrees (Eq. 1)."""
        self.edge_index = {}
        self.degree_by_behavior = {}
        item_degrees = []
        user_degrees = []
        all_pairs = []
        user_rows = self.item_offset
        item_rows = self.item_count + 1

        for behavior in self.behaviors:
            users, items = [], []
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                for line in f:
                    user, item, *_ = line.split()
                    users.append(int(user))
                    items.append(int(item))
            user_ids = torch.tensor(users, dtype=torch.long)
            item_ids = torch.tensor(items, dtype=torch.long)

            user_degree = torch.bincount(user_ids, minlength=user_rows).float()
            item_degree = torch.bincount(item_ids, minlength=item_rows).float()
            user_degrees.append(user_degree)
            item_degrees.append(item_degree)
            self.degree_by_behavior[behavior] = torch.cat((user_degree, item_degree))

            # Store both directions in the joint node space.
            forward = torch.stack((user_ids, item_ids + user_rows))
            self.edge_index[behavior] = torch.cat((forward, forward.flip(0)), dim=1)
            all_pairs.append(user_ids * item_rows + item_ids)

        self.item_behaviour_degree = torch.stack(item_degrees, dim=1)
        self.user_behaviour_degree = torch.stack(user_degrees, dim=1)

        # Deduplicate the behavior union for local extraction (Section IV-B).
        union_pairs = torch.unique(torch.cat(all_pairs))
        union_users = torch.div(union_pairs, item_rows, rounding_mode='floor')
        union_items = torch.remainder(union_pairs, item_rows) + user_rows
        forward = torch.stack((union_users, union_items))
        self.all_edge_index = torch.cat((forward, forward.flip(0)), dim=1)

    def __build_offline_popularity_pool(self):
        """Build the semantic candidate pool from target training degrees (Eq. 4)."""
        target_behavior_idx = len(self.behaviors) - 1
        item_degrees = self.item_behaviour_degree[:, target_behavior_idx]
        sorted_indices = torch.argsort(item_degrees, descending=True)
        valid_sorted_items = sorted_indices[sorted_indices != 0]

        # Keep the top 5% of catalog items (Section V-A.4).
        top_k_num = max(1, int(valid_sorted_items.numel() * 0.05))
        self.popular_items_tensor = valid_sorted_items[:top_k_num]
        self.popular_items = self.popular_items_tensor.tolist()

    def behavior_dataset(self):
        if self.loss_type == 'bpr':
            return BehaviorData(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)
        else:
            return NegativeSampleData(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return TestData(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestData(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['view', 'cart', 'buy'], help='Sequential behavior chain')
    parser.add_argument('--data_path', type=str, default='./data/beibei', help='Path to dataset')
    parser.add_argument('--loss_type', type=str, default='bpr', help='Optimization objective')
    args = parser.parse_args()

    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5)

    for index, batch_ids in enumerate(loader):
        print("Batch IDs shape (User, Pos, Neg):", batch_ids.shape)
        break
