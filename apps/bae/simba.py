import torch
import torch.nn.functional as F
from tqdm import tqdm

class SimBA:

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def get_probs(self, x, y):
        output = self.forward_fn(x)
        if output is None:
            return None
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)

    def get_preds(self, x):
        output = self.forward_fn(x)
        if output is None:
            return None
        _, preds = output.data.max(1)
        return preds

    def get_preds_and_probs(self, x, y):
        output = self.forward_fn(x)
        if output is None:
            return None, None
        _, preds = output.data.max(1)
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return preds, probs

    def simba_single(self, x, y, num_iters=10000, epsilon=0.2):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)

        n_defenses = 0
        n_queries = 0
        orig_pred, last_prob = self.get_preds_and_probs(x, y)

        if last_prob is None:
            print(f'FALSE POSITIVE=1')
            return x.squeeze()

        if orig_pred != y:
            print(f'WRONG FROM BEGINNING=1')

        for i in tqdm(range(min(len(perm), num_iters))):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon

            pred, left_prob = self.get_preds_and_probs((x - diff.view(x.size())).clamp(0, 1), y)
            n_queries += 1
            if left_prob is None:
                n_defenses += 1
                continue

            if left_prob < last_prob:
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                pred, right_prob = self.get_preds_and_probs((x + diff.view(x.size())).clamp(0, 1), y)
                n_queries += 1
                if right_prob is None:
                    n_defenses += 1
                    continue
                if right_prob < last_prob:
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob

            if pred != orig_pred or n_queries >= num_iters:
                break

            # if i % 10 == 0:
                # print(last_prob)

        success = pred != orig_pred
        print(f'{n_queries} - success rate={int(success)}/1 (...) - avg # queries=... - med # queries=... - loss=... - defenses={n_defenses}')
        return x.squeeze()
