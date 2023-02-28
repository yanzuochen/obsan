#! /usr/bin/env python3

from typing import Tuple
import argparse
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from tqdm.contrib import tzip

class DLAAlarmModel(nn.Module):
    def __init__(self, fc_out_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(fc_out_dim, 112),
            nn.ReLU(),
            nn.Linear(112, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 77),
            nn.ReLU(),
            nn.Linear(77, 2),
        )

    def forward(self, x):
        return self.model(x)

class DLAProtectedModule(nn.Module):
    """This implementation assumes that the protected model has and only has
    one FC layer, located at the end of the model."""

    def __init__(self, model, alarm_models) -> None:
        super().__init__()
        self.model = model
        self.alarm_models = nn.ModuleList(alarm_models)

    @staticmethod
    def from_state_dicts(model, alarm_model_state_dicts):
        fc = getattr(model, 'fc', getattr(model, 'classifier', None))
        alarm_models = [DLAAlarmModel(fc.out_features) for _ in alarm_model_state_dicts]
        for alarm_model, alarm_model_state_dict in zip(alarm_models, alarm_model_state_dicts):
            alarm_model.load_state_dict(alarm_model_state_dict)
        return DLAProtectedModule(model, alarm_models)

    def get_sus_scores(self, fc_out):
        votes = torch.cat(
            [
                torch.argmax(alarm_model(fc_out), dim=1, keepdim=True)
                for alarm_model in self.alarm_models
            ],
            dim=1,
        )
        sus_scores = torch.mean(votes.float(), dim=1)
        return sus_scores

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        sus_scores = self.get_sus_scores(logits)  # As there's only 1 FC layer
        return logits, sus_scores

    def train_forward(self, x):
        logits = self.model(x)
        raw_sus_scores = torch.cat(
            [alarm_model(logits) for alarm_model in self.alarm_models], dim=1
        )
        return raw_sus_scores

def train_alarm_model(base_model, benign_loader, ae_loader, epochs, device, logctx):
    assert len(benign_loader) == len(ae_loader)

    base_model.eval()
    fc = getattr(model, 'fc', getattr(model, 'classifier', None))
    alarm_model = DLAAlarmModel(fc.out_features).to(device)
    dla_model = DLAProtectedModule(base_model, [alarm_model])

    optimizer = torch.optim.Adam(alarm_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(alarm_model, log_freq=100)
    logctx.dict = {'per-alg-step': 0}
    alg = logctx.alg

    for i in tqdm(range(epochs)):
        for (x_benign, _), (x_ae, _) in tzip(benign_loader, ae_loader):
            optimizer.zero_grad()
            raw_sus_scores = dla_model.train_forward(torch.cat([x_benign, x_ae]).to(device))
            y = torch.cat([
                torch.zeros(x_benign.shape[0], dtype=torch.long),
                torch.ones(x_ae.shape[0], dtype=torch.long)
            ]).to(device)
            preds = torch.argmax(raw_sus_scores, dim=1)
            acc = torch.mean((preds == y).float())
            loss = criterion(raw_sus_scores, y)
            loss.backward()
            optimizer.step()
            logctx.dict.update({
                f'{alg}/loss': loss.item(),
                f'{alg}/acc': acc.item(),
            })
            wandb.log(logctx.dict)
            logctx.dict['per-alg-step'] += 1

    return alarm_model.state_dict()

if __name__ == '__main__':
    import sys, os
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    import modman
    import dataman

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, choices=['resnet50', 'googlenet', 'densenet121'])
    parser.add_argument("outfile", type=str, help="Path to the build output file.")
    parser.add_argument("--algorithms", type=str, nargs='+', default=['FGSM', 'BIM', 'CW', 'DeepFool', 'PGD'])
    parser.add_argument("--dataset", default='CIFAR10', choices=['CIFAR10'])
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size.")
    parser.add_argument("--dataset-size", type=int, default=50000, help="Max size for each dataset.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", '-d', type=str, default='cuda')
    args = parser.parse_args()

    wandb.init(project='obsan-dla')
    wandb.config.update(args)

    # load model
    model = None
    if args.model_name in ['resnet50', 'googlenet', 'densenet121']:
        # These are all models with only one FC layer, at the end of the graph
        model = modman.get_torch_mod(args.model_name, args.dataset)
    else:
        raise NotImplementedError
    model = model.to(args.device)

    # load data
    half_batch_size = args.batch_size // 2
    benign_loader, ae_loaders = None, []
    if args.dataset == "CIFAR10":
        benign_loader = dataman.get_benign_loader(
            args.dataset, 32, 'train', half_batch_size,
            size_limit=args.dataset_size, shuffle=True
        )
        for algorithm in args.algorithms:
            ae_loaders.append(
                dataman.get_ae_loader(
                    args.model_name, args.dataset, half_batch_size, alg=algorithm,
                    size_limit=args.dataset_size, shuffle=True
                )
            )
    else:
        raise NotImplementedError

    # train alarm models
    alarm_model_state_dicts = []
    for i, ae_loader in enumerate(ae_loaders):
        print(f"Training alarm model {i+1}/{len(ae_loaders)}...")
        alarm_model_state_dicts.append(
            train_alarm_model(
                model, benign_loader, ae_loader, args.epochs, args.device,
                argparse.Namespace(alg=args.algorithms[i])
            )
        )

    # save
    torch.save(alarm_model_state_dicts, args.outfile)

    artifact = wandb.Artifact(f'alarm-model-state-dicts-{args.model_name}', type='model')
    artifact.add_file(args.outfile)
    wandb.log_artifact(artifact)
