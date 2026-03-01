import argparse
from pathlib import Path

import torch
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from dataloaders import FolderLoader

from models import create_model
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device('cuda')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation')

    parser.add_argument('--input', type=str,
                        help='Input Data Root')
    parser.add_argument('--output', type=str,
                        help='Output Checkpoint Root')

    parser.add_argument('--model', type=str,
                        help='Model Name')

    parser.add_argument('--batch_size', type=int,
                        help='Batch Size for training')
    parser.add_argument('--epochs', type=int,
                        help='Epochs for training')
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Min Epochs')
    parser.add_argument('--valid_interval', type=int, default=1,
                        help='Valid')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='Test')

    parser.add_argument('--opt', type=str,
                        help='Optimizer name')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    parser.add_argument('--sched', type=str,
                        help='Scheduler name')

    return parser.parse_args()


def main(args):
    print(args)

    batch_size = args.batch_size
    epochs = args.epochs

    training_dir = Path(args.input, 'training')
    training_dataloader = FolderLoader(training_dir, batch_size=batch_size, shuffle=True, num_workers=16)

    validation_dir = Path(args.input, 'validation')
    validation_dataloader = FolderLoader(validation_dir, batch_size=batch_size, shuffle=False, num_workers=16)

    checkpoint_dir = Path(args.output, args.model)

    model = create_model(args.model, spatial_dims=2, in_channels=3, num_classes=2, img_size=256)
    optimizer = create_optimizer(args, model)
    scheduler, _ = create_scheduler(args, optimizer)

    fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=training_dataloader,
        valid_loader=validation_dataloader,
        epochs=epochs,
        result_dir=checkpoint_dir,
    )


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.
    for step in loader:
        x, y = step[0].to(device), step[1].to(device).squeeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.
    for step in loader:
        x, y = step[0].to(device), step[1].to(device).squeeze(1)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def fit(model, optimizer, scheduler, train_loader, valid_loader, epochs, result_dir) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    model = model.cuda()

    e0 = 0
    checkpoint_path = result_dir/'train.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        e0 = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(e0, epochs):
        ep_idx = epoch + 1
        print('epoch=>{}:'.format(ep_idx))
        train_loss = train(model, train_loader, optimizer)
        print('\ttrain=>loss:{}'.format(train_loss))

        valid_loss = validate(model, valid_loader)
        print('\tvalid=>loss:{}'.format(valid_loss))

        scheduler.step(epoch)

        data = {
            'epoch': ep_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(data, checkpoint_path)
        if ep_idx % 50 == 0:
            torch.save(data, result_dir / f'epoch_{ep_idx:03d}.pth')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
