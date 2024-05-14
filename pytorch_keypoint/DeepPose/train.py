import os
import sys

import torch
import torch.amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transforms
from tqdm import tqdm

from model import create_deep_pose_model
from datasets import WFLWDataset
from losses import WingLoss
from metrics import NMEMetric


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch DeepPose Training", add_help=add_help)
    parser.add_argument("--dataset_dir", type=str, default="/home/wz/datasets/WFLW", help="WFLW dataset directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="training device, e.g. cpu, cuda:0")
    parser.add_argument("--save_weights_dir", type=str, default="./weights", help="save dir for model weights")
    parser.add_argument("--save_freq", type=int, default=5, help="save frequency for weights and generated imgs")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers, default: 8")
    parser.add_argument("--num_keypoints", type=int, default=98, help="number of keypoints")
    parser.add_argument("--lr", type=float, default=5e-4, help="SGD: learning rate")
    parser.add_argument('--lr_steps', default=[20, 25], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument("--warmup_epoch", type=int, default=2, help="number of warmup epoch for training")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')

    return parser


def main(args):
    torch.manual_seed(1234)
    dataset_dir = args.dataset_dir
    save_weights_dir = args.save_weights_dir
    save_freq = args.save_freq
    num_keypoints = args.num_keypoints
    num_workers = args.num_workers
    epochs = args.epochs
    bs = args.batch_size
    start_epoch = 0
    os.makedirs(save_weights_dir, exist_ok=True)

    if "cuda" in args.device and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"using device: {device} for training.")

    # tensorboard writer
    tb_writer = SummaryWriter()

    # create model
    model = create_deep_pose_model(num_keypoints)
    model.to(device)

    # config dataset and dataloader
    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=(256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, rotation=None, fixed_size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_dataset = WFLWDataset(root=dataset_dir,
                                train=True,
                                transforms=data_transform["train"])
    val_dataset = WFLWDataset(root=dataset_dir,
                              train=False,
                              transforms=data_transform["val"])

    train_loader = DataLoader(train_dataset,
                              batch_size=bs,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=bs,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers)

    # define loss function
    loss_func = WingLoss()

    # define optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # define learning rate scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=len(train_loader) * args.warmup_epoch
    )
    multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[len(train_loader) * i for i in args.lr_steps],
        gamma=0.1
    )

    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, multi_step_scheduler])

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(start_epoch))

    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (imgs, targets) in enumerate(train_bar):
            imgs = imgs.to(device)
            labels = targets["keypoint"].to(device)

            optimizer.zero_grad()
            # use mixed precision to speed up training
            with torch.autocast(device_type=device.type):
                pred = model(imgs)
                loss = loss_func(pred.reshape((-1, num_keypoints, 2)), labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            global_step = epoch * len(train_loader) + step
            tb_writer.add_scalar("train loss", loss.item(), global_step=global_step)
            tb_writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], global_step=global_step)

        # eval
        model.eval()
        with torch.inference_mode():
            metric = NMEMetric(h=224, w=224)
            eval_bar = tqdm(val_loader, file=sys.stdout, desc="evaluation")
            for step, (imgs, targets) in enumerate(eval_bar):
                imgs = imgs.to(device)
                labels = targets["keypoint"].to(device)

                pred = model(imgs)
                metric.update(pred.reshape((-1, num_keypoints, 2)), labels)

            nme = metric.evaluate()
            tb_writer.add_scalar("evaluation nme", nme, global_step=epoch)
            print(f"evaluation NME: {nme:.3f}")

        if epoch % save_freq == 0 or epoch == args.epochs - 1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(save_files, os.path.join(save_weights_dir, f"model_weights_{epoch}.pth"))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
