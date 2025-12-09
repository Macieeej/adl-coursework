#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple, Tuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
#from torch.nn import functional as F
#import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse
from pathlib import Path
import sys
import os
from torchvision.transforms import Resize

#torch.cuda.empty_cache()
#everything is in 'data' directory which is just a symlink to the work storage on bc4
current_dir = os.path.dirname(os.path.abspath(__file__))
mit_data_dir = os.path.join(current_dir, 'data')
fix_map_data_dir = os.path.join(current_dir, 'data', 'ALLFIXATIONMAPS')
checkpoint_dir = os.path.join(current_dir, 'data', 'CHECKPOINTS')
sys.path.append(mit_data_dir)
sys.path.append(fix_map_data_dir)
sys.path.append(checkpoint_dir)
all_mean_aucs = {}

from dataset import MIT, crop_to_region
from metrics import calculate_auc, roc_auc

parser = argparse.ArgumentParser(
    description="Train a Multiresolution CNN on MIT dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.01, type=float, help="Learning rate (default=0.01)")
parser.add_argument("--batch-size", default=256, type=int, help="Number of images within each mini-batch",)
parser.add_argument("-s", "--max-steps", default=5000, type=int, help="Number of steps to train for (default=5000)",)
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum factor for SGD optimizer",)
parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to tensorboard in number of steps",)
parser.add_argument("--print-frequency", default=10, type=int, help="How frequently to print progress to the command line in number of steps",)
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.",)
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout corruption probability (default=0.2)")
parser.add_argument("--weight-decay", default=0.0002, type=float, help="Weight decay factor (default=0.0002)")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


class MultiResolutionCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout
        # I've got 99 problems and python is all of them
        # problem #56: you cannot put this in a for loop since "stream" is a local variable to the for loop and doesn't set the outer stream definitions like a logical and sane programming language would
        self.stream1 = nn.Sequential(
            nn.Conv2d(3, 96, (7,7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(96, 160, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(160, 288, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(p=self.dropout)

        )
        self.initialise_layer(self.stream1)

        self.stream2 = nn.Sequential(
            nn.Conv2d(3, 96, (7,7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(96, 160, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(160, 288, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(p=self.dropout)

        )
        self.initialise_layer(self.stream2)

        self.stream3 = nn.Sequential(
            nn.Conv2d(3, 96, (7,7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(96, 160, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(160, 288, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Dropout(p=self.dropout)

        )
        self.initialise_layer(self.stream3)

        # trying to figure out how to make this bs layer work made me lose brain cells
        self.fc_stream1 = nn.Sequential(
            nn.Linear(288*3*3, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.initialise_layer(self.fc_stream1)

        self.fc_stream2 = nn.Sequential(
            nn.Linear(288*3*3, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.initialise_layer(self.fc_stream2)

        self.fc_stream3 = nn.Sequential(
            nn.Linear(288*3*3, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.initialise_layer(self.fc_stream3)

        # this feels wrong, maybe?
        self.fc2 = nn.Sequential(
            nn.Linear(512*3, 1),
            #nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.initialise_layer(self.fc2)

    def forward(self, batch):
        x1 = batch[:, 0, :, :, :]
        x2 = batch[:, 1, :, :, :]
        x3 = batch[:, 2, :, :, :]

        # I want to put this into a for loop
        x1 = self.stream1(x1)
        x2 = self.stream2(x2)
        x3 = self.stream3(x3)

        # I probably still can't put this in a for loop
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        x1 = self.fc_stream1(x1)

        x2 = self.fc_stream2(x2)
      
        x3 = self.fc_stream3(x3)
   

        x = torch.cat((x1, x2, x3), 1)
        
        x = self.fc2(x)
    
        return x

    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dataset: MIT,
        test_dataset: MIT,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        args,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.args = args
        self.val_count = 0

    def train(
        self,
        max_steps,
        print_frequency: int = 20,
        log_frequency: int = 5,
        epoch: int = 0,
    ):
        global all_mean_aucs
        self.model.train()
        while self.step <= max_steps:
            self.model.train()
            data_load_start_time = time.time()

            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                self.optimizer.zero_grad()
                loss = self.criterion(logits, labels)
                loss.backward()
                if loss > 100:
                    print("ERROR: Loss went over 100, terminate! " + str(loss))
                    print(" " * 7 + "Momentum: " + str(self.optimizer.param_groups[0]["momentum"]))
                    sys.exit(1)
                self.optimizer.step()

                # decay learning rate over time
                self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * 0.999
                # mutate momentum linearly from 0.9 to 0.99
                # Note to self, you are utterly regarded for not noticing why this broke momentum hypertuning in the other branch
                # Seriously, i hope that autism diagnosis comes soon because this is just sad
                self.optimizer.param_groups[0]["momentum"] = 0.9 + (0.99 - 0.9) * (self.step / max_steps)


                with torch.no_grad():
                    preds = (logits > 0.5).float()
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

                self.summary_writer.add_scalar("epoch", epoch, self.step)
                if (self.step % 200) == 0:
                    self.validate()
                    # self.validate() will put the model in validation mode,
                    # so we have to switch back to train mode afterwards
                    self.model.train() 
            epoch += 1
        self.validate()
        print("Mean auc scores for all validations:", all_mean_aucs)

        best_checkpoint = max(all_mean_aucs, key=all_mean_aucs.get)
        print(f"Best checkpoint: {best_checkpoint} with mean auc score of {all_mean_aucs[best_checkpoint]}")
        return best_checkpoint


    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        preds_dict = {}
        targets_dict = {}
        total_loss = 0
        self.model.eval()

        crit_for_loss = nn.BCELoss()
        global all_mean_aucs

        self.model.eval()
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            startIndex = 0
            endIndex = 2500

            for i in range(len(self.val_dataset.dataset)):
                batch = []  

                for j in range(startIndex, endIndex):
                    img, _ = self.val_dataset.__getitem__(j) 
                    batch.append(img)  

                file_name = self.val_dataset.dataset[i]['file']
                batch = torch.stack(batch, dim=0).to(self.device)

                logits = self.model(batch)
                preds = torch.sigmoid(logits)

                height, width = self.val_dataset.dataset[i]['X'].shape[1:]  # Shape is [3, H, W]

                # this is the saliency map our model generates!!!
                preds_gray_interpolated = process_predictions(preds, (height, width))

                gt_path_a = f"{file_name[:-5]}_fixMap.jpg"
                gt_path = os.path.join(fix_map_data_dir, gt_path_a)
                fixation_map = Image.open(gt_path).convert('L')
                fixation_map_np = np.array(fixation_map) 
                
                preds_dict[file_name] = preds_gray_interpolated.cpu().numpy()
                targets_dict[file_name] = fixation_map_np

             
                fixation_map_tensor = torch.tensor(fixation_map_np, dtype=torch.float32).to(self.device) / 255.0 # Shape [H, W]
                #print('Maximum value in fixation_map_tensor:', torch.max(fixation_map_tensor).item())
                #print('Minimum value in fixation_map_tensor:', torch.min(fixation_map_tensor).item())
                
          
                preds_gray_resized = preds_gray_interpolated.squeeze(0) / 255.0  # Shape [H, W]
                #print('Maximum value in preds_gray_resized:', torch.max(preds_gray_resized).item())
                #print('Minimum value in preds_gray_resized:', torch.min(preds_gray_resized).item())


                print('fixation_map_tensor.shape', fixation_map_tensor.shape)
                print('preds_gray_resized.shape', preds_gray_resized.shape)

                # Compute loss between predictions and ground truth fixation maps
                loss = crit_for_loss(preds_gray_resized, fixation_map_tensor)
                total_loss += loss.item()


                if endIndex == 250000:
                  break
                startIndex += 2500
                endIndex += 2500

            mean_auc = calculate_auc(preds_dict, targets_dict)
            print(f"{self.val_count}: Mean AUC score for validation set: {mean_auc}\n")
            self.log_validate_metrics(mean_auc, epoch=self.step//95, total_loss=total_loss)
    
            all_mean_aucs[self.val_count] = mean_auc
            self.val_count += 1

            print(f"Saving model to {os.path.abspath(os.path.join(checkpoint_dir, (get_checkpoint_log_name(self.args) + str(len(all_mean_aucs)))))}")
            torch.save({
                'args': self.args,
                'model': self.model.state_dict(),
                'accuracy': mean_auc
            }, os.path.abspath(os.path.join(checkpoint_dir, (get_checkpoint_log_name(self.args) + str(len(all_mean_aucs))))))
    
    # logging val to tensorboard hopefully
    def log_validate_metrics(self, mean_auc, epoch, total_loss):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalar("accuracy/val", mean_auc, self.step)


        average_loss = total_loss / len(self.val_dataset.dataset)

        self.summary_writer.add_scalars(
                "auc",
                {"test": mean_auc},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, mean AUC: {mean_auc}")



def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"MrCNN_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"momentum={args.momentum}_" +
        f"weight_decay={args.weight_decay}_" +
        f"dropout={args.dropout}_" +
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

#This gets an actual unique directory name for the checkpoint
def get_checkpoint_log_dir(args: argparse.Namespace) -> str:
    checkpoint_log_dir = (
        f"MrCNN_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"momentum={args.momentum}_" +
        f"dropout={args.dropout}_" +
        f"weightdecay={args.weight_decay}_" +
        f"checkpoints"
    )

    if os.path.exists(checkpoint_log_dir):
        print("Checkpoint directory already exists, may overwrite!")
    return str(checkpoint_log_dir)

#its just the name of a single checkpoint file
def get_checkpoint_log_name(args: argparse.Namespace) -> str:
    tb_log_dir = (
        f"MrCNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"momentum={args.momentum}_" +
        f"dropout={args.dropout}_" +
        f"weightdecay={args.weight_decay}_" +
        f"checkpoint_"
    )
    return str(tb_log_dir)


def process_predictions(preds, target_shape):
    """
    Process predictions to create a resized grayscale image.

    Args:
        preds (torch.Tensor): 1D tensor of shape [2500].
        target_shape (tuple): Target height and width (H, W).

    Returns:
        torch.Tensor: Resized grayscale image of shape [1, H, W].
    """

    preds = preds.view(-1)

    grid_size = int(preds.shape[0]**0.5)
    preds_gray = preds.view(grid_size, grid_size) * 255  

   
    height, width = target_shape
    preds_gray = preds_gray.unsqueeze(0).unsqueeze(0)  
    resize = Resize((height, width))
    preds_gray_resized = resize(preds_gray)

    return preds_gray_resized.squeeze(0)  



class Tester:
    def __init__(
        self,
        model: nn.Module,
        test_dataset: MIT,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.test_dataset = test_dataset
        self.summary_writer = summary_writer
        self.step = 0

    def test(self):
        preds_dict = {}
        targets_dict = {}
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            #for batch, label in self.val_dataset:
            startIndex = 0
            endIndex = 2500

            for i in range(len(self.test_dataset.dataset)):
                batch = []  # To store all 2500 images


                for j in range(startIndex, endIndex):
                    img, _ = self.test_dataset.__getitem__(j)  # Unpack the tuple
                    batch.append(img)  # Append the image to the batch list

                file_name = self.test_dataset.dataset[i]['file']
                batch = torch.stack(batch, dim=0).to(self.device)

                logits = self.model(batch)

                preds = torch.sigmoid(logits)

                height, width = self.test_dataset.dataset[i]['X'].shape[1:]  # Shape is [3, H, W]

                preds_gray_interpolated = process_predictions(preds, (height, width))

                gt_path_a = f"{file_name[:-5]}_fixMap.jpg"
                gt_path = os.path.join(fix_map_data_dir, gt_path_a)

                fixation_map = Image.open(gt_path).convert('L')
                preds_dict[file_name] = preds_gray_interpolated.cpu().numpy()

                fixation_map_np = np.array(fixation_map)  # Convert to NumPy array
                targets_dict[file_name] = fixation_map_np

                if endIndex == 250000:
                    break
                startIndex += 2500
                endIndex += 2500

            mean_auc = calculate_auc(preds_dict, targets_dict)
            print(f"Mean AUC score for test set: {mean_auc}")


def main(args):
    err = False
    if not os.path.exists(mit_data_dir):
        print("MIT data directory not found!")
        err = True
    if not os.path.exists(fix_map_data_dir):
        print("Fixation map data directory not found!")
        err = True
    if err:
        return


    torch.cuda.empty_cache()
    train_dataset = MIT(dataset_path=os.path.join(mit_data_dir, 'train_data.pth.tar'))
    test_dataset = MIT(dataset_path=os.path.join(mit_data_dir, 'test_data.pth.tar'))
    val_dataset = MIT(dataset_path=os.path.join(mit_data_dir, 'val_data.pth.tar'))

    # something doesnt work and puts it in the working directory
    # problem #77: python needs you to declare a global variable within scope to actually access it because weakly typed languages suck
    global checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Created checkpoint directory at " + checkpoint_dir)
    checkpoints_dir = os.path.join(checkpoint_dir, get_checkpoint_log_dir(args))
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print("Created checkpoints directory at " + checkpoints_dir)
    else:
        print("Checkpoints directory already exists, may overwrite!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    # why tho
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=args.worker_count,
        pin_memory=True,
    )


    # need to pass dropout
    model = MultiResolutionCNN(dropout=args.dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )


    trainer = Trainer(
        model, train_loader, val_dataset, test_dataset, criterion, optimizer, summary_writer, DEVICE, args
    )


    best_checkpoint = trainer.train(
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        max_steps=args.max_steps,
    )

    tester = Tester(
        trainer.model, test_dataset, summary_writer, DEVICE
    )

    tester.test()


    summary_writer.close()

    # Return code for the best checkpoint (abuse this in the script!)
    sys.exit(best_checkpoint)


if __name__ == "__main__":
    main(parser.parse_args())