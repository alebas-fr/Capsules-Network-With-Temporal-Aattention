from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

# Import Datasets
from datasets.Briareo import Briareo
from datasets.SHREC import SHREC

from models.model_utilizer import ModuleUtilizer

# Import Model
from models.temporal import GestureTransoformer
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from torchsummary import summary
from models.fusion import Fusion

# Import loss

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter
from tensorboardX import SummaryWriter

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class GestureTrainer(object):
    """Gesture Recognition Train class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory

        # Losses
        self.losses = {
            'train': AverageMeter(),                      #: Train loss avg meter
            'val': AverageMeter(),                        #: Val loss avg meter
            'test': AverageMeter()                        #: Test loss avg meter
        }

        # Train val and test accuracy
        self.accuracy = {
            'train': AverageMeter(),                      #: Train accuracy avg meter
            'val': AverageMeter(),                        #: Val accuracy avg meter
            'test': AverageMeter()                        #: Test accuracy avg meter
        }

        # DataLoaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None
        self.lr = None

        # Training procedure
        self.optimizer = None
        self.iters = None
        self.epoch = 0
        self.train_transforms = None
        self.val_transforms = None
        self.loss = None

        # Tensorboard and Metrics
        self.tbx_summary = SummaryWriter(str(Path(configer.get('checkpoints', 'tb_path'))  #: Summary Writer plot
                                             / configer.get("dataset")                     #: data with TensorboardX
                                             / configer.get('checkpoints', 'save_name')))
        self.tbx_summary.add_text('parameters', str(self.configer).replace("\n", "\n\n"))
        self.save_iters = self.configer.get('checkpoints', 'save_iters')    #: int: Saving ratio

        # Other useful data
        self.in_planes = 0                                     #: int: Input channels
        self.clip_length = self.configer.get("data", "n_frames")    #: int: Number of frames per sequence
        self.n_classes = self.configer.get("data", "n_classes")     #: int: Total number of classes for dataset
        self.data_type = self.configer.get("data", "type")          #: str: Type of data (rgb, depth, ir, leapmotion)
        self.dataset = self.configer.get("dataset").lower()         #: str: Type of dataset
        self.optical_flow = self.configer.get("data", "optical_flow")
        if self.optical_flow is None:
            self.optical_flow = True
        self.scheduler = None

        self.imsize = (self.configer.get("data","imsize"),self.configer.get("data","imsize"))
        self.transforms_size = (self.configer.get("data","imsize_transform"),self.configer.get("data","imsize_transform"))

    def init_model(self):
        """Initialize model and other data for procedure"""
        if isinstance(self.data_type,list):
            i = 0
            while i<len(self.data_type):
                data_type_unique = self.data_type[i]
                if data_type_unique in ["depth", "ir"]:
                    self.in_planes += 1
                else:
                    self.in_planes += 3
                i+=1
        else:  
            if self.optical_flow is True:
                self.in_planes = 2
            elif self.data_type in ["depth", "ir"]:
                self.in_planes = 1
            else:
                self.in_planes = 3
        

        self.loss = nn.CrossEntropyLoss().to(self.device)

        # Selecting correct model and normalization variable based on type variable
        self.net = Fusion(self.configer.get("device"),self.configer.get("network",'backbone'),data_types=self.data_type,model_utility=self.model_utility,
                                       n_classes=self.n_classes,
                                       n_head=self.configer.get("network", "n_head"),
                                       n_caps=self.configer.get("network","n_caps"),
                                       input_dim=self.configer.get("network","input_dim"),
                                       pretrained=self.configer.get("network","pretrained"),
                                       layers_to_unfreeze=self.configer.get("network","layers_to_unfreeze"),
                                       layers_to_delete=self.configer.get("network","layers_to_delete"),
                                       caps_dims=self.configer.get("network","prim_caps_dim"),
                                       output_dims=self.configer.get("network","dense_caps_dim"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module"))
        
        print("Device : ",self.device)

        # Initializing training
        self.iters = 0
        self.epoch = None
        phase = self.configer.get('phase')

        # Starting or resuming procedure
        if phase == 'train':
            self.net, self.iters, self.epoch, optim_dict = self.model_utility.load_net(self.net)
        else:
            raise ValueError('Phase: {} is not valid.'.format(phase))

        print(summary(self.net,(self.in_planes*self.clip_length,self.transforms_size[0],self.transforms_size[1])))

        if self.epoch is None:
            self.epoch = 0

        # ToDo Restore optimizer and scheduler from checkpoint
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net, self.iters)
        self.scheduler = MultiStepLR(self.optimizer, self.configer["solver", "decay_steps"],gamma=self.configer.get("solver","gamma"),verbose=True)

        #  Resuming training, restoring optimizer value
        if optim_dict is not None:
            print("Resuming training from epoch {}.".format(self.epoch))
            self.optimizer.load_state_dict(optim_dict)

        # Selecting Dataset and DataLoader
        if self.dataset == "briareo":
            Dataset = Briareo
            self.train_transforms = iaa.Sequential([
                iaa.Resize((0.85, 1.15)),
                iaa.CropToFixedSize(width=self.transforms_size[0], height=self.transforms_size[1]),
                iaa.Rotate((-15, 15))
            ])
            self.val_transforms = iaa.CenterCropToFixedSize(self.transforms_size[0],self.transforms_size[1])

        elif self.dataset == "shrec2017":
            raise NotImplementedError("SHREC17 is not supported for features fusion")
        else:
            raise NotImplementedError(f"Dataset not supported: {self.configer.get('dataset')}")

        # Setting Dataloaders
        self.train_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="train", data_type=self.data_type,number_of_labels=self.n_classes,
                    transforms=self.train_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow,imsize=self.imsize),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=True, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(
            Dataset(self.configer, self.data_path, split="val", data_type=self.data_type,number_of_labels=self.n_classes,
                    transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow,imsize=self.imsize),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        if self.dataset == "shrec2017":
            self.test_loader = None
        else:
            self.test_loader = DataLoader(
                Dataset(self.configer, self.data_path, split="test", data_type=self.data_type,number_of_labels=self.n_classes,
                        transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow,imsize=self.imsize),
                batch_size=1, shuffle=False, drop_last=True,
                num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __train(self):
        """Train function for every epoch."""

        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):
            """
            input, gt
            """
            inputs = data_tuple[0].to(self.device)
            gt = data_tuple[1].to(self.device)

            output = self.net(inputs)

            self.optimizer.zero_grad()
            #loss = self.loss(output, gt.squeeze(dim=1))
            loss = self.loss(output, gt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            output = F.softmax(output,-1)
            predicted = torch.argmax(output.detach(), dim=1)
            correct = gt.detach()

            self.iters += 1
            self.update_metrics("train", loss.item(), inputs.size(0),
                                float((predicted==correct).sum()) / len(correct))


    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            # for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=str(self.accuracy["val"].avg))):
            for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=""+str(np.random.randint(200)))):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)
                gt = data_tuple[1].to(self.device)
                output = self.net(inputs)
                loss = self.loss(output, gt)

                output = F.softmax(output,-1)
                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach()

                self.iters += 1
                self.update_metrics("val", loss.item(), inputs.size(0),
                                    float((predicted == correct).sum()) / len(correct))

        self.tbx_summary.add_scalar('val_loss', self.losses["val"].avg, self.epoch + 1)
        self.tbx_summary.add_scalar('val_accuracy', self.accuracy["val"].avg, self.epoch + 1)
        accuracy = self.accuracy["val"].avg
        print("\nVal Accuracy: ",accuracy)
        self.accuracy["val"].reset()
        self.losses["val"].reset()

        ret = self.model_utility.save(accuracy, self.net, self.optimizer, self.iters, self.epoch + 1)
        if ret < 0:
            return -1
        elif ret > 0 and self.test_loader is not None:
            self.__test()
        return ret

    def __test(self):
        """Testing function."""
        self.net.eval()

        with torch.no_grad():
            for i, data_tuple in enumerate(tqdm(self.test_loader, desc="Test", postfix=str(self.accuracy["test"].avg))):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)
                gt = data_tuple[1].to(self.device)

                output = self.net(inputs)
                loss = self.loss(output, gt)
                
                output = F.softmax(output,-1)
                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach()

                self.iters += 1
                self.update_metrics("test", loss.item(), inputs.size(0),
                                    float((predicted == correct).sum()) / len(correct))
        self.tbx_summary.add_scalar('test_loss', self.losses["test"].avg, self.epoch + 1)
        self.tbx_summary.add_scalar('test_accuracy', self.accuracy["test"].avg, self.epoch + 1)
        print("Test Accuracy: ",self.accuracy["test"].avg)
        self.losses["test"].reset()
        self.accuracy["test"].reset()

    def train(self):
        print("Number of epochs: ",self.configer.get("epochs"))
        for n in range(self.configer.get("epochs")):
            print("Starting epoch {}".format(self.epoch + 1))
            self.__train()
            ret = self.__val()
            if ret < 0:
                print("Got no improvement for {} epochs, current epoch is {}."
                      .format(self.configer.get("checkpoints", "early_stop"), n))
                break
            self.epoch += 1

    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()
