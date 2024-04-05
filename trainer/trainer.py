from typing import List, Dict
import os
import glob
import time
import json
import jsbeautifier
import torch
import wandb

import experiments
import utils

from .utils import build_from_config, find_best_checkpoint

try:
    # torch 2.x
    from torch.optim.lr_scheduler import LRScheduler
except:
    # torch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Trainer:

    def __init__(self, config: dict):
        self.config = config

    # ====================================================================================================
    # ====================================================================================================

    def _init_work_dir_(self):
        # input checks
        assert 'work_dir' in self.config.keys()
        assert type(self.config['work_dir']) == str, f"{type(self.config['work_dir'])=}"
        # set work dir and session index
        work_dir = self.config['work_dir']
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir
        session_idx: int = len(glob.glob(os.path.join(self.work_dir, "train_val*.log")))
        # git log
        git_log = os.path.join(self.work_dir, f"git_{session_idx}.log")
        utils.logging.echo_page_break(filepath=git_log, heading="git branch -a")
        os.system(f"git branch -a >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git status")
        os.system(f"git status >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git log")
        os.system(f"git log >> {git_log}")
        # training log
        self.logger = utils.logging.Logger(
            filepath=os.path.join(self.work_dir, f"train_val_{session_idx}.log"),
        )

    def _init_determinism_(self):
        self.logger.info("Initializing determinism...")
        utils.determinism.set_determinism()
        # get seed for initialization steps
        assert 'init_seed' in self.config.keys()
        init_seed = self.config['init_seed']
        assert type(init_seed) == int, f"{type(init_seed)=}"
        utils.determinism.set_seed(seed=init_seed)
        # get seeds for training
        assert 'train_seeds' in self.config.keys()
        train_seeds = self.config['train_seeds']
        assert type(train_seeds) == list, f"{type(train_seeds)=}"
        for seed in train_seeds:
            assert type(seed) == int, f"{type(seed)=}"
        self.train_seeds = train_seeds

    def _init_dataloaders_(self):
        self.logger.info("Initializing dataloaders...")
        # initialize training dataloader
        assert 'train_dataset' in self.config and 'train_dataloader' in self.config
        train_dataset: torch.utils.data.Dataset = build_from_config(self.config['train_dataset'])
        self.train_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=train_dataset, shuffle=True, config=self.config['train_dataloader'],
        )
        # initialize validation dataloader
        assert 'val_dataset' in self.config and 'val_dataloader' in self.config
        val_dataset: torch.utils.data.Dataset = build_from_config(self.config['val_dataset'])
        self.val_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=val_dataset, shuffle=False, batch_size=1, config=self.config['val_dataloader'],
        )
        # initialize test dataloader
        assert 'test_dataset' in self.config and 'test_dataloader' in self.config
        test_dataset: torch.utils.data.Dataset = build_from_config(self.config['test_dataset'])
        self.test_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=test_dataset, shuffle=False, batch_size=1, config=self.config['test_dataloader'],
        )

    def _init_model_(self):
        self.logger.info("Initializing model...")
        assert 'model' in self.config
        model: torch.nn.Module = build_from_config(self.config['model'])
        model = model.cuda()
        self.model = model

    def _init_criterion_(self):
        self.logger.info("Initializing criterion...")
        assert 'criterion' in self.config
        self.criterion = build_from_config(self.config['task_criteria'])

    def _init_metric_(self):
        self.logger.info("Initializing metric...")
        assert 'metric' in self.config
        self.metric = build_from_config(self.config['task_metrics'])

    def _init_optimizer_(self):
        r"""Requires self.model.
        """
        self.logger.info("Initializing optimizer...")
        assert 'optimizer' in self.config
        assert hasattr(self, 'model') and isinstance(self.model, torch.nn.Module)
        self.optimizer: torch.optim.Optimizer = build_from_config(
            params=self.model.parameters(), config=self.config['optimizer'],
        )

    def _init_scheduler_(self):
        r"""Requires self.train_dataloader and self.optimizer.
        """
        self.logger.info("Initializing scheduler...")
        assert 'scheduler' in self.config
        assert hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        assert hasattr(self, 'optimizer') and isinstance(self.optimizer, torch.optim.Optimizer)
        lr_lambda = self.config['scheduler']['args']['lr_lambda']
        if type(lr_lambda) == dict:
            lr_lambda = build_from_config(steps=len(self.train_dataloader), config=lr_lambda)
        assert callable(lr_lambda)
        self.config['scheduler']['args']['lr_lambda'] = lr_lambda
        self.scheduler: LRScheduler = build_from_config(
            optimizer=self.optimizer, config=self.config['scheduler'],
        )

    @property
    def expected_files(self):
        return ["training_losses.pt", "validation_scores.json"]

    def _init_state_(self):
        self.logger.info("Initializing state...")
        # input checks
        assert 'epochs' in self.config.keys()
        # init epoch numbers
        self.cum_epochs = 0
        self.tot_epochs = self.config['epochs']
        assert len(self.train_seeds) == self.tot_epochs, f"{len(self.train_seeds)=}, {self.tot_epochs=}"
        # no need to resume if no checkpoint saved
        if len(glob.glob(os.path.join(self.work_dir, "**", "checkpoint.pt"))) == 0:
            return
        # determine where to resume from
        load_idx: int = -1
        for idx in range(self.tot_epochs):
            cond = all([os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", filename))
                for filename in self.expected_files
            ])
            if not cond:
                break
            if os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", "checkpoint.pt")):
                load_idx = idx
        # resume state
        checkpoint_filepath = os.path.join(self.work_dir, f"epoch_{load_idx}", "checkpoint.pt")
        try:
            checkpoint = torch.load(checkpoint_filepath)
            self.cum_epochs = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            self.logger.error(f"Got {e} when loading from {checkpoint_filepath}. Starting over training.")

    # ====================================================================================================
    # iteration-level methods
    # ====================================================================================================

    def _train_step_(self, example: Dict[str, torch.Tensor]) -> None:
        r"""
        Args:
            example (Dict[str, torch.Tensor]): a dictionary containing the image and the multi-task
                ground-truth label for the current input.
        """
        # init time
        start_time = time.time()
        # copy to GPU
        example = utils.apply_tensor_op(func=lambda x: x.cuda(), inputs=example)
        # do computation
        self.optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(example)
            losses: Dict[str, torch.Tensor] = self.criterion(y_pred=outputs, y_true=example)
        losses.sum().backward()
        # update logger
        self.logger.update_buffer({"learning_rate": self.scheduler.get_last_lr()})
        self.logger.update_buffer({
            'tot_loss': sum(losses.values()),
            'avg_loss': sum(losses.values()) / len(losses.values()),
        })
        self.logger.update_buffer(dict(("loss_"+task, val) for task, val in losses.items()))
        # update states
        self.optimizer.step()
        self.scheduler.step()
        # log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    def _eval_step_(self, example: Dict[str, torch.Tensor]) -> None:
        r"""
        Args:
            example (Dict[str, torch.Tensor]): a dictionary containing the image and the multi-task
                ground-truth label for the current input.
        """
        # init time
        start_time = time.time()
        # copy to GPU
        example = utils.apply_tensor_op(func=lambda x: x.cuda(), inputs=example)
        # do computation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(example)
            scores: Dict[str, torch.Tensor] = self.metric(y_pred=outputs, y_true=example)
        # update logger
        self.logger.update_buffer(scores)
        # log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    # ====================================================================================================
    # training and validation epochs
    # ====================================================================================================

    def _train_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # do training loop
        self.model.train()
        self.criterion.reset_buffer()
        for idx, example in enumerate(self.train_dataloader):
            self._train_step_(example=example)
            self.logger.flush(prefix=f"Training [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.train_dataloader)}].")
        # after training loop
        self._after_train_loop_()
        # log time
        self.logger.info(f"Training epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _val_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()
        for idx, example in enumerate(self.val_dataloader):
            self._eval_step_(example=example)
            self.logger.flush(prefix=f"Validation [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.val_dataloader)}].")
        # after validation loop
        self._after_val_loop_()
        # log time
        self.logger.info(f"Validation epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _after_train_loop_(self) -> None:
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        if not os.path.isdir(epoch_root):
            os.makedirs(epoch_root)
        # save training losses to disk
        losses: Dict[str, torch.Tensor] = self.criterion.summarize()
        torch.save(obj=losses, f=os.path.join(epoch_root, "training_losses.pt"))
        # save checkpoint to disk
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        torch.save(obj={
            'epoch': self.cum_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, f=latest_checkpoint)
        # set latest checkpoint
        soft_link: str = os.path.join(self.work_dir, "checkpoint_latest.pt")
        if os.path.isfile(soft_link):
            os.system(' '.join(["rm", soft_link]))
        os.system(' '.join(["ln", "-s", os.path.relpath(path=latest_checkpoint, start=self.work_dir), soft_link]))

    def _after_val_loop_(self) -> None:
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        if not os.path.isdir(epoch_root):
            os.makedirs(epoch_root)
        # save validation scores to disk
        scores: Dict[str, float] = self.metric.summarize()
        with open(os.path.join(epoch_root, "validation_scores.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(json.dumps(scores), jsbeautifier.default_options()))
        # set best checkpoint
        checkpoints: List[str] = glob.glob(os.path.join(self.work_dir, "epoch_*", "checkpoint.pt"))
        best_checkpoint: str = find_best_checkpoint(checkpoints=checkpoints)
        soft_link: str = os.path.join(self.work_dir, "checkpoint_best.pt")
        if os.path.isfile(soft_link):
            os.system(' '.join(["rm", soft_link]))
        os.system(' '.join(["ln", "-s", os.path.relpath(path=best_checkpoint, start=self.work_dir), soft_link]))
        # cleanup checkpoints
        checkpoints.remove(best_checkpoint)
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        if latest_checkpoint in checkpoints:
            checkpoints.remove(latest_checkpoint)
        os.system(' '.join(["rm", "-f"] + checkpoints))

    # ====================================================================================================
    # test epoch
    # ====================================================================================================

    @torch.no_grad()
    def _test_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # before test loop
        best_idx: int = self._before_test_loop_()
        # do test loop
        self.model.eval()
        self.metric.reset_buffer()
        for idx, example in enumerate(self.test_dataloader):
            self._eval_step_(example=example)
            self.logger.flush(prefix=f"Test epoch [Iteration {idx}/{len(self.test_dataloader)}].")
        # after test loop
        self._after_test_loop_(best_idx=best_idx)
        # log time
        self.logger.info(f"Test epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _before_test_loop_(self) -> int:
        r"""
        Returns:
            best_idx (int): the index of the best checkpoint.
        """
        # find best model
        best_idx = self.tot_epochs - 1
        # load model
        checkpoint = torch.load(os.path.join(self.work_dir, f"epoch_{best_idx}", "checkpoint.pt"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return best_idx

    def _after_test_loop_(self, best_idx: int) -> None:
        r"""
        Args:
            best_idx (int): the index of the best checkpoint.
        """
        # initialize test results directory
        test_root = os.path.join(self.work_dir, "test")
        if not os.path.isdir(test_root):
            os.makedirs(test_root)
        # save test results to disk
        scores = self.metric.summarize()
        scores['checkpoint_filepath'] = os.path.join(f"epoch_{best_idx}", "checkpoint.pt")
        with open(os.path.join(test_root, "test_results.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(json.dumps(scores), jsbeautifier.default_options()))

    # ====================================================================================================
    # ====================================================================================================

    def train(self):
        # skip if finished
        if experiments.utils.has_finished(
            work_dir=self.config['work_dir'], expected_epochs=self.config['epochs'], wait_time=60.0,
        ):
            return
        # initialize modules
        self._init_work_dir_()
        self._init_determinism_()
        self._init_dataloaders_()
        self._init_model_()
        self._init_criterion_()
        self._init_metric_()
        self._init_optimizer_()
        self._init_scheduler_()
        self._init_state_()
        # initialize run
        wandb.init()
        start_epoch = self.cum_epochs
        self.logger.page_break()
        # training and validation epochs
        for idx in range(start_epoch, self.tot_epochs):
            utils.determinism.set_seed(seed=self.train_seeds[idx])
            self._train_epoch_()
            self._val_epoch_()
            self.logger.page_break()
            self.cum_epochs = idx + 1
        # test epoch
        self._test_epoch_()
        # cleanup run
        wandb.finish()
