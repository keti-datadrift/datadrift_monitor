from tqdm import tqdm
import math

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel


class AverageMetric:

    def __init__(self,):
        self._running_scores = torch.zeros(1)
        self._count = 0.

    def update_state(self, value, counts):

        if math.isnan(value) or math.isinf(value):
            return
        
        self._count += counts
        self._running_scores += value

    def reset_state(self,):
        self._running_scores = torch.zeros_like(self._running_scores)
        self._count = 0

    def result(self, distributed=False, reduce_device=None):
        if distributed and dist.is_available() and dist.is_initialized():
            scores = self._running_scores.clone()
            count = torch.tensor([self._count], dtype=scores.dtype)
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            return scores / count

        return self._running_scores/self._count


class Accuracy(AverageMetric):

    def __init__(self, topk=(1,)):
        super().__init__()
        self._maxk = max(topk)
        self._running_scores = torch.zeros(len(topk))
        self._topk = topk

    def update_state(self, outputs, targets):
        with torch.no_grad():
            self._count += targets.size(0)
            _, pred = outputs.topk(self._maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for i, k in enumerate(self._topk):
                self._running_scores[i] += \
                    correct[:k].reshape(-1).float().sum(0).to('cpu')

def train(
        net,
        trainloader,
        testloader,
        criterion,
        optimizer,
        epochs,
        batch_size,
        n_classes,
        log_every_n=250,
        checkpoint=None,
        summary_writer=None,
        save_only_head=False,
        device=None,
        distributed=False,
        rank=0,
        train_sampler=None,
        is_main_process=True,
        ):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracy = Accuracy((1, min(5, n_classes)))

    train_loss = AverageMetric()
    test_loss = AverageMetric()
    best_acc = (0., 0.)
    start_epoch = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    global_steps = 0

    for epoch in range(start_epoch, epochs):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process:
            print('\nEpoch: %d' % epoch)
            print('\nLearning Rate: %.4f' %
                  scheduler.get_last_lr()[0])

        net.train()

        train_loss.reset_state()

        for inputs, targets in tqdm(
                trainloader,
                desc=f"Training epoch {epoch}",
                disable=not is_main_process):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            net.zero_grad()
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.update_state(loss.item(), 1)
            global_steps += 1

            if is_main_process and summary_writer is not None and global_steps % log_every_n == 0:
                summary_writer.add_scalar("train/loss", train_loss.result(), global_steps)
                summary_writer.add_scalar("train/epoch", epoch, global_steps)
                summary_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_steps)
                train_loss.reset_state()

        scheduler.step()
        accuracy.reset_state()

        net.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(
                    enumerate(testloader),
                    desc=f"Evaluating epoch {epoch}",
                    disable=not is_main_process):
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss.update_state(loss.item(), 1)
                accuracy.update_state(outputs, targets)

        val_top1, val_top5 = accuracy.result(distributed, device)
        val_loss = test_loss.result(distributed, device)

        if is_main_process:
            print(f"Top 1 acc: {val_top1}")
            print(f"Top 5 acc: {val_top5}")

        if is_main_process and summary_writer is not None:
            summary_writer.add_scalar("test/loss", val_loss, epoch)
            summary_writer.add_scalar("test/acc_top1", val_top1, epoch)
            summary_writer.add_scalar("test/acc_top5", val_top5, epoch)
        
        accuracy.reset_state()
        test_loss.reset_state()

        val_top1_float = float(val_top1.item())
        val_top5_float = float(val_top5.item())

        if is_main_process and (checkpoint is not None) and (val_top1_float > best_acc[0]):
            best_acc = (val_top1_float, val_top5_float)
            print("Saving...")
            model = net.module if isinstance(net, (torch.nn.DataParallel, DistributedDataParallel)) else net
            if save_only_head:
                torch.save(model.head.state_dict(), checkpoint)
            else:
                torch.save(model.state_dict(), checkpoint)
            
    return best_acc
