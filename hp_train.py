import torch
import torch.nn as nn
import os
import argparse
import torch.nn.functional as F

from avalanche.training import Naive

from clip.hp_clip import load_clip_to_cpu, CustomCLIP

from datasets.caltech101_ava import MyDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HP_CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        print("HP_CLIP init")

    def get_model(self, args, classnames):
        print(f"Loading CLIP (backbone: {args.backbone})")
        # 1. 加载原生CLIP模型，如vit-B/16
        clip_model = load_clip_to_cpu(args).cuda()

        print("Building custom CLIP")
        # 2. 进行CLIP改造，
        self.model = CustomCLIP(args, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        return self.model

    def get_optimizer(self, args, param_groups):
        optim = args.NAME
        lr = args.LR
        weight_decay = args.WEIGHT_DECAY
        momentum = args.MOMENTUM
        sgd_dampening = args.SGD_DAMPNING
        sgd_nesterov = args.SGD_NESTEROV
        rmsprop_alpha = args.RMSPROP_ALPHA
        adam_beta1 = args.ADAM_BETA1
        adam_beta2 = args.ADAM_BETA2
        staged_lr = args.STAGED_LR
        new_layers = args.NEW_LAYERS
        base_lr_mult = args.BASE_LR_MULT

        if optim == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )

        elif optim == "amsgrad":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
                amsgrad=True,
            )

        elif optim == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=sgd_dampening,
                nesterov=sgd_nesterov,
            )

        elif optim == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                alpha=rmsprop_alpha,
            )

        elif optim == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

        return optimizer

    AVAI_SCHEDS = []
    def build_lr_scheduler(self, optimizer, args):
        global AVAI_SCHEDS
        """A function wrapper for building a learning rate scheduler.

        Args:
            optimizer (Optimizer): an Optimizer.
            args (argsNode): optimization config.
        """
        lr_scheduler = args.LR_SCHEDULER
        stepsize = args.STEPSIZE
        gamma = args.GAMMA
        max_epoch = args.MAX_EPOCH

        if lr_scheduler not in AVAI_SCHEDS:
            raise ValueError(
                f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
            )

        if lr_scheduler == "single_step":
            if isinstance(stepsize, (list, tuple)):
                stepsize = stepsize[-1]

            if not isinstance(stepsize, int):
                raise TypeError(
                    "For single_step lr_scheduler, stepsize must "
                    f"be an integer, but got {type(stepsize)}"
                )

            if stepsize <= 0:
                stepsize = max_epoch

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=stepsize, gamma=gamma
            )

        elif lr_scheduler == "multi_step":
            if not isinstance(stepsize, (list, tuple)):
                raise TypeError(
                    "For multi_step lr_scheduler, stepsize must "
                    f"be a list, but got {type(stepsize)}"
                )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=stepsize, gamma=gamma
            )

        elif lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                # optimizer, float(max_epoch)
                optimizer, max_epoch
            )

        return scheduler

def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    # 1. 加载数据集
    caltech=MyDataset(args)
    train_stream,test_stream=caltech.getStreams()

    hp=HP_CLIP()
    # 2. 加载训练模型
    model = hp.get_model(args,caltech.get_class_names())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # 使用的gpu id从执行命令中指定不在代码中显示指定
    model.to(device)
    optimizer = hp.get_optimizer(args,[])#todo []
    scheduler = hp.build_lr_scheduler(optimizer,args)
    # criterion是计算损失函数的，loss
    criterion = nn.CrossEntropyLoss()

    #算法指代的是strategy，strategy下面是plugin
    strategy_our = Naive(model=model,optimizer=optimizer, criterion=criterion,train_epochs=24,
                  train_mb_size=64,
                  plugins=[Hierachical_Prompt_Plugin(),
                           ],
                  device=device)

    result = []
    strategy = strategy_our
    for exp_id, experience in enumerate(train_stream):  # experience流
        print("taskId={},classes={}".format(experience.task_label, experience.classes_in_this_experience))
        strategy.train(experience, eval_streams=[])
        result.append(strategy.eval(test_stream[0:exp_id + 1]))
    torch.save(strategy.model.state_dict(), "trained_model.pth")


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits, logits_i, logits_t = self.model(image)
        loss = F.cross_entropy(logits, label)
        loss_i = F.cross_entropy(logits_i, label)
        loss_t = F.cross_entropy(logits_t, label)
        loss = loss + loss_i + loss_t

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def compute_accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for
        the specified values of k.

        Args:
            output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
            topk (tuple, optional): accuracy at top-k will be computed. For example,
                topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

        Returns:
            list: accuracy at top-k.
        """
        maxk = max(topk)
        batch_size = target.size(0)

        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc)

        return res
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="D:/dataset", help="parent path of dataset")
    parser.add_argument("--dataset", type=str, default="Caltech101", help="name of dataset")
    parser.add_argument("--gpt_dir", type=str, default="D:/project/2024-AAAI-HPT-main/data/gpt_data", help="path of gpt's data,including classes,descriptions,sturctures")
    parser.add_argument("--output-dir", type=str, default="", help="log output directory")
    parser.add_argument("--pool_size", type=int, default=20, help="log output directory")
    parser.add_argument("--description_num", type=int, default=5, help="log output directory")
    parser.add_argument("--t_prompt_length", type=int, default="", help="log output directory")
    parser.add_argument("--v_prompt_length", type=int, default="", help="log output directory")
    parser.add_argument("--output-dir", type=str, default="", help="log output directory")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument(
        "--resume-model-path",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="D:/project/2024-AAAI-HPT-main/configs/trainers/HPT/b2n.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="D:/project/2024-AAAI-HPT-main/configs/datasets/b2n/caltech101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="HPT", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default=True,action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="D:/project/2024-AAAI-HPT-main/results/output/B2N/train_base/caltech101/HPT/b2n_shots_16/seed1",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", default=10,type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    print("the run parameters:")
    main(args)