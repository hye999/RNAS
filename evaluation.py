import sys
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from thop import profile, clever_format
import torchattacks
from autoattack.autoattack import AutoAttack
from models import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./cifar-data', help='location of the data corpus')
parser.add_argument('--model_path', type=str, default='./mobilenetv2.pt', help='path of pretrained model')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    model = MobileNetV2()
    model = model.cuda()
    utils.load(model, args.model_path)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()


    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_queue2 = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False, pin_memory=True, num_workers=2)

    # Calculate clean acc
    test_acc, test_obj = eval_standard(test_queue, model, criterion)
    logging.info('clean_acc %f', test_acc)

    # evaluate model by FGSM
    fgsm_acc, fgsm_obj = eval_fgsm(test_queue, model, criterion)
    logging.info('fgsm_acc %f', fgsm_acc)

    # evaluate model by PGD
    pgd_acc, pgd_obj = eval_pgd(test_queue, model, criterion)
    logging.info('pgd_acc %f', pgd_acc)

    # evaluate model by CW
    cw_acc, cw_obj = eval_cw(test_queue, model, criterion)
    logging.info('cw_acc %f', cw_acc)

    # evaluate model by AutoAttacks
    eval_auto_attack(model, test_queue2)

    # Calculate model size and FLOPs
    eval_model_size(model)


def eval_fgsm(test_queue, model, criterion):
    """
  evaluate model by FGSM
  """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    atk = torchattacks.FGSM(model, eps=8 / 255)

    for step, (input, target) in enumerate(test_queue):
        input = input.cuda()
        target = target.cuda()
        adversarial_input = atk(input, target)
        logits = model(adversarial_input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('fgsm %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def eval_pgd(test_queue, model, criterion):
    """
  evaluate model by PGD
  """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
    for step, (input, target) in enumerate(test_queue):
        input = input.cuda()
        target = target.cuda()
        adversarial_input = atk(input, target)
        logits = model(adversarial_input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('pgd %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def eval_cw(test_queue, model, criterion):
    """
  evaluate model by CW
  """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    atk = torchattacks.CW(model, c=0.5, kappa=0, steps=100, lr=0.01)
    for step, (input, target) in enumerate(test_queue):
        input = input.cuda()
        target = target.cuda()
        adversarial_input = atk(input, target)
        logits = model(adversarial_input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('cw %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def eval_standard(test_queue, model, criterion):
    """
      clean acc
      """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def eval_auto_attack(model, test_loader):
    """
  evaluate model by Auto Attack
  """
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8. / 255., version='standard')

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        x_adv = adversary.run_standard_evaluation(data, target, bs=250)


def eval_model_size(model):
    """
      Calculate model size and FLOPs
      """
    model.eval()
    input = torch.randn(1, 3, 32, 32)
    input = input.cuda()
    flops, params = profile(model, (input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops: ", flops)
    print("params: ", params)



if __name__ == '__main__':
    main()
