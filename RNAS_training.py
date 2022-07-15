import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetworkCIFAR as Network
from autoattack.autoattack import AutoAttack
from thop import profile, clever_format
import torchattacks

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='RNAS_H', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='RNAS_H', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


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

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    AA_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=10000, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_pgd_acc = 0.0
    best_clean_acc = 0.0
    clean = []
    pgd = []
    for epoch in range(args.epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # eval_model_size(model)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        scheduler.step(epoch)

        clean_acc, clean_obj = clean_valid(valid_queue, model, criterion)
        clean.append(clean_acc)
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
        logging.info('clean_acc %f, best_clean_acc %f', clean_acc, best_clean_acc)

        pgd_acc, pgd_obj = pgd_valid(valid_queue, model, criterion)
        pgd.append(pgd_acc)
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            utils.save(model, os.path.join(args.save, 'best_pgd_RNAS.pt'))

        logging.info('pgd_acc %f, best_pgd_acc %f', pgd_acc, best_pgd_acc)

        utils.save(model, os.path.join(args.save, 'RNAS-{}.pt'.format(epoch)))

        if (epoch) % 30 == 0:
            eval_fgsm(model, valid_queue)
            eval_pgd(model, valid_queue)
            eval_CW(model, valid_queue)
            eval_auto_attack(model, AA_queue)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()

        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
        adversarial_input = atk(input, target)

        optimizer.zero_grad()
        logits = model(adversarial_input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def pgd_valid(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20)
        adversarial_input = atk(input, target)

        logits = model(adversarial_input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('PGD %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def clean_valid(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
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
                logging.info('clean %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def eval_fgsm(model, test_loader):
    fgsm_loss = 0
    fgsm_acc = 0
    n = 0
    model.eval()
    atk = torchattacks.FGSM(model, eps=8 / 255)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        adversarial_images = atk(X, y)

        with torch.no_grad():
            output = model(adversarial_images)
            loss = F.cross_entropy(output, y)
            fgsm_loss += loss.item() * y.size(0)
            fgsm_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print('fgsm_Loss: %.3f | fgsm_Acc: %.3f%% ' % (fgsm_loss / n, 100 * fgsm_acc / n))
    return fgsm_loss / n, fgsm_acc / n


def eval_pgd(model, test_loader):
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        adversarial_images = atk(X, y)

        with torch.no_grad():
            output = model(adversarial_images)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print('pgd_Loss: %.3f | pgd_Acc: %.3f%% ' % (pgd_loss / n, 100 * pgd_acc / n))
    return pgd_loss / n, pgd_acc / n


def eval_CW(model, test_loader):
    CW_loss = 0
    CW_acc = 0
    n = 0
    model.eval()

    atk = torchattacks.CW(model)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        adversarial_images = atk(X, y)

        with torch.no_grad():
            output = model(adversarial_images)
            loss = F.cross_entropy(output, y)
            CW_loss += loss.item() * y.size(0)
            CW_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print('CW_Loss: %.3f | CW_Acc: %.3f%% ' % (CW_loss / n, 100 * CW_acc / n))
    return CW_loss / n, CW_acc / n


def eval_auto_attack(model, test_loader):
    """
evaluate model by AutoAttack
"""
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8. / 255., version='standard')
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        x_adv = adversary.run_standard_evaluation(data, target, bs=25)


def eval_model_size(model):
    model.eval()
    input = torch.randn(1, 3, 32, 32)
    input = input.cuda()
    flops, params = profile(model, (input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops: ", flops)
    print("params: ", params)


if __name__ == '__main__':
    main()
