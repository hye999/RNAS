import os
import sys
import time
import glob
import numpy as np
import utils
import logging
import argparse
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models import *
from torch.autograd import Variable
from evaluate import *
from models import *


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./cifar-data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='VGG16', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
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

if args.set == 'cifar10':
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

    model = VGG('VGG16')
    # model = ResNet18()
    # model = MobileNetV2()
    # model = DenseNet121()
    model = model.cuda()

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

    if args.set == 'cifar10':
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    AA_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=10000, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)

    best_pgd_acc = 0.0
    best_clean_acc = 0.0
    clean = []
    pgd = []
    for epoch in range(args.epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        scheduler.step()

        clean_acc, clean_obj = test(valid_queue, model, criterion)
        clean.append(clean_acc)
        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
        logging.info('clean_acc %f, best_clean_acc %f', clean_acc, best_clean_acc)

        pgd_acc, pgd_obj = adv_test(valid_queue, model, criterion)
        pgd.append(pgd_acc)
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            utils.save(model, os.path.join(args.save, 'VGG16_best_pgd.pt'))
            e = epoch
        logging.info('pgd_acc %f, best_pgd_acc{} %f'.format(e), pgd_acc, best_pgd_acc)
        utils.save(model, os.path.join(args.save, '{}--VGG16.pt'.format(epoch)))


        # if (epoch + 1) % 20 == 0:
        #     model_size(model)
        #     eval_fgsm(model, valid_queue)
        #     eval_CW(model, valid_queue)
        #     eval_auto_attack(model, AA_queue)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

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


def test(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

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


def adv_test(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
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


if __name__ == '__main__':
    main()
