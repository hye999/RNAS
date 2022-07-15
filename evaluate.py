import torch
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack
from thop import profile, clever_format


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


def eval_pgd20(model, test_loader):
    pgd20_loss = 0
    pgd20_acc = 0
    n = 0
    model.eval()
    atk = torchattacks.PGD(model, eps=8 / 255, steps=20)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        adversarial_images = atk(X, y)

        with torch.no_grad():
            output = model(adversarial_images)
            loss = F.cross_entropy(output, y)
            pgd20_loss += loss.item() * y.size(0)
            pgd20_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print('pgd20_Loss: %.3f | pgd20_Acc: %.3f%% ' % (pgd20_loss / n, 100 * pgd20_acc / n))
    return pgd20_loss / n, pgd20_acc / n


def eval_pgd100(model, test_loader):
    pgd100_loss = 0
    pgd100_acc = 0
    n = 0
    model.eval()
    atk = torchattacks.PGD(model, eps=8 / 255, steps=100)
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        adversarial_images = atk(X, y)

        with torch.no_grad():
            output = model(adversarial_images)
            loss = F.cross_entropy(output, y)
            pgd100_loss += loss.item() * y.size(0)
            pgd100_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    print('fgsm_Loss: %.3f | fgsm_Acc: %.3f%% ' % (pgd100_loss / n, 100 * pgd100_acc / n))
    return pgd100_loss / n, pgd100_acc / n


def eval_CW(model, test_loader):
    CW_loss = 0
    CW_acc = 0
    n = 0
    model.eval()
    atk = torchattacks.CW(model, c=0.5, kappa=0, steps=100, lr=0.01)
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
  evaluate model by Auto Attack
  """
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8. / 255., version='standard')
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_target_classes = 9
    adversary.apgd_targeted.n_restarts = 1

    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        x_adv = adversary.run_standard_evaluation(data, target)


def model_size(model):
    model.eval()
    input = torch.randn(1, 3, 32, 32)
    input = input.cuda()
    flops, params = profile(model, (input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops: ", flops)
    print("params: ", params)
