import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg_module
import os
import argparse
import torch.backends.cudnn as cudnn

best_prec1 = 0
model_names=sorted(name for name in vgg_module.__dict__)
model_names = ['vgg11','vgg16']
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--save-dir', dest='save_dir')

def main():
    global best_prec1, args
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    #model = vgg_module.vgg16(vgg_module.VGG)() ## model
    model = vgg_module.__dict__['vgg11']()
    model=model.cuda()
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data',train = True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32,4), ## transforms.RandomCrop(32,32)
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std = [0.229,0.224,0.225]) ## transforms.Normalize
        ]), download = True),
        batch_size = 256, shuffle = True, num_workers = 0, pin_memory = True
        ## num_workers
        )
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data',train=False, transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.465, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])), batch_size=256, shuffle=False, num_workers = 0, pin_memory = True)
    
    cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9, 
                                weight_decay = 5e-4)
    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    for epoch in range(0,30):
        #adjust_learning_rate(optimizer, epoch)
        
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
            'epoch' : epoch +1,
            'state_dict' : model.state_dict(),
            'best_prec1' : best_prec1,
        }, is_best, filename=os.path.join(args.save_dir,'checkpoint_{}.tar'.format(epoch)))
        print(f"{epoch+1}th epoch is done")
    
def adjust_learning_rate(optimizer, epoch):
    lr = 5e-4 * (0.5**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        
        output = model(input).cuda()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        
        prec1 = accuracy(output.data, target)[0]  # top1
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))    
        if i % 20 ==0:
            print(f'Epoch : {epoch}[{i}/{len(train_loader)}],  Loss:{losses.val:.4f}({losses.avg:.4f}),  Prec@1 : {top1.val:.3f}({top1.avg:.3f})')
                        
def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        
        with torch.no_grad():
            output=model(input)
            loss = criterion(output, target)

        output=output.float() 
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0] # top1
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        if i % 20 ==0:
            print(f'Test : [{i}/{len(val_loader)}],  Loss : {losses.val:.4f}({losses.avg:.4f}),  Prec@1 : {top1.val:.3f}({top1.avg:.3f})')
    print(f' * Prec@1 {top1.avg:.3f}')
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    
    maxk = max(topk)
    batch_size = target.size(0)
    
    _,pred=output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk: 
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
    

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self, val, n=1):
        self.val = val
        self.sum+=val*n
        self.count+=n
        self.avg = self.sum/self.count
        

if __name__ == '__main__':
    main()
    
