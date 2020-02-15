import argparse
from engine import *
from models import *
from util import *
from dataLoader import *

from datasets.ProgramWeb import ProgramWebDataset, split_train_val_dataset


parser = argparse.ArgumentParser(description='Training Super-parameters')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset (e.g. data/')

# parser.add_argument('-num_classes', default=115, type=int, metavar='N',
#                     help='number of domains')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_PW():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # train_dataset, val_dataset = CrossValidationSplitter(args.data,  inp_name='data/ProgrammerWeb/train.csv')
    dataset = ProgramWebDataset('data/ProgrammerWeb/programweb-data.csv')
    assert 1 == 0
    model = gcn_bert(num_classes=len(dataset.tag2id), t=0.4, co_occur_mat=dataset.co_occur_mat)
    train_dataset, val_dataset = split_train_val_dataset(dataset)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': args.num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/ProgrammerWeb/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_PW()
