import argparse
from engine import *
from models import *
from util import *
from dataLoader import *


parser = argparse.ArgumentParser(description='Training Super-parameters')
#
# parser.add_argument('data_path', default='data/ProgrammerWeb/', type=str,
#                     help='path to dataset (e.g. data/')
# parser.add_argument('-num_classes', default=115, type=int, metavar='N',
#                     help='number of domains')
parser.add_argument('-seed', default=0, type=int, metavar='N',
                    help='random seed')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
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
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def multiLabel_text_classify():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    dataset = build_dataset('data/ProgrammerWeb/programweb-data.csv','data/ProgrammerWeb/domainnet.csv')
    # dataset = build_dataset(os.path.join(args.data_path, 'data/ProgrammerWeb/programweb-data.csv'),
    #                         os.path.join(args.data_path, 'data/ProgrammerWeb/tagnet.csv'))

    dataset.data[1450] = dataset.data[0]
    dataset.data[4560] = dataset.data[0]
    dataset.data[8744] = dataset.data[0]
    dataset.data[1333] = dataset.data[0]
    dataset.data[10733] = dataset.data[0]
    dataset.data[5590] = dataset.data[0]

    encoded_tag, tag_mask = dataset.encode_tag()
    # data_block = CrossValidationSplitter(dataset, seed)  #Shuffle the data and divide it into ten blocks（store dataIDs）

    # valData_block = 9  # choose a block as validation data

    train_dataset, val_dataset = load_train_val_dataset(dataset)

    model = gcn_bert(num_classes=len(dataset.tag2id), t=0.4, co_occur_mat=dataset.co_occur_mat)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': dataset.get_tags_num()}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/ProgrammerWeb/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['encoded_tag'] = encoded_tag
    state['tag_mask'] = tag_mask
    state['device_ids'] = args.device_ids
    state['print_freq'] = args.print_freq
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    multiLabel_text_classify()
