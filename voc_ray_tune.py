import argparse

from engine import *
from models.attention_pair_norm_2layer import *
from optuna.samplers import TPESampler
from voc import *
from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.02207089093258345, type=float,
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
args = parser.parse_args()

def trainer(config,data_dir = None,checkpoint_dir = None):
    global best_prec1, use_gpu
    
    
    use_gpu = torch.cuda.is_available()
    

    # define dataset
    train_dataset = Voc2007Classification(data_dir, 'trainval', inp_name=data_dir+'/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(data_dir, 'test', inp_name=data_dir+'/voc_glove_word2vec.pkl')

    num_classes = 20
    
    # load model
    if args.image_size==448:
        model = attention_gcn_pairnorm(num_classes=num_classes, t=config["threshold"], adj_file=args.data+'/voc_adj.pkl')
    else:
        model = attention_gcn_pairnorm(num_classes=num_classes, t=config["threshold"], adj_file=args.data+'/voc_adj.pkl')
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model = model.to(device)
    
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                            lr=config["lr"],
                            momentum=config["momentum"],
                            weight_decay=args.wd)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = config["lr"]
    
    if args.evaluate:
        state['evaluate'] = True

    engine = GCNMultiLabelMAPEngine(state)
    mAP = engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
    tune.report(mAP=mAP)



if __name__ == '__main__':
    # for early stopping
    scheduler = ASHAScheduler(
        metric="mAP",
        mode="max",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "mAP"])
    
    config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
            "threshold":tune.quniform(0.2,0.9,0.1),
    }
    print(os.getcwd())
    analysis = tune.run(
        run_or_experiment = partial(trainer,data_dir=args.data,checkpoint_dir="../RayTuneCheckpoints"),
        num_samples=20,
        name="mAP_Tuning",
        scheduler=scheduler,
        stop={
            "mAP": 0.98,
        },
        resources_per_trial={
            'cpu':1,
            'gpu':1
        },
        config=config,
        progress_reporter=reporter
    )

    best_trial = analysis.get_best_trial("mAP", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation mAP score: {}".format(
        best_trial.last_result["mAP"]))