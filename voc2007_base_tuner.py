import argparse

from engine import *
from models.attention_pair_norm_2layer import *
from optuna.samplers import TPESampler
from voc import *
import optuna 
from optuna.trial import TrialState

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


def objective(trial):
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()

    torchvision_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ])


    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval', inp_name=args.data+'/voc_glove_word2vec.pkl',transform=torchvision_transform)
    val_dataset = Voc2007Classification(args.data, 'test', inp_name=args.data+'/voc_glove_word2vec.pkl')

    num_classes = 20
    
    args.lr = trial.suggest_float('learning rate ',9e-3,3e-1,log = True)
    t = trial.suggest_categorical('Threshold',[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    #args.weight_decay = trial.suggest_float('weight decay',7e-5,3e-4,log = True)
    # load model
    if args.image_size==448:
        model = attention_gcn_pairnorm(num_classes=num_classes, t=t, adj_file=args.data+'/voc_adj.pkl',trial = trial)
    else:
        model = attention_gcn_pairnorm(num_classes=num_classes, t=t, adj_file=args.data+'/voc_adj.pkl')
    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    
    if args.evaluate:
        state['evaluate'] = True

    engine = GCNMultiLabelMAPEngine(state)
    return engine.learning(model, criterion, train_dataset, val_dataset, optimizer,trial)



if __name__ == '__main__':

    sampler = TPESampler(seed = 42)
    study = optuna.create_study(direction="maximize",study_name="Find most important hyperparameters",sampler = sampler)
    study.optimize(objective, n_trials=20)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    importance_dict = optuna.importance.get_param_importances(study)
    print(importance_dict)
    mdi = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
    mdi_importance = mdi.evaluate(study = study)
    print(mdi_importance)
    fanova =  optuna.importance.FanovaImportanceEvaluator()
    fanova_importance = fanova.evaluate(study = study)
    print(fanova_importance)

