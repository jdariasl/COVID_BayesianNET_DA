import argparse
import torch
import lightning as L
import numpy as np
import utils.util as util
from trainer.train import initialize,  validation, validation_bayesian, initialize_from_saved_model
#from torch.utils.tensorboard import SummaryWriter



def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    args.save = args.save + args.model + '_' + util.datestr()

    if (args.cuda):
        torch.cuda.manual_seed(SEED)
    if args.new_training:
        model, training_generator, val_generator, Last_epoch, bflag = initialize_from_saved_model(args)
    else:
        model, training_generator, val_generator, bflag = initialize(args)
        Last_epoch = 0

    #-------------------------------------------------
    # Initialize a trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
    )
    #-------------------------------------------------

    best_pred_loss = 0#lo cambie por balanced accuracy

    print('Checkpoint folder ', args.save)
    # writer = SummaryWriter(log_dir='../runs/' + args.model, comment=args.model)
    for epoch in range(1, args.nEpochs + 1):
        # Train the model ⚡
        trainer.fit(model, training_generator)
        if bflag:
            val_metrics, confusion_matrix = validation_bayesian(args, model, val_generator, Last_epoch+epoch)
        else:
            val_metrics, confusion_matrix = validation(args, model, val_generator, Last_epoch+epoch)
       
        BACC = BalancedAccuray(confusion_matrix.numpy())
        val_metrics.replace({'bacc': BACC})
        best_pred_loss = util.save_model(model, args, val_metrics, Last_epoch+epoch, best_pred_loss, confusion_matrix)

        print(confusion_matrix)
        model.lr_schedulers.step(val_metrics.avg_loss())

def BalancedAccuray(CM):
    Nc = CM.shape[0]
    BACC = np.zeros(Nc)
    for i in range(Nc):
        BACC[i] = CM[i,i]/np.sum(CM[i,:])
    print(np.mean(BACC))
    return np.mean(BACC)



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_training', action='store_true', default=False,
                        help='load saved_model as initial model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--dataset_name', type=str, default="COVIDx")
    parser.add_argument('--nEpochs', type=int, default=24)
    parser.add_argument('--n_monte_carlo', type=int, default=20, help='number of Monte Carlo runs during training')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='DenseNet',
                        choices=('DenseNet','BDenseNet','EfficientNet','BEfficientNet'))
    parser.add_argument('--init_from', action='store_true', default=False)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='COVID_BayesianNET/Data/OrgImagesRescaled/',
                        help='path to dataset ')
    parser.add_argument('--pre_processing', type=str, default='None',
                        choices=('None','Equalization','CLAHE'))
    parser.add_argument('--saved_model', type=str, default='COVID_BayesianNET/models_saved/Model_best_checkpoint.pth.tar',
                        help='path to save_model ')
    parser.add_argument('--save', type=str, default='COVID_BayesianNET/models_saved/',
                        help='path to checkpoint ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
