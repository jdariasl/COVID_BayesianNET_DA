import torch
import torch.optim as optim

def get_output_shape(model, image_dim):
    return model(torch.rand(*image_dim)).data.shape

def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
class Metrics:
    def __init__(self, path, keys=None, writer=None):
        self.writer = writer

        self.data = {'correct': 0,
                     'total': 0,
                     'loss': 0,
                     'accuracy': 0,
                     'bacc':0,
                     }
        self.save_path = path

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    def update_key(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.data[key] += value

    def update(self, values):
        for key in self.data:
            self.data[key] += values[key]
    
    def replace(self, values):
        for key in values:
            self.data[key] = values[key]

    def avg_acc(self):
        return self.data['correct'] / self.data['total']

    def avg_loss(self):
        return self.data['loss'] / self.data['total']

    def save(self):
        with open(self.save_path, 'w') as save_file:
            a = 0  # csv.writer()
            # TODO

def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print("Epoch:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}".format(epoch,
                                                                                         num_samples,
                                                                                         len(
                                                                                             trainloader) * args.batch_size,
                                                                                         metrics.data[
                                                                                             'loss'] / num_samples,
                                                                                         metrics.data[
                                                                                             'correct'] /
                                                                                         metrics.data[
                                                                                             'total']))

def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + "\n SUMMARY EPOCH:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\tBalancedAccuracy:{:.2f}\n".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples ,
                                                                                                     metrics.data[
                                                                                                         'loss'] / num_samples,                                                                             
                                                                                                     metrics.data[
                                                                                                         'correct'] /
                                                                                                     metrics.data[
                                                                                                         'total'],
                                                                                                     metrics.data[
                                                                                                         'bacc']/num_samples))