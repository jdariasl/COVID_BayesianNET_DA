import lightning as L
from torch.autograd import Function
import torch.nn.functional as F
import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from utils.util_model import get_output_shape, select_optimizer, Metrics, print_summary
from model.loss import crossentropy_loss
from model.metric import accuracy
from sklearn.metrics import balanced_accuracy_score
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Model_DA(L.LightningModule):
    def __init__(self, model, args, weights, n_databases):
        super().__init__()
        self.model_base = model
        self.domain_classifier = torch.nn.Sequential()
        dim_features = get_output_shape(self.model_base.features,(1, 3, 224, 224))
        print(f'Dim = {dim_features}')
        self.domain_classifier.add_module('dc_l1',torch.nn.Linear(dim_features, 256))
        self.domain_classifier.add_module('dc_l2',torch.nn.Linear(256, n_databases))
        self.lr_schedulers = ReduceLROnPlateau(self.optimizers(), factor=0.5, patience=3, min_lr=1e-5, verbose=True)
        self.args = args
        self.class_weight = weights[0]
        self.db_weight = weights[1]
        self.metrics = Metrics('')
        self.metrics.reset()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        if self.training:
            class_output = self.model_base(x)
            feature = self.model_base.features(x)
            feature = F.relu(feature, inplace=True)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = torch.flatten(feature,1)
            reverse_feature = ReverseLayerF.apply(feature, 1)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output
        else:
            class_output = self.model_base(x)
            return class_output

    def training_step(self, batch):
        self.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()

        input_data, target = batch
        target_class = target['label_class']
        db = target['label_db']

        out_class, out_da = self(input_data)
        loss_class = crossentropy_loss(out_class, target_class, weight=self.class_weight)
        loss_da = crossentropy_loss(out_da, db, weight=self.db_weight)
        loss = loss_class + loss_da
        loss.backward()
        optimizer.step()
        correct, total, acc = accuracy(out_class, target)

        #num_samples = batch_idx * self.args.batch_size + 1
        _, output_class = out_class.max(1)
        bacc = balanced_accuracy_score(target.cpu().detach().numpy(),output_class.cpu().detach().numpy())
        self.metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})

        return 

    def on_train_epoch_end(self):
        print_summary(self.args, self.metrics, mode="Training")
        self.metrics.reset()

    def configure_optimizers(self):
        return select_optimizer(self.args,self)


class BModel_DA(L.LightningModule):
    def __init__(self, model, args, weights, n_databases):
        super().__init__()
        self.model_base = model
        self.domain_classifier = torch.nn.Sequential()
        dim_features = get_output_shape(self.model_base.features,(1, 3, 224, 224))
        print(f'Dim = {dim_features}')
        self.domain_classifier.add_module('dc_l1',torch.nn.Linear(dim_features, 256))
        self.domain_classifier.add_module('dc_l2',torch.nn.Linear(256, n_databases))
        self.lr_schedulers = ReduceLROnPlateau(self.optimizers(), factor=0.5, patience=3, min_lr=1e-5, verbose=True)
        self.args = args
        self.class_weight = weights[0]
        self.db_weight = weights[1]
        self.metrics = Metrics('')
        self.metrics.reset()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        if self.training:
            class_output = self.model_base(x)
            feature = self.model_base.features(x)
            feature = F.relu(feature, inplace=True)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = torch.flatten(feature,1)
            reverse_feature = ReverseLayerF.apply(feature, 1)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output
        else:
            class_output = self.model_base(x)
            return class_output

    def training_step(self, batch):
        self.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()

        input_data, target = batch
        target_class = target['label_class']
        db = target['label_db']

        out_class, out_da = self(input_data)
        kl = get_kl_loss(self.model_base)
        ce_loss = crossentropy_loss(out_class, target_class, weight=self.class_weight)
        loss_da = crossentropy_loss(out_da, db, weight=self.db_weight)
        loss = loss_da + ce_loss + kl / self.args.batch_size 
        
        loss.backward()
        optimizer.step()

        self.eval()
        with torch.no_grad():
            output_mc = []
            for _ in range(self.args.n_monte_carlo):
                logits = self(input_data)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)
            output = torch.stack(output_mc)  
            pred_mean = output.mean(dim=0)

        correct, total, acc = accuracy(pred_mean, target)

        _, output_class = pred_mean.max(1)
        bacc = balanced_accuracy_score(target.cpu().detach().numpy(),output_class.cpu().detach().numpy())
        self.metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})

    def on_train_epoch_end(self):
        print_summary(self.args, self.metrics, mode="Training")
        self.metrics.reset()
        
    def configure_optimizers(self):
        return select_optimizer(self.args,self)

def BDenseNet(n_classes=3, saved_model = ''):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    if saved_model:
        checkpoint = torch.load(saved_model,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    #Turn model into a Bayesian version (in place)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    return model

def DenseNet(n_classes=3):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    
    return model

def BEfficientNet(n_classes=3, saved_model = ''):
    model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        #torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=2304, out_features=n_classes)
        ) 
    if saved_model:
        checkpoint = torch.load(saved_model,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
  
    #Turn model into a Bayesian version (in place)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    return model

def EfficientNet(n_classes=3):

    model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        #torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=2304, out_features=n_classes)
        ) 

    return model

def DA_model(model_base,args,weights,b_flag=False,n_data_bases=8):
    if b_flag:
        return BModel_DA(model_base,args,weights,n_data_bases)
    else:
        return Model_DA(model_base,args,weights,n_data_bases)