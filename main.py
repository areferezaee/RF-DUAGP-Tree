#from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader#, non_deterministic
from tqdm import trange
from utils import *
from torchvision import transforms
from DUAGP_Tree.dualearner import ModelBinaryTree2
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gpytorch
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
classification_report
)

from sklearn.preprocessing import label_binarize

torch.set_printoptions(profile="full")
from argparse import ArgumentParser

torch.set_printoptions(profile="full")
import torch
import time

#from feature_engineering.capsulorhexis import get_capsulorhexis_feats
from feature_engineering.JIGSAWS import get_jigsaws_feats


# =========================
# Main
# =========================

class Main:
    def __init__(self, args):
        
        self.args = args
        set_seed(self.args.seed)
        length = self.args.length
        
        self.device = torch.device('cpu')#('cuda')
        self.batch_size = args.batch_size
        self.mode= 'train'
        
        print(f"Using device: {self.device}")

        #======= Data =======
        if self.args.dataset == 'JIGSAWS':

          self.X_train, self.Y_train, self.Xtrain_idx, self.nis_train = get_jigsaws_feats('train')
          self.X_val, self.Y_val, self.Xval_idx, self.nis_val = get_jigsaws_feats('val')
          self.X_test, self.Y_test, self.Xtest_idx, self.nis_test = get_jigsaws_feats('test')
          self.num_classes = 3
          self.args.feature_length = torch.tensor([1, self.X_train.size(1)])
        elif self.args.dataset == 'Capsulorhexis':

          self.X_train, self.Y_train, self.Xtrain_idx, self.nis_train = get_capsulorhexis_feats('train')
          self.X_val, self.Y_val, self.Xval_idx, self.nis_val = get_capsulorhexis_feats('val')
          self.X_test, self.Y_test, self.Xtest_idx, self.nis_test = get_capsulorhexis_feats('test')
          self.num_classes = 2
          self.args.feature_length = torch.tensor([1, self.X_train.size(1)])
        self.allclses=[]
        self.allpreds=[]
        #======= Model =======
        # build initial model
        self.gp_counter = 0
        self.tree_model = ModelBinaryTree2(self.args, self.device, pretrained=False)
        self.tree_model.to(self.device)
        # ==========
        # Initialization Train
        # ==========
        self.num_epochs = 20
        self.built_tree = False
        self.to_print = False
        self.writer = SummaryWriter()
        self.start_time = time.time()
        self.gibbs=False
        
        self.tree_optimizer = optim.Adam(self.tree_model.parameters(), lr=1e-5)
    # =========================
    # train-val-test 
    # =========================
    def train(self):
       #model = model.cuda()   
       #max_grad = 100
       self.tree_model.train()
       acc = .0
       self.tree_optimizer.zero_grad()
       with torch.autograd.detect_anomaly():
         loss, pred, loss2 = self.tree_model(self.X_train, self.Y_train, self.Xtrain_idx, self.nis_train, self.num_classes, to_print=self.to_print, mode=self.mode)
       predcls = torch.max(pred, dim=1)[1]
       corrects = torch.sum((predcls == self.Y_train).int())
       acc = corrects.item()                 
       
       # optimize GP hyper-parameters 
       
       loss.backward(retain_graph=True )
       self.tree_optimizer.step()
       return loss, loss2, acc
       

    def val(self):
      self.tree_model.eval()
      acc = .0
      loss, pred = self.tree_model(self.X_val, self.Y_val, self.Xval_idx, self.nis_val, self.num_classes, to_print=self.to_print, mode=self.mode) 
      predcls = torch.max(pred, dim=1)[1]
      corrects = torch.sum((predcls == self.Y_val).int())
      acc = corrects.item()
      return loss, acc

    def test(self):

      self.tree_model.eval()
      loss, pred = self.tree_model(self.X_test, self.Y_test, self.Xtest_idx, self.nis_test, self.num_classes, to_print=self.to_print, mode=self.mode)
      predcls = torch.max(pred, dim=1)[1]
      corrects = torch.sum((predcls == self.Y_test).int())
      acc = corrects.item()
      self.allpreds.append(predcls)
      self.allclses.append(self.Y_test)
      ##################################################metrics comp#####################################################################
      y_pred = predcls.detach().cpu().numpy()
      y_true = self.Y_test.detach().cpu().numpy()
      y_prob = pred.detach().cpu().numpy()

      accu = accuracy_score(y_true, y_pred)
      precision = precision_score(y_true, y_pred, average='macro')
      recall = recall_score(y_true, y_pred,average='macro')
      f1 = f1_score(y_true,y_pred,average='macro')

      cm = confusion_matrix(y_true, y_pred)

      # One-hot encoding
      y_true_bin = label_binarize(y_true, classes=[0,1,2])
      if args.dataset == 'Capsulorhexis':
         auc = roc_auc_score( y_true_bin, y_prob, average='macro',multi_class='ovr')

      #torch.save(self.allclses, 'allclses3.pt')
      #torch.save(self.allpreds, 'allpreds3.pt')
      return loss, acc
    # === Run ===
    def run(self):
      for epoch in range(self.num_epochs):
        if epoch == (self.num_epochs - 1):
           self.mode= 'test'
           loss, acc = self.test()
           print('TEST:loss, acc', loss, acc)
        else:
           # build tree
           if epoch == 0 and not self.built_tree:
              
              self.gp_counter += self.tree_model.build_base_tree(self.X_train, self.Y_train, self.Xtrain_idx)
              self.built_tree = True
           self.mode= 'train'
           loss, loss2, acc = self.train()
           print('TRAIN:loss, loss2, acc', loss, loss2, acc)
           self.mode= 'val'
           loss, acc = self.val()
           print('VAL:loss, acc', loss, acc)
           


# =========================
# CLI
# =========================

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='FSCIL GP - trainer')
   parser.add_argument('--script-name', default='CUB')
   parser.add_argument('--dataset', type=str, default='JIGSAWS')
   parser.add_argument('--exp-name', type=str, default='', metavar='N',
                    help='experiment name suffix')
   parser.add_argument('--num-sessions', type=int, default=4, help='Number of few shot sessions')
   parser.add_argument('--N-way', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='66,22,22,22',
                    help='number of classes per session')
   parser.add_argument('--N-shot', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='10000000,5,5,5,5,5,5,5,5,5,5',
                    help='Number of samples per session')
   parser.add_argument('--feature-length', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='456704',
                    help='layers after feature extractor')
   parser.add_argument('--base-num-epochs', type=int, default=100, help='Number of epochs')
   parser.add_argument('--move-to-gp-epoch', type=int, default=20, help='epoch to start training with GP')
   parser.add_argument('--dataroot', type=str, default='./dataset', help='dataset root')
   parser.add_argument('--scheduler', default=True, type=str2bool, help='use learning rate scheduler')
   parser.add_argument('--optimizer', default='sgd', choices=['adam', 'sgd'], type=str,
                    help='use learning rate scheduler')
   parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
   parser.add_argument('--base-lr', default=1e-2, type=float, help='learning rate')
   parser.add_argument('--natural-lr', default=.1, type=float,
                    help='natural GA learning rate. If not using stochastic updates - may use a value of 1.')
   parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
   parser.add_argument('--batch_size', type=int, default=1, help='batch size')
   parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
   parser.add_argument('--base-milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='40,60')
   parser.add_argument('--num-steps', type=int, default=10, help='number of sampling iterations')
   parser.add_argument('--num-draws', type=int, default=30, help='number of parallel gibbs chains')
   #parser.add_argument('--kernel-function', type=str, default='ARDRBFKernel',
   #                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'AdditiveKernel','PowRBFKernel', 'PeriodicKernel', 'ARDRBFKernel'],
   #                    help='kernel function')
   #=['UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel'],
   #['PowRBFKernel', 'PowRBFKernel', 'PowRBFKernel', 'PowRBFKernel', 'PowRBFKernel']
   parser.add_argument('--kernel-function', type=str, default=['UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel',
                                                             'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel', 'UaARDRBFKernel'],
                    choices=[['RBFKernel', 'LinearKernel', 'MaternKernel'], ['PeriodicKernel', 'PeriodicKernel', 'PeriodicKernel'], ['UaARDRBFKernel','ARDRBFKernel']],
                    help='kernel function')
   parser.add_argument('--num-inducing-points', type=int, default=5,
                    help='Number of inducing points per class')
   parser.add_argument('--learn-location', default=True, type=str2bool, help='learn inducing point location')
   parser.add_argument('--gibbs-outputscale', type=float, default=4., help='output scale')
   parser.add_argument('--gibbs-lengthscale', type=float, default=1., help='length scale')
   parser.add_argument('-length', type=int, default=1)
   parser.add_argument('--outputscale', type=float, default=2., help='output scale')
   parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
   parser.add_argument('--eval-every', type=int, default=10, help='num. epochs between test set eval')
   parser.add_argument('--out-dir', type=str, default='./outputs', help='Output dir')
   parser.add_argument('--seed', default=42, type=int, help='random seed')
   parser.add_argument('--num-workers', default=48, type=int, help='num wortkers')
   parser.add_argument('--gpus', type=str, default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

   args = parser.parse_args()
   trainer = Main(args)
   trainer.run()

