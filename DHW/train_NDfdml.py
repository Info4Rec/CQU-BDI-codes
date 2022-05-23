import os
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import get_datasets, get_data_loaders
from test import test
from model import get_model
from transforms import get_transform
import torch.nn.functional as F
from NDFDML import nd,nd_loss



from losses.get_loss import  get_loss


class Trainer:
    def __init__(self, args):

        # Training configurations
        self.method = args.method
        self.dataset = args.dataset
        self.dim = args.dim
        self.lr_init = args.lr_init
        self.gamma_m = args.gamma_m
        self.gamma_s = args.gamma_s
        self.batch_size = args.batch_size
        self.val_batch_size = self.batch_size // 2
        self.iteration = args.iteration
        self.evaluation = args.evaluation
        self.show_iter = 1000
        self.update_epoch = args.update_epoch
        self.balanced = args.balanced
        self.instances = args.instances
        self.inter_test = args.intertest
        self.cm = args.cm
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_class = args.batch_size // args.instances
        self.classes = args.classes
        self.pretrained = args.pretrained
        self.model_save_interval = args.model_save_interval


        self.file_name = '{}_{}_{}'.format(
            self.method,
            self.dataset,
            self.iteration,
        )
        print('========================================')
        print(json.dumps(vars(args), indent=2))
        print(self.file_name)

        # Paths

        self.root_dir = os.path.join('/', 'data')
        self.data_dir = os.path.join(self.root_dir, self.dataset)
        self.model_dir = self._get_path('./trained_model')
        self.plot_dir = self._get_path('./plot_model')
        self.code_dir = self._get_path(os.path.join('codes', self.dataset))
        self.fig_dir = self._get_path(os.path.join('fig', self.dataset, self.file_name))

        # Preparing data
        self.transforms = get_transform()
        self.datasets = get_datasets(dataset=self.dataset, data_dir=self.data_dir, transforms=self.transforms)

        self.data_loaders = get_data_loaders(
            datasets=self.datasets,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            n_instance=self.instances,
            balanced=self.balanced,
            #cm=self.cm_sampler if self.cm else None
        )
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'test']}


        self.mean = (torch.zeros((self.classes,self.classes)).add(1.5)-1.0*torch.eye(self.classes)).to(self.device)
        self.std = (torch.zeros((self.classes,self.classes)).add(0.15)).to(self.device)
        self.last_delta_mean = torch.zeros((self.classes,self.classes)).to(self.device)
        self.last_delta_std = torch.zeros((self.classes,self.classes)).to(self.device)

        
        self.ndmodel = nd.NDfdml(n_class=self.n_class,batch_size=self.batch_size,instances=self.instances,pretrained=self.pretrained).to(self.device)
        
        
        optimizer_c = optim.SGD(
            [
                {'params': self.ndmodel.googlelayer.parameters()},
                {'params': self.ndmodel.embedding_layer.parameters(), 'lr': self.lr_init * 10, 'momentum': 0.9}
            ],
            lr=self.lr_init, momentum=0.9
        )


        self.scheduler = lr_scheduler.StepLR(optimizer_c, step_size=4000, gamma=0.9)



    @staticmethod
    def _get_path(path): 
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path


    def __inter_test(self, it):
        print("test in {} iteration".format(it))
        test_ft, test_labels = self.generate_codes()
        metrics = test(self.file_name, test_ft, test_labels)


    def feed_embeddings(self, dataset, numpy=False):
        code = torch.FloatTensor().to(self.device)
        label = torch.LongTensor()
        start = time.time()
        for sample in self.data_loaders[dataset]:
            inputs = sample['image'].to(self.device)
            labels = sample['label']
            with torch.set_grad_enabled(False):
                feature = self.ndmodel.googlelayer(inputs)
                outputs = self.ndmodel.embedding_layer(feature)
            code = torch.cat((code, outputs), 0)
            label = torch.cat((label, labels.long()), 0)
        end = time.time()
        print('Finish generating {} {} codes in {:.0f}s'.format(
            len(code),
            dataset,
            end - start))
        if numpy:
            code = code.cpu().numpy()
            label = label.cpu().numpy()
        return code, label, end-start

    def generate_codes(self):
        self.ndmodel.eval()
        test_feature, test_label, _ = self.feed_embeddings('test', numpy=True)
        np.save(os.path.join(self.code_dir, 'test_ft.npy'), test_feature)
        np.save(os.path.join(self.code_dir, 'test_labels.npy'), test_label)
        return test_feature, test_label

    def save_model(self):
        model_path = os.path.join(self.model_dir, self.file_name+'.pkl')
        torch.save(self.ndmodel.state_dict(), model_path)

    def load_model(self):
        model_path = os.path.join(self.model_dir, self.file_name+'.pkl')
        if not os.path.exists(model_path):
            raise Exception('Can not find trained model {}'.format(model_path))
        self.ndmodel.load_state_dict(torch.load(model_path))



    def train(self):
        since = time.time()
        start = time.time()

        self.ndmodel.train()

        running_iter = 0
        running_loss = 0.0
        running_count = 0
        running_epoch = 0

        optimizer_c = optim.SGD(
            [
                {'params': self.ndmodel.googlelayer.parameters()},
                {'params': self.ndmodel.embedding_layer.parameters(), 'lr': self.lr_init * 10, 'momentum': 0.9}
            ],
            lr=self.lr_init, momentum=0.9
        )



        print('Start training')
        while running_iter < self.iteration:

            for sample in self.data_loaders['train']:
                if running_iter > self.iteration:
                    break
                inputs = sample['image'].to(self.device)
                labels = sample['label'].to(self.device)
                

                optimizer_c.zero_grad()

                with torch.set_grad_enabled(True):
                    
                    last_delta_mean,last_delta_std,loss = self.ndmodel(inputs,labels,self.mean,self.std,self.last_delta_mean,self.last_delta_std,running_iter)
                    
                    loss.backward()
                   
                    
                    optimizer_c.step()



                if  running_epoch == 0 or running_epoch % self.update_epoch == 0:
                    self.mean = self.gamma_m * last_delta_mean + self.mean
                    self.std = self.gamma_s * last_delta_std + self.std
                  

                if (running_iter + 1) % self.show_iter == 0:
                    self.gamma_m = 0.9 * self.gamma_m
                    self.gamma_s = 0.9 * self.gamma_s
                    
                    print('Iteration {}/{} loss {:.4f} Spending {:.0f}s'.format(
                        running_iter + 1,
                        self.iteration,
                        loss.item(),
                        time.time() - start
                    ))
                    start = time.time()

                if (running_iter + 1) % self.inter_test == 0:
                    self.__inter_test((running_iter+1))
                    self.ndmodel.train()


                self.scheduler.step()
                running_iter += 1
            running_epoch += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return self.ndmodel

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DM in Metric Baselines')
    parser.add_argument('--lr_init', default=9e-05, type=float)
    parser.add_argument('--iteration', default=20000, type=int)
    parser.add_argument('-p', '--pretrained', action='store_true', default=True, help='Use pretrained weight.')
    parser.add_argument('--dim', default=1024, type=int)
    parser.add_argument('--batch-size', default=120, type=int)
    parser.add_argument('--intertest', default=1000, type=int)
    parser.add_argument('--model_save_interval', default=4000, type=int)
    parser.add_argument('--dataset', default='CUB', type=str)#CARS196
    parser.add_argument('-e', '--evaluation', dest='evaluation', action='store_true')
    parser.add_argument('--method', default='Triplet', type=str)
    parser.add_argument('--balanced', default=True, action='store_true')
    parser.add_argument('--instances', default=3, type=int)
    parser.add_argument('--cm', default=False,action='store_true')
    parser.add_argument('--gamma_m',default=0.1, type=float)
    parser.add_argument('--gamma_s',default=0.03,type=float)
    parser.add_argument('--classes',default=100,type=int)
    parser.add_argument('--update_epoch',default=50,type=int)
    trainer = Trainer(parser.parse_args())

    if not trainer.evaluation:
        trainer.train()
        trainer.save_model()
    else:
        trainer.load_model()
    test_ft, test_labels = trainer.generate_codes()
    test(trainer.file_name, test_ft, test_labels)

