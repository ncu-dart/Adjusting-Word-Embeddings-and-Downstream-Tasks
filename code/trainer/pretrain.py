import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int, seq_len: int,
                 train_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 loss1='Euclidean', loss2='MSE', alpha=0.5,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.seq_len = seq_len

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train data loader
        self.train_data = train_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # MSELoss for predicting the masked_token
        self.criterion = nn.MSELoss()

        self.log_freq = log_freq
        
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.train_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            bert_output, original_emb = self.model.forward(data["bert_input"])
            
            # 2. MSELoss of predicting masked token word
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)

            key = bert_output[:,0,:].unsqueeze(1).repeat(1, self.seq_len, 1)
            label = data["syn_label"].type(torch.FloatTensor).cuda()
            target = label
            for idx1, tar in enumerate(target):
                for idx2, t in enumerate(tar):
                    if t == -1:
                        tar[idx2] = 0
            
            """
            # loss_1: wrong loss function
            loss_1 = (torch.mul(cos(bert_output, key), label).sum(dim=1)/torch.abs(label).sum(dim=1)).mean()
            """
            if self.loss1 == 'Cosine':
                loss_1 = ((torch.sub(target,torch.mul(cos(bert_output, key), label))).sum(dim=1)/torch.abs(label).sum(dim=1)).mean()
            if self.loss1 == 'Euclidean':
                loss_1 = (torch.mul(torch.sqrt(torch.sum((bert_output-key)**2)), label).sum(dim=1)/torch.abs(label).sum(dim=1)).mean()
            if self.loss2 == 'MSE':
                loss_2 = (torch.mul(((bert_output-original_emb)**2).mean(dim=2), torch.abs(label)).sum(dim=1)/torch.abs(label).sum(dim=1)).mean()
            if self.loss2 == 'Euclidean':
                loss_2 = (torch.mul(torch.sqrt(torch.sum((bert_output-original_emb)**2)), label).sum(dim=1)/torch.abs(label).sum(dim=1)).mean()

            loss = self.alpha * loss_1 + (1 - self.alpha) * loss_2

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            else:
                with open('../output/embeddings/raw/result_input_iter{}.pkl'.format(i), "wb") as fb:
                    pickle.dump(data["bert_input"], fb)
                with open('../output/embeddings/raw/result_output_iter{}.pkl'.format(i), "wb") as fb:
                    pickle.dump(bert_output, fb)

            # next sentence prediction accuracy
            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path