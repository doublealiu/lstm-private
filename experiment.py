import matplotlib.pyplot as plt
import nltk.tokenize
import numpy as np
import torch.nn as nn
from datetime import datetime

from IPython.core.display_functions import display
from tqdm import tqdm

from utils import caption_utils
from utils.caption_utils import *
from utils.constants import ROOT_STATS_DIR
from utils.dataset_factory import get_datasets
from utils.file_utils import *
from model.model_factory import get_model

from torchvision import transforms


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = ROOT_STATS_DIR

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__save = config_data['experiment']['save']
        self.__load = config_data['experiment']['load']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.example_img = None

        # Criterion and Optimizers set
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(
            params=self.__model.parameters(),
            lr=config_data['experiment']['learning_rate'],
        )

        self.__init_model()

        # Load Experiment Data if available
        if self.__load:
            self.__load_experiment()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir) and self.__load:
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def print_props(self):
        print('=====================================')
        print(f'Running {self.__model.model_type}')
        print(f'for {self.__epochs} epochs!')
        print('=====================================')

    def generate_example_caption(self):
        if self.example_img is not None:
            img = self.example_img
            batched = torch.unsqueeze(img, dim=0).to(self.device)
            generated_captions = self.__model.generate(batched, self.__generation_config)
            print("Generated captions:")
            print(" ".join(generated_captions[0]))

        # Main method to run your experiment. Should be self-explanatory.

    def run(self):
        start_epoch = self.__current_epoch

        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            self.generate_example_caption()
            if epoch % 10 == 0 and epoch != start_epoch:
                print(f'Epoch {epoch}. Running test!')
                self.test()
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train(epoch)
            val_loss = self.__val(epoch)
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            self.plot_stats()
        self.plot_stats(True)

    def show_image(self, image):
        image = image.to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        image = image * std.view(3, 1, 1)
        image = image + mean.view(3, 1, 1)
        image = torch.reshape(image, (3, 256, 256))
        image = image.to("cpu")
        image = transforms.ToPILImage(mode='RGB')(image)
        display(image)

    def __train(self, epoch):
        self.__model.train()
        training_loss = 0
        # genned = False
        pbar = tqdm(total=len(self.__train_loader), desc=f'Epoch {epoch + 1} training')
        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            if self.example_img is None:
                self.example_img = images[0]
                self.show_image(self.example_img)
            # test code ignore
            # if not genned: # generate and print single example
            #     single_batch = torch.unsqueeze(images[0], 0)
            #     single_batch = single_batch.to(self.device)
            #     gen_captions = self.__model.generate(single_batch, self.__generation_config)
            #     # print(gen_captions.size())
            #     print(gen_captions[0])

            self.__optimizer.zero_grad()
            images = images.to(self.device)
            captions = captions.to(self.device)
            # print(f'images: {images.size()}')
            outputs = self.__model(images, captions)
            # print(f'outputs: {outputs.size()}')
            # print(f'captions: {captions.size()}')
            loss = self.__criterion(outputs, captions)
            loss.backward()

            training_loss += loss.item()
            self.__optimizer.step()
            pbar.update(1)
        # avg training loss
        training_loss /= len(self.__train_loader)
        pbar.close()
        # print(f'Epoch {epoch + 1}\tTrain Loss {training_loss}')
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self, epoch):
        self.__model.eval()
        val_loss = 0
        pbar = tqdm(total=len(self.__val_loader), desc=f'Epoch {epoch + 1} validation')
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                outputs = self.__model(images, captions)

                loss = self.__criterion(outputs, captions)

                val_loss += loss.item()
                pbar.update(1)
        val_loss /= len(self.__val_loader)
        pbar.close()
        # print(f'Epoch {epoch + 1}\tVal Loss {val_loss}')
        return val_loss

    # Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0

        pbar = tqdm(total=len(self.__test_loader), desc='Testing...')
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)
                # loss
                outputs = self.__model(images, captions)
                loss = self.__criterion(outputs, captions)
                test_loss += loss
                # bleu
                generated_captions = self.__model.generate(images, self.__generation_config)
                total_bleu1 = 0
                total_bleu4 = 0
                num_bleu = 0
                max_bleu = -1
                max_idx = -1
                for i in range(captions.size()[0]):
                    test_captions = []
                    for annotation in self.__coco_test.imgToAnns[img_ids[i]]:
                        test_caption = annotation['caption']
                        tokenized = nltk.tokenize.word_tokenize(str(test_caption).lower())
                        test_captions.append(tokenized)
                    ex_bleu1 = caption_utils.bleu1(test_captions, generated_captions[i])
                    if ex_bleu1 > max_bleu:
                        max_bleu = ex_bleu1
                        max_idx = i
                    total_bleu1 += ex_bleu1
                    total_bleu4 += caption_utils.bleu4(test_captions, generated_captions[i])
                    num_bleu += 1
                if iter % 10 == 0:
                    self.show_image(images[max_idx])
                    tqdm.write(f'Bleu1: {max_bleu}')
                    tqdm.write(generated_captions[max_idx])
                bleu1 += total_bleu1 / num_bleu
                bleu4 += total_bleu4 / num_bleu
                pbar.update(1)
        bleu1 /= len(self.__test_loader)
        bleu4 /= len(self.__test_loader)
        test_loss /= len(self.__test_loader)
        pbar.close()
        result_str = "Test Loss: {}\tBleu1: {}\tBleu4: {}".format(test_loss, bleu1, bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        if self.__save:
            root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
            model_dict = self.__model.state_dict()
            state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
            torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        if self.__save:
            write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
            write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        if self.__save:
            log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
            if file_name is not None:
                log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self, save=True):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        if self.__save and save:
            plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
