import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import tensorflow as tf
import urllib.request
import zipfile



class Datasets():
    def __init__(self, output_dir="scratch_output"):
        self.data_params_set = False
        self.data_downloaded = False
        self.datasets_made = False
        if isinstance(output_dir, str):
            self.output_dir = Path(output_dir)  
        elif isinstance(output_dir, Path):          
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_data_params(self, 
                        batch_size = 1500,
                        image_size = 64):
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_params_set = True

    def download_data(self):
        # make the data directory
        Path('/tmp/anime').mkdir(exist_ok=True)
        if not Path('/tmp/anime/images/').exists:
            # download the zipped dataset to the data directory
            data_url = "https://storage.googleapis.com/learning-datasets/Resources/anime-faces.zip"
            data_file_name = "animefaces.zip"
            download_dir = '/tmp/anime/'
            urllib.request.urlretrieve(data_url, data_file_name)

            # extract the zip file
            zip_ref = zipfile.ZipFile(data_file_name, 'r')
            zip_ref.extractall(download_dir)
            zip_ref.close()
        self.data_downloaded = True

    def make_train_and_validation_sets(self):
        if self.datasets_made:
            return
        if not self.data_downloaded:
            self.download_data()
        if not self.data_params_set:
            self.set_data_params()

        def get_dataset_slice_paths(image_dir):
            '''returns a list of paths to the image files'''
            image_file_gen = Path(image_dir).iterdir()
            image_paths = [fname for fname in image_file_gen 
                           if fname.suffix in ['.jpg', '.jpeg', '.png']]

            return image_paths


        def map_image(image_filename):
            '''preprocesses the images'''
            img_raw = tf.io.read_file(image_filename)
            image = tf.image.decode_jpeg(img_raw)

            image = tf.cast(image, dtype=tf.float32)
            image = tf.image.resize(image, (self.image_size, 
                                            self.image_size))
            image = image / 255.0  
            image = tf.reshape(image, shape=(self.image_size, 
                                            self.image_size, 
                                            3,))

            return image

        # get the list containing the image paths
        paths = get_dataset_slice_paths("/tmp/anime/images/")

        # shuffle the paths
        random.shuffle(paths)

        # split the paths list into to training (80%) and validation sets(20%).
        paths_len = len(paths)
        train_paths_len = int(paths_len * 0.8)

        train_paths = paths[:train_paths_len]
        val_paths = paths[train_paths_len:]

        # load the training image paths into tensors, create batches and shuffle
        train_files = list(map(str, train_paths))
        training_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        training_dataset = training_dataset.map(map_image)
        training_dataset = training_dataset.shuffle(1000).batch(self.batch_size)

        # load the validation image paths into tensors and create batches
        val_files = list(map(str, val_paths))
        validation_dataset = tf.data.Dataset.from_tensor_slices(val_files)
        validation_dataset = validation_dataset.map(map_image)
        validation_dataset = validation_dataset.batch(self.batch_size)

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.datasets_made = True
        print(f'number of batches in the training set: {len(training_dataset)}')
        print(f'number of batches in the validation set: {len(validation_dataset)}')

    def display_train_data(self, size=9):
        self.display_sample_data("train", size)


    def display_validation_data(self, size=9):
        self.display_sample_data("valid", size)


    def display_sample_data(self, dataset_choice, size):
        '''Takes a sample from a dataset batch, plots it in a grid, and saves/plots it.'''
        if not self.datasets_made:
            self.make_train_and_validation_sets()
        if dataset_choice[0].lower() == "t":
            dataset = self.training_dataset
            data_samp_set = "Train"
        elif dataset_choice[0].lower() == "v":
            dataset = self.validation_dataset
            data_samp_set = "Validation"
        else:
            print("You must choose either the training or validation set with either a 't' or 'v' respectively.")
            return

        dataset = dataset.unbatch().take(size)
        n_cols = round(np.sqrt(size))
        n_rows = size // n_cols + 1
        plt.figure(figsize=(5, 5))
        i = 0
        for image in dataset:
            i += 1
            disp_img = np.reshape(image, (64, 64, 3))
            plt.subplot(n_rows, n_cols, i)
            plt.xticks([])
            plt.yticks([])
            plt.suptitle(data_samp_set + " Images")
            # Set the title of the window
            plt.gcf().canvas.manager.set_window_title(data_samp_set + " Images")
            plt.imshow(disp_img)

        if self.output_dir:
            trgt_dir = self.output_dir / "original_images"
            trgt_dir.mkdir(parents=True, exist_ok=True)
            file_name = data_samp_set + "_sample.png"
            outfile = trgt_dir / file_name
            plt.savefig(outfile)
            print(f"Sample image saved to {outfile}")
        else:
            print("No specified output directory. Images of sample data are not being saved")


if __name__ == '__main__':
    data = Datasets()
    data.set_data_params()
    data.download_data()
    data.make_train_and_validation_sets()
    data.display_sample_data('t', 25)
    data.display_sample_data('v', 18)