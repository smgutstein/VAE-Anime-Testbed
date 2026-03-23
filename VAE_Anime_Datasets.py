import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import tensorflow as tf
import urllib.request
import zipfile

from utils import setup_logging



class Datasets():
    ''' This class is used to download and display images to be for a VAE model.
    It currently assumes that the images are of anime faces. But this is
    subject to change.

    It has the following methods:  
    0. __init__: initializes the class, creates the output directory,
                and sets flags indicating things to be done.      
    1. set_data_params: sets the batch size and image size.
    2. download_data: downloads the dataset.
    3. make_train_and_validation_sets: creates training and validation datasets.
    4. display_train_data: displays a sample of the training dataset.
    5. display_validation_data: displays a sample of the validation dataset.
    6. display_sample_data: displays a sample of the dataset specified by the user.

    
    '''
    def __init__(self, output_dir="scratch_output"):
        '''Initializes the class, creates the output directory, 
        using "scratch_output" as the default,     
        and sets flags indicating things to be done.'''

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
        '''Sets the batch size and 
        the desired size of square images for the dataset.'''
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_params_set = True

    def download_data(self):
        '''Downloads the dataset if it has not been downloaded.
        Currently, the dataset is a zipped file of anime faces.'''

        # make the data directory
        Path('/tmp/anime').mkdir(exist_ok=True)
        if len(list(Path('/tmp/anime').glob('*'))) < 10:
            # download the zipped dataset to the data directory
            data_url = "https://storage.googleapis.com/learning-datasets/Resources/anime-faces.zip"
            data_file_name = "animefaces.zip"
            download_dir = '/tmp/anime/'
            zip_path = Path(download_dir) / data_file_name
            urllib.request.urlretrieve(data_url, zip_path)

            # extract the zip file
            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(download_dir)
            zip_ref.close()
        self.data_downloaded = True

    def make_train_and_validation_sets(self):
        '''Creates training and validation datasets from the downloaded images.
        The images are preprocessed and reshaped to the desired size.'''

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
            image = tf.io.decode_image(img_raw, channels=3, expand_animations=False)

            image = tf.cast(image, dtype=tf.float32)
            image = tf.image.resize(image, (self.image_size, 
                                            self.image_size))
            # Normalizes the images to the range of [0., 1.]
            # Also assumes images are in [0, 255]. This 
            # may need to change if other datasets are used.
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
        training_dataset = training_dataset.map(map_image, 
                                                num_parallel_calls=tf.data.AUTOTUNE)
        training_dataset = training_dataset.shuffle(1000).batch(self.batch_size, 
                                                                drop_remainder=True).prefetch(tf.data.AUTOTUNE)


        # load the validation image paths into tensors and create batches
        val_files = list(map(str, val_paths))
        validation_dataset = tf.data.Dataset.from_tensor_slices(val_files)
        validation_dataset = validation_dataset.map(map_image, 
                                                num_parallel_calls=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self.batch_size, 
                                                      drop_remainder=True).prefetch(tf.data.AUTOTUNE)


        # set the training and validation datasets and print the number of batches in each
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.datasets_made = True
        logging.info(f'number of batches in the training set: {len(training_dataset)}')
        logging.info(f'number of batches in the validation set: {len(validation_dataset)}')

    def display_train_data(self, size=9):
        self.display_sample_data("train", size)


    def display_validation_data(self, size=9):
        self.display_sample_data("valid", size)


    def display_sample_data(self, dataset_choice, size):
        '''Takes a desired number of samples 
        from either train or validation set,
        plots them in a grid, and saves the plot'''
        if not self.datasets_made:
            self.make_train_and_validation_sets()
        if dataset_choice[0].lower() == "t":
            dataset = self.training_dataset
            data_samp_set = "Train"
        elif dataset_choice[0].lower() == "v":
            dataset = self.validation_dataset
            data_samp_set = "Validation"
        else:
            logging.error("You must choose either the training or validation set with either a 't' or 'v' respectively.")
            return

        # Get desired number of samples from the dataset
        dataset = dataset.unbatch().take(size)

        # Set grid dimensions
        n_cols = round(np.sqrt(size))
        n_rows = size // n_cols + 1
        plt.figure(figsize=(5, 5))

        # Display the images in a grid
        i = 1
        for image in dataset:
            disp_img = np.reshape(image, (self.image_size, self.image_size, 3))
            plt.subplot(n_rows, n_cols, i)
            plt.xticks([])
            plt.yticks([])
            plt.suptitle(data_samp_set + " Images")
            # Set the title of the window
            # plt.gcf().canvas.manager.set_window_title(data_samp_set + " Images")
            plt.imshow(disp_img)
            i += 1

        # Save the plot, if output directory is specified. 
        # Default shd be "scratch_output"
        if self.output_dir:
            trgt_dir = self.output_dir / "original_images"
            trgt_dir.mkdir(parents=True, exist_ok=True)
            file_name = data_samp_set + "_sample.png"
            outfile = trgt_dir / file_name
            plt.savefig(outfile)
            logging.info(f"Sample image saved to {outfile}")
        else:
            logging.info("No specified output directory. Images of sample data are not being saved")


if __name__ == '__main__':
    # Test code
    parser = argparse.ArgumentParser(description= 'Specify output directory')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='scratch_output', help='Output directory')
    parser.add_argument("--log", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log)
    
    data = Datasets(args.output_dir)
    data.set_data_params()
    data.download_data()
    data.make_train_and_validation_sets()
    data.display_sample_data('t', 25)
    data.display_sample_data('v', 18)