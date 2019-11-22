
import model_train as mt

MP = {}

# Directory of train/dev/test image and crater hdf5 files.
MP['dir'] = 'catalogues/'

# Image width/height, assuming square images.
MP['dim'] = 256

# Batch size
MP['bs'] =3

# Number of training epochs.
MP['epochs'] = 5

# to be a multiple of batch size.
MP['n_train'] = 30000
MP['n_dev'] = 3000
MP['n_test'] = 3000

# Save model (binary flag) and directory.
MP['save_models'] = 1
MP['save_dir'] = 'models/model_deepResUnet_5000_5.h5'

# Model Parameters (to potentially iterate over, keep in lists).
MP['N_runs'] = 1                # Number of runs
MP['filter_length'] = [112]       # Filter length
MP['lr'] = [0.0001]             # Learning rate
MP['n_filters'] = [112]        # Number of filters
MP['init'] = ['he_normal']      # Weight initialization
MP['lambda'] = [1e-6]           # Weight regularization
MP['dropout'] = [0.15]          # Dropout fraction

if __name__ == '__main__':
    mt.get_models(MP)
