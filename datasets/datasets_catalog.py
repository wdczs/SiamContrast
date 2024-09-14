# Required dataset entry keys
_PREFIX = 'image_directory'
_SOURCE_INDEX = 'image_and_label_list_file'
_MEAN = 'rgb_mean'
_STD = 'rgb_std'
_CLASSES_LIST = 'class_names_list'
_NUM_CLASSES = 'identities number'
_VIS_COLORS = 'classes_colors_for_vis'

# Available datasets
_DATASETS = {

    'Hi-UCDv2': {
        _PREFIX: '/data1/wj20/data/Hi-UCD-S/Hi-UCD-S-upload',
        _SOURCE_INDEX: {
            'train': 'datasets/Hi-UCDv2/Hi-UCDv2_train.txt',
            'val': 'datasets/Hi-UCDv2/Hi-UCDv2_val.txt',
            'test': 'datasets/Hi-UCDv2/Hi-UCDv2_test.txt'
        },
        _MEAN: [0.4359, 0.4494, 0.3951],
        _STD: [0.157, 0.1576, 0.1427],
        _NUM_CLASSES: 10,
        _VIS_COLORS: {
            'Unlabeled': [255, 255, 255],
            'Water': [0, 152, 254],
            'Grass': [202, 254, 122],
            'Buildings': [254, 0, 26],
            'Greenhouse': [228, 0, 254],
            'Road': [254, 230, 0],
            'Bridge': [254, 180, 196],
            'Others': [0, 246, 230],
            'Bareland': [174, 122, 254],
            'Woodland': [26, 254, 0],
        }
    },

}

def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()

def contains(name):
    return name in _DATASETS.keys()

def get_prefix(name):
    return _DATASETS[name][_PREFIX]

def get_source_index(name):
    return _DATASETS[name][_SOURCE_INDEX]

def get_num_classes(name):
    return _DATASETS[name][_NUM_CLASSES]

def get_mean(name):
    return _DATASETS[name][_MEAN]

def get_std(name):
    return _DATASETS[name][_STD]

def get_names_list(name):
    return _DATASETS[name][_CLASSES_LIST]

def get_vis_colors(name):
    return _DATASETS[name][_VIS_COLORS]
