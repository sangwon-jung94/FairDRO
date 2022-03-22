import os
import os.path
from PIL import Image
import numpy as np
import pickle

from data_handler import GenericDataset, SSLDataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import transforms

def rgb_to_grayscale(img):
    """Convert image to gray scale"""
    pil_gray_img = img.convert('L')
    np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
    np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

    return np_gray_img

class CIFAR_100S(SSLDataset):
    #normalize
    name = 'cifar100s'
    train_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()]
            )
    test_transform = transforms.Compose(
             [transforms.ToTensor()]
            )

    def __init__(self, skewed_ratio=0.8, **kwargs):
        transform = self.train_transform if kwargs['split'] == 'train' else self.test_transform

        SSLDataset.__init__(self, transform=transform, **kwargs)

#         super(CIFAR_100S, self).__init__(root, transform=transform, target_transform=target_transform)

        self.n_classes = 100
        self.n_groups = 2

        imgs, labels, colors, data_count = self._make_skewed(self.split, self.seed, skewed_ratio, self.n_classes)
        
        self.dataset = {}
        self.dataset['image'] = np.array(imgs)
        self.dataset['label'] = np.array(labels)
        self.dataset['color'] = np.array(colors)
        
        #add by sw
        self.features = self._combine(self.dataset)

        self.n_data, self.idxs_per_group = self._data_count(self.features, self.n_groups, self.n_classes)

        self.weights = self._make_weights()    
        
    def _combine(self, dataset):
        image = self.dataset['image']
        label = self.dataset['label']
        color = self.dataset['color']
        features = []
        for g, l, img in zip(color, label, image):
            features.append([g, l, img])
        return np.array(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        s, l, image = self.features[index]

        if self.transform:
            image = self.transform(image)

        if 'group' in self.version:
            return image, 1, np.float32(l), np.int64(s), (index)
        else :
            return image, 1, np.float32(s), np.int64(l), (index)


    def _make_skewed(self, split='train', seed=0, skewed_ratio=1., n_classes=10):

        train = False if split =='test' else True
        cifardata = CIFAR100('./data', train=train, shuffle=True, seed=seed, download=True)

        n_data = 50000 if split =='train' else 20000

        imgs = np.zeros((n_data, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros(n_data)
        colors = np.zeros(n_data)
        data_count = np.zeros((2, n_classes), dtype=int)

        n_total_train_data = int((50000 // n_classes))
        n_skewed_train_data = int((50000 * skewed_ratio) // n_classes)

        
        for i, data in enumerate(cifardata):
            img, target = data

            if split == 'test':
                imgs[i] = rgb_to_grayscale(img)
                imgs[i+10000] = np.array(img)
                labels[i] = target
                labels[i+10000] = target
                colors[i] = 0
                colors[i+10000] = 1
                data_count[0, target] += 1
                data_count[1, target] += 1
            else:
                if target < (n_classes / 2):
                    if data_count[0, target] < (n_skewed_train_data):
                        imgs[i] = rgb_to_grayscale(img)
                        colors[i] = 0
                        data_count[0, target] += 1
                    else:
                        imgs[i] = np.array(img)
                        colors[i] = 1
                        data_count[1, target] += 1
                    labels[i] = target
                else:
                    if data_count[0, target] < (n_total_train_data - n_skewed_train_data):
                        imgs[i] = rgb_to_grayscale(img)
                        colors[i] = 0
                        data_count[0, target] += 1
                    else:
                        imgs[i] = np.array(img)
                        colors[i] = 1
                        data_count[1, target] += 1
                    labels[i] = target

#         print('<# of Skewed data>')
#         print(data_count)
        return imgs, labels.astype(int), colors.astype(int), data_count


class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, shuffle=False, seed=0):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if shuffle:
            np.random.seed(seed)
            idx = np.arange(len(self.data), dtype=np.int64)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


