import cv2
import numpy as np
import random

from path import Path
from typing import Tuple
from collections import namedtuple

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batchSize')

class LoadData:
    def __init__(self,
                 dataDir: Path,
                 batchSize: int,
                 splitData: float = 0.95) -> None:

        assert dataDir.exists()

        self.augmentData = False
        self.index = 0
        self.batchSize = batchSize
        self.images = []

        f = open(dataDir / 'gt/words.txt')
        
        charSet = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset

        for line in f:
            if not line or line[0] == '#':
                continue

            splitLine = line.strip().split(' ')
            assert len(splitLine) >= 9

            file_name_split = splitLine[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = splitLine[0] + '.png'
            file_name = dataDir / 'img' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if splitLine[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            gt_text = ' '.join(splitLine[8:])
            charSet = charSet.union(set(list(gt_text)))

            self.images.append(Sample(gt_text, file_name))

        splitIndex = int(splitData * len(self.images))
        self.train_samples = self.images[:splitIndex]
        self.validation_samples = self.images[splitIndex:]

        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        self.trainingDataSet()

        self.char_list = sorted(list(charSet))

    def trainingDataSet(self) -> None:  
        self.augmentData = True
        self.index = 0
        random.shuffle(self.train_samples)
        self.images = self.train_samples
        self.curr_set = 'train'

    def validationDataSet(self) -> None:
        self.augmentData = False
        self.index = 0
        self.images = self.validation_samples
        self.curr_set = 'val'

    def getInfo(self) -> Tuple[int, int]:
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.images) / self.batchSize)) 
        else:
            num_batches = int(np.ceil(len(self.images) / self.batchSize)) 
        curr_batch = self.index// self.batchSize + 1
        return curr_batch, num_batches

    def isNext(self) -> bool:
        if self.curr_set == 'train':
            return self.index + self.batchSize <= len(self.images)  
        else:
            return self.index < len(self.images)  

    def getDataImage(self, i: int) -> np.ndarray:
        img = cv2.imread(self.images[i].file_path, cv2.IMREAD_GRAYSCALE)
        return img

    def getNextDataImage(self) -> Batch:
        batch_range = range(self.index, min(self.index + self.batchSize, len(self.images)))

        imgs = [self.getDataImage(i) for i in batch_range]
        gt_texts = [self.images[i].gt_text for i in batch_range]

        self.index += self.batchSize
        return Batch(imgs, gt_texts, len(imgs))