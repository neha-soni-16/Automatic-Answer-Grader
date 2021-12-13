import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataHandler.loadData import LoadData, Batch
from model import Model, DecoderType
from dataHandler.preprocess import Preprocess

class FilePaths:
    fn_char_list = './charList.txt'
    fn_summary = './summary.json'
    fn_corpus = './dataset/corpus.txt'

def get_img_height() -> int:
    return 32

def get_img_size() -> Tuple[int, int]:
    return 128, get_img_height()

def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)

def train(model: Model,
          loader: LoadData,
          early_stopping: int = 25) -> None:

    epoch = 0  
    summary_word_accuracies = []
    summary_char_error_rates = []
    preprocess = Preprocess(get_img_size(), augmentData=True)
    best_char_error_rate = float('inf')  
    no_improvement_since = 0  

    while True:

        if epoch == 32 :
            break

        epoch += 1
        print('Epoch:', epoch)

        print('Train NN')

        loader.trainingDataSet()
        while loader.isNext():
            iter_info = loader.getInfo()
            batch = loader.getNextDataImage()
            batch = preprocess.processBatch(batch)
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        char_error_rate, word_accuracy = validate(model, loader)

        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, saving model....')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: LoadData) -> Tuple[float, float]:

    print('Starting validation....')
    loader.validationDataSet()
    preprocess = Preprocess(get_img_size())
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.isNext():
        iter_info = loader.getInfo()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.getNextDataImage()
        batch = preprocess.processBatch(batch)
        recognized, _ = model.inferBatch(batch)

        print('Actual Word -> Recognized word')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('correctly recognized' if dist == 0 else '[Error(using edit distance):%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> None:
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocess = Preprocess(get_img_size(), dynamic_width=True, padding=16)
    img = preprocess.processImage(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.inferBatch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train'], default='infer')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=100)
    parser.add_argument('--data_dir', help='Data Directory', type=Path, required=False)
    parser.add_argument('--img_file', help='Image Path', type=Path, default='./plis.jpeg')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    args = parser.parse_args()

    decoder_type = DecoderType.BestPath

    if args.mode in ['train']:
        loader = LoadData(args.data_dir, args.batch_size)
        char_list = loader.char_list

        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        if args.mode == 'train':
            model = Model(char_list, decoder_type)
            train(model, loader, early_stopping=args.early_stopping)

    elif args.mode == 'infer':
        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True)
        infer(model, args.img_file)


if __name__ == '__main__':
    main()
