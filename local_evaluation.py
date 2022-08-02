"""
=========
IMPORTANT
=========

THE CONTENTS OF THIS FILE WILL BE REPLACED DURING EVALUATION.
ANY CHANGES MADE TO THIS FILE WILL BE DROPPED DURING EVALUATION.

THIS FILE IS PROVIDED ONLY FOR YOUR CONVINIENCE TO TEST THE CODE LOCALLY.

"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from evaluation_utils.clear_evaluator import CLEAREvaluator
from evaluation_utils.base_predictor import BaseCLEARPredictor
from evaluation_utils.CLEAR10 import CLEAR10IMG
from torch.utils.data import DataLoader
from evaluation_setup import load_models, data_transform


class CLEARPredictor(BaseCLEARPredictor):
    def __init__(self, args, bucket_num=10, use_gpu=True):
        self.bucket_num = bucket_num
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.use_gpu = use_gpu
        self.models = [None] * self.bucket_num

    def prediction_setup(self, models_path):
        print("[DEBUG] Loading models...")
        self.models = load_models(models_path, num_classes=self.num_classes)
        assert(len(self.models)) == self.bucket_num

    def prediction(self, image_file_path: str):
        # Data Loader
        print("[DEBUG] Loading and Transforming Test Data...")
        transform = data_transform()
        # transform = data_fivecrop_transform()
        test_data = []
        for i in range(self.bucket_num):
            test_data.append(CLEAR10IMG(image_file_path, i, form="all", 
                                        debug=False, transform=transform))
        
        test_loaders = []
        for i in range(len(test_data)):
            test_loaders.append(DataLoader(test_data[i], batch_size=self.batch_size, 
                                           shuffle=False, num_workers=self.num_workers))

        # Inference
        print("[DEBUG] Inference...")
        R = np.zeros((self.bucket_num,)*2)  # accuracy matrix
        for i, model in enumerate(self.models):
            for j, test_loader in enumerate(test_loaders):
                print('Evaluate timestamp %d model on bucket %d' % (i, j))
                if self.use_gpu: 
                    model.to('cuda')
                R[i, j] = self.test(model, test_loader)
            del model

        return R

    def test(self, model, test_loader):
        total_test_acc = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                if self.use_gpu:
                    xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
                y_pred = model(xb)
                _, preds = torch.max(y_pred.data, 1)
                total_test_acc += torch.sum(preds == yb.data)
        avg_test_acc = total_test_acc / len(test_loader.dataset)
        print('Test Accuracy: {:.2f}'.format(avg_test_acc))
        
        return avg_test_acc.cpu().numpy().squeeze()


def main():
    parser = argparse.ArgumentParser(description="Local Evaluation Test")
    parser.add_argument(
        "--dataset-path",
        required=True,
        dest="dataset_path",
        help="Path to dir containing extracted dataset",
    )
    parser.add_argument(
        '--resume', 
        default='', 
        type=str, 
        metavar='PATH',
        help='Resume full model from checkpoint (default: none)'
    )
    parser.add_argument(
        '--save_path', 
        default='.', 
        type=str, 
        metavar='PATH',
        help='save path for prediction files and matrix images (default: none)'
    )
    parser.add_argument(
        '-b', 
        '--batch_size', 
        type=int, 
        default=128, 
        metavar='N',
        help='Input batch size for training (default: 128)'
    )
    parser.add_argument(
        '-j', 
        '--num_workers', 
        type=int, 
        default=4, 
        metavar='N',
        help='how many training processes to use (default: 4)'
    )
    parser.add_argument(
        '--num_classes', 
        default=11, 
        type=int, 
        metavar='N',
        help='number of label classes (default: 11)'
    )
    
    args = parser.parse_args()
    predictions_file_path = os.path.join(args.save_path, "predictions.txt")
    
    evaluator = CLEAREvaluator(
        test_data_path=args.dataset_path,
        models_path=args.resume,
        predictions_file_path=predictions_file_path,
        save_file_path=args.save_path,
        predictor=CLEARPredictor(args),
    )
    evaluator.evaluation()  # Make predictions and calculate accuracy matrix

    # Compute four metrics and plot accuracy matrix at accuracy_matrix.png by default
    scores = evaluator.scoring(prediction_file_path=predictions_file_path)
    print("Weighted Average Score: %.3f" % scores['score'])
    print("Next-Domain Accuracy: %.3f" % scores['score_secondary'])
    print("In-Domain Accuracy: %.3f" % scores['meta']['in_domain_accuracy'])
    print("Backward Transfer Accuracy: %.3f" % scores['meta']['backward_transfer'])
    print("Forward Transfer Accuracy: %.3f" % scores['meta']['forward_transfer'])


if __name__ == "__main__":
    main()
