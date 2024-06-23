import argparse
import os

import torch


def average_ckpts(model_folders, saved_folder):
    """
    Args:
        model_folders: list of folders containing checkpoints
        saved_folder: folder to save the averaged checkpoint
    """
    model_list = []
    for folder in model_folders:
        print(f'loading model from {folder}')
        model_list.append(
            torch.load(os.path.join(folder, 'model_tp0_pp0.pt'),
                       map_location='cpu'))

    averaged_model = model_list[0]
    for key in averaged_model.keys():
        weights = []
        for i in range(len(model_list)):
            weights.append(model_list[i][key])
        avg_weight = torch.mean(torch.stack(weights), dim=0)
        assert avg_weight.shape == averaged_model[key].shape
        print(f'averaging {key} with shape {avg_weight.shape}')
        # print mean delta
        for i in range(len(model_list)):
            delta = torch.mean(avg_weight) - torch.mean(model_list[i][key])
            print(f'delta {i} of {key}: {delta}')
        print(f'avg weight of {key}: {torch.mean(avg_weight)}')
        print(f'model 0 weight of {key}: {torch.mean(model_list[0][key])}')
        print(f'model 1 weight of {key}: {torch.mean(model_list[1][key])}')
        print(f'model 2 weight of {key}: {torch.mean(model_list[2][key])}')
        print(f'model 3 weight of {key}: {torch.mean(model_list[3][key])}')

        averaged_model[key] = avg_weight
    os.makedirs(saved_folder, exist_ok=True)
    torch.save(averaged_model, os.path.join(saved_folder, 'model_tp0_pp0.pt'))


def momentum_averaged_ckpts(model_folders, saved_folder, momentum=0.9):
    """
    Args:
        model_folders: list of folders containing checkpoints
        saved_folder: folder to save the averaged checkpoint
        momentum: momentum for averaging
    """
    model_list = []
    for folder in model_folders:
        print(f'loading model from {folder}')
        model_list.append(
            torch.load(os.path.join(folder, 'model_tp0_pp0.pt'),
                       map_location='cpu'))

    averaged_model = model_list[0]
    for key in averaged_model.keys():
        for i in range(1, len(model_list)):
            print(
                f'difference between averaged model and model {i} for {key}: {torch.mean(averaged_model[key]) - torch.mean(model_list[i][key])}'  # noqa: E501
            )
            averaged_model[key] = averaged_model[key] * momentum + model_list[
                i][key] * (1 - momentum)
    os.makedirs(saved_folder, exist_ok=True)
    torch.save(averaged_model, os.path.join(saved_folder, 'model_tp0_pp0.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folders',
                        nargs='+',
                        help='list of folders containing checkpoints')
    parser.add_argument('--saved_folder',
                        help='folder to save the averaged checkpoint')
    parser.add_argument('--momentum',
                        type=float,
                        default=-1,
                        help='momentum for averaging')
    args = parser.parse_args()

    if args.momentum > 0:
        assert args.momentum < 1
        momentum_averaged_ckpts(args.model_folders, args.saved_folder,
                                args.momentum)
    else:
        average_ckpts(args.model_folders, args.saved_folder)
