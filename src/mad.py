import argparse
from collections import OrderedDict, defaultdict
import copy
import json
import os
import pickle
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

from heads import get_classification_head, get_original_classification_head
from modeling import ImageEncoder
from tv_datasets.common import get_dataloader, maybe_dictionarize
from tv_datasets.registry import get_dataset


def parse_arguments_for_merge() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default='/path/to/datasets',
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        default='/path/to/checkpoints',
        help="The root directory for the encoder checkpoint.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=8,
        choices=[8, 14, 20, 30],
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/path/to/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument(
        "--hard",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--n-samples",
        default=128,
    )
    parser.add_argument(
        "--normalize",
        action='store_true',
        default=True,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--bank-type",
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--tallmask-setting",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--cumulative",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--analysis",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args


def set_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(seed=0)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)


def compute_gaussian_distribution(features: torch.Tensor) -> Tuple[torch.Tensor]:
    if len(features.shape) != 2:
        raise ValueError(f"Input tensor must have shape (batch_size, feature_dim), but {features.shape}.")

    mean = features.mean(dim=0)
    centered_features = features - mean
    covariance = torch.mm(centered_features.T, centered_features) / (features.shape[0] - 1)
    covariance += torch.eye(covariance.shape[0], device=covariance.device) * 1e-6
    covariance_inv = torch.linalg.inv(covariance)
    
    return mean, covariance_inv


def compute_mahalanobis_distance(features: torch.Tensor, mean: torch.Tensor, covariance_inv: torch.Tensor) -> torch.Tensor:
    centered_features = features - mean
    
    left_term = torch.mm(centered_features, covariance_inv)
    # distances = torch.sqrt(torch.sum(left_term * centered_features, dim=1))
    distances = (left_term * centered_features).sum(dim=1).sqrt()
    
    return distances


def get_features(
    args: argparse.Namespace, 
    source_dataset_name: str, 
    device: str, 
    base_encoder: ImageEncoder,
) -> Dict[str, Dict[str, torch.Tensor]]:
    features_lw = defaultdict(list)

    def get_lw_features(name: str) -> Callable:
        def hook(module: nn.Module, input: Tuple[torch.Tensor]) -> None:
            feature = input[0].clone()
            if name == 'model.visual.ln_post':
                if args.normalize:
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                features_lw[name].append(feature)
            elif name.startswith('model.visual.transformer.resblocks') and len(name.split('.')) == 5:
                feature = feature.permute(1, 0, 2)  # LND -> NLD
                feature = feature[:, 0, :]  # layer-wise updated cls token?
                if args.normalize:
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                features_lw[name].append(feature)
        return hook
    
    modules = {}
    for i, m in enumerate(base_encoder.model.visual.transformer.resblocks):
        modules[f'model.visual.transformer.resblocks.{i}'] = m
    modules['model.visual.ln_post'] = base_encoder.model.visual.ln_post

    handles = []
    for name, module in modules.items():
        handle = module.register_forward_pre_hook(get_lw_features(name))
        handles.append(handle)

    dataset = get_dataset(
        source_dataset_name if args.bank_type == 'test' else f'{source_dataset_name}Val',
        base_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=(args.bank_type == 'train'), args=args)

    if isinstance(args.n_samples, float):
        n_samples = round(len(dataset.train_dataset if args.bank_type == 'train' else dataset.test_dataset) * args.n_samples)
    else:
        n_samples = args.n_samples
    print(f'n_samples: {n_samples}')

    with torch.no_grad():
        n = 0
        with tqdm(dataloader, unit=f'batch({args.batch_size})') as tepoch:
            for data in tepoch:
                tepoch.set_description(source_dataset_name)

                data = maybe_dictionarize(data)
                x = data['images'].to(device)

                if n >= n_samples:
                    break

                if n_samples - n < args.batch_size:
                    x = x[:n_samples - n, ...]

                _ = base_encoder(x)

                n += x.size(0)

        memory_bank_per_ds = {}
        for name, features in features_lw.items():
            features = torch.cat(features)
            mean, cov_inv = compute_gaussian_distribution(features)
            memory_bank_per_ds[name] = {'mean': mean, 'cov_inv': cov_inv}
    
    for handle in handles:
        handle.remove()
    
    return memory_bank_per_ds


def get_last_feature_sw(
    args: argparse.Namespace, 
    device: str,
    sample: torch.Tensor,
    base_encoder: ImageEncoder,
    ft_state_dicts: List[Dict[str, torch.Tensor]],
    memory_bank: Dict[str, Dict[str, torch.Tensor]],
    target_task_id: int,
) -> torch.Tensor:
    scores = {}
    if args.debug:
        msd_cache_for_debug = {}
    msd_cache = torch.zeros(len(DATASETS), device=device)

    def set_lw_weights(name: str) -> Callable:
        def hook(module: nn.Module, input: Tuple[torch.Tensor]) -> None:            
            if name == 'model.visual.ln_post':
                feature = input[0].clone()
                if args.normalize:
                    feature = feature / feature.norm(dim=-1, keepdim=True)

                for i, source_dataset_name in enumerate(DATASETS):
                    msd = compute_mahalanobis_distance(
                        feature, 
                        memory_bank[source_dataset_name][name]['mean'], 
                        memory_bank[source_dataset_name][name]['cov_inv'],
                    )
                    msd_cache[i] = msd

                if args.debug:
                    msd_cache_for_debug[name] = msd_cache.clone().cpu()

                score = F.softmin(msd_cache, dim=0)
                scores[name] = score
                if args.cumulative:
                    cumulative_score = torch.stack(list(scores.values()))
                    score = cumulative_score.mean(dim=0)

                for i in range(len(DATASETS)):
                    if i == 0:
                        module.weight.data = score[i] * ft_state_dicts[i][f'{name}.weight'].clone()
                        module.bias.data = score[i] * ft_state_dicts[i][f'{name}.bias'].clone()
                    else:
                        module.weight.data += score[i] * ft_state_dicts[i][f'{name}.weight'].clone()
                        module.bias.data += score[i] * ft_state_dicts[i][f'{name}.bias'].clone()
            elif name.startswith('model.visual.transformer.resblocks') and len(name.split('.')) == 5:
                feature = input[0].clone()
                feature = feature.permute(1, 0, 2)  # LND -> NLD
                feature = feature[:, 0, :]  # layer-wise updated cls token?
                if args.normalize:
                    feature = feature / feature.norm(dim=-1, keepdim=True)

                for i, source_dataset_name in enumerate(DATASETS):
                    msd = compute_mahalanobis_distance(
                        feature, 
                        memory_bank[source_dataset_name][name]['mean'], 
                        memory_bank[source_dataset_name][name]['cov_inv'],
                    )
                    msd_cache[i] = msd
                    
                if args.debug:
                    msd_cache_for_debug[name] = msd_cache.clone().cpu()

                score = F.softmin(msd_cache, dim=0)
                scores[name] = score
                if args.cumulative:
                    cumulative_score = torch.stack(list(scores.values()))
                    score = cumulative_score.mean(dim=0)

                for i in range(len(DATASETS)):
                    if i == 0:
                        module.ln_1.weight.data = score[i] * ft_state_dicts[i][f'{name}.ln_1.weight'].clone()
                        module.ln_1.bias.data = score[i] * ft_state_dicts[i][f'{name}.ln_1.bias'].clone()
                        module.attn.in_proj_weight.data = score[i] * ft_state_dicts[i][f'{name}.attn.in_proj_weight'].clone()
                        module.attn.in_proj_bias.data = score[i] * ft_state_dicts[i][f'{name}.attn.in_proj_bias'].clone()
                        module.attn.out_proj.weight.data = score[i] * ft_state_dicts[i][f'{name}.attn.out_proj.weight'].clone()
                        module.attn.out_proj.bias.data = score[i] * ft_state_dicts[i][f'{name}.attn.out_proj.bias'].clone()
                        module.ln_2.weight.data = score[i] * ft_state_dicts[i][f'{name}.ln_2.weight'].clone()
                        module.ln_2.bias.data = score[i] * ft_state_dicts[i][f'{name}.ln_2.bias'].clone()
                        module.mlp.c_fc.weight.data = score[i] * ft_state_dicts[i][f'{name}.mlp.c_fc.weight'].clone()
                        module.mlp.c_fc.bias.data = score[i] * ft_state_dicts[i][f'{name}.mlp.c_fc.bias'].clone()
                        module.mlp.c_proj.weight.data = score[i] * ft_state_dicts[i][f'{name}.mlp.c_proj.weight'].clone()
                        module.mlp.c_proj.bias.data = score[i] * ft_state_dicts[i][f'{name}.mlp.c_proj.bias'].clone()
                    else:
                        module.ln_1.weight.data += score[i] * ft_state_dicts[i][f'{name}.ln_1.weight'].clone()
                        module.ln_1.bias.data += score[i] * ft_state_dicts[i][f'{name}.ln_1.bias'].clone()
                        module.attn.in_proj_weight.data += score[i] * ft_state_dicts[i][f'{name}.attn.in_proj_weight'].clone()
                        module.attn.in_proj_bias.data += score[i] * ft_state_dicts[i][f'{name}.attn.in_proj_bias'].clone()
                        module.attn.out_proj.weight.data += score[i] * ft_state_dicts[i][f'{name}.attn.out_proj.weight'].clone()
                        module.attn.out_proj.bias.data += score[i] * ft_state_dicts[i][f'{name}.attn.out_proj.bias'].clone()
                        module.ln_2.weight.data += score[i] * ft_state_dicts[i][f'{name}.ln_2.weight'].clone()
                        module.ln_2.bias.data += score[i] * ft_state_dicts[i][f'{name}.ln_2.bias'].clone()
                        module.mlp.c_fc.weight.data += score[i] * ft_state_dicts[i][f'{name}.mlp.c_fc.weight'].clone()
                        module.mlp.c_fc.bias.data += score[i] * ft_state_dicts[i][f'{name}.mlp.c_fc.bias'].clone()
                        module.mlp.c_proj.weight.data += score[i] * ft_state_dicts[i][f'{name}.mlp.c_proj.weight'].clone()
                        module.mlp.c_proj.bias.data += score[i] * ft_state_dicts[i][f'{name}.mlp.c_proj.bias'].clone()
        return hook

    with torch.no_grad():
        base_encoder.model.visual.conv1.weight.data = ft_state_dicts[target_task_id][f'model.visual.conv1.weight'].clone()
        base_encoder.model.visual.class_embedding.data = ft_state_dicts[target_task_id][f'model.visual.class_embedding'].clone()
        base_encoder.model.visual.positional_embedding.data = ft_state_dicts[target_task_id][f'model.visual.positional_embedding'].clone()
        base_encoder.model.visual.ln_pre.weight.data = ft_state_dicts[target_task_id][f'model.visual.ln_pre.weight'].clone()
        base_encoder.model.visual.ln_pre.bias.data = ft_state_dicts[target_task_id][f'model.visual.ln_pre.bias'].clone()
    
        modules = {}
        for i, m in enumerate(base_encoder.model.visual.transformer.resblocks):
            modules[f'model.visual.transformer.resblocks.{i}'] = m
        modules['model.visual.ln_post'] = base_encoder.model.visual.ln_post

        handles = []
        for name, module in modules.items():
            set_handle = module.register_forward_pre_hook(set_lw_weights(name))
            handles.append(set_handle)
                
        last_feature = base_encoder(sample)

        if args.debug:
            print(f'msd_cache: {msd_cache_for_debug}')
            print(f'scores: {scores}')

        for handle in handles:
            handle.remove()

    if args.analysis:
        return last_feature, scores, copy.deepcopy(base_encoder.state_dict())

    return last_feature


def merge_over_two_ckpts(args: argparse.Namespace) -> None:
    device = args.device

    load_model_paths = []
    for source_dataset_name in DATASETS:
        ds_name = f'{source_dataset_name}Val' if args.tallmask_setting else source_dataset_name
        ckpt_name = 'nonlinear_finetuned.pt' if args.tallmask_setting else 'finetuned.pt'
        load_model_path = os.path.join(args.model_ckpt_dir, args.model, ds_name, ckpt_name)
        print(f'loading a checkpoint from {load_model_path}')
        load_model_paths.append(load_model_path)        

    ft_state_dicts = []
    for i, model_path in enumerate(load_model_paths):
        try:
            state_dict = torch.load(model_path)
        except RuntimeError:
            state_dict = pickle.load(open(model_path, 'rb'))

        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()

        for name in state_dict.keys():
            state_dict[name] = state_dict[name].to(device)
            
        ft_state_dicts.append(state_dict)

    base_model = ImageEncoder(args)
    base_model = base_model.to(device)
    base_model.eval()

    pre_state_dict = base_model.state_dict()

    heads = {}
    for i, d in enumerate(DATASETS):
        args.dataset_name = f'{d}Val' if args.tallmask_setting else d
        if args.tallmask_setting:
            heads[d] = get_classification_head(args, d)
        else: 
            heads[d] = get_original_classification_head(args)
        heads[d] = heads[d].to(device)
        heads[d].eval()

    num_classes = []
    for d in DATASETS:
        ds = get_dataset(
            d,
            None,
            location=args.data_location
        )
        num_classes.append(len(ds.classnames))

    if isinstance(args.n_samples, str):
        args.n_samples = int(args.n_samples) if args.n_samples.isdigit() else float(args.n_samples)

    exp_name = 'mahalanobis_lw_revised_ft_1st'
    if args.bank_type != 'test':
        exp_name += f'_{args.bank_type}'
    if args.n_samples != 128:
        exp_name += f'_ns{args.n_samples}'
    if args.batch_size != 128:
        exp_name += f'_bs{args.batch_size}'
    if args.hard:
        exp_name += f'_hard'
    if args.cumulative:
        exp_name += f'_cumulative'

    save_params_dir = os.path.join('params', exp_name, args.model)
    os.makedirs(save_params_dir, exist_ok=True)
    fn_bank = 'memory_bank'
    if args.tallmask_setting:
        fn_bank += '_tallmask'
    save_params_path = os.path.join(save_params_dir, f'{fn_bank}.pt')
    if not os.path.exists(save_params_path):
        memory_bank = {}
        for i, source_dataset_name in enumerate(DATASETS):
            print('Generating memory bank of', source_dataset_name)
            base_model.load_state_dict(ft_state_dicts[i])
            memory_bank[source_dataset_name] = get_features(
                args, source_dataset_name, device, base_model
            )
        torch.save(memory_bank, save_params_path)
    else:
        memory_bank = torch.load(save_params_path, map_location=device)
        print(f'Cached memory bank has been uploaded.')

    if args.analysis:
        module_names = []
        for i in range(len(base_model.model.visual.transformer.resblocks)):
            module_names.append(f'model.visual.transformer.resblocks.{i}')
        module_names.append('model.visual.ln_post')

        temp_model = ImageEncoder(args)
        temp_model = temp_model.to(device)
        temp_model.eval()

        analysis_data = {}

    for target_dataset_name in DATASETS:
        print('Evaluating on', target_dataset_name)

        dataset_to_merge = '_'.join(DATASETS)
        #!FIXME: f'{dataset_to_merge}__{target_dataset_name}' -> f'{target_dataset_name}__{dataset_to_merge}'
        ckpt_name = 'nonlinear_finetuned.pt' if args.tallmask_setting else 'finetuned.pt'
        save_result_dir = os.path.join(ckpt_name, args.model, f'{dataset_to_merge}__{target_dataset_name}', exp_name)
        os.makedirs(save_result_dir, exist_ok=True)
        save_result_path = os.path.join(save_result_dir, f'{args.model}.json')
        
        if args.analysis:
            save_analysis_dir = os.path.join(ckpt_name, args.model, exp_name)
            save_analysis_path = os.path.join(save_analysis_dir, 'iccv_analysis.pt')
            if os.path.exists(save_result_path) and os.path.exists(save_analysis_path):
                print(f'{save_result_path} and {save_analysis_path} already exist')
                continue
        else:
            if os.path.exists(save_result_path):
                print(f'{save_result_path} already exists')
                continue
        
        dataset = get_dataset(
            target_dataset_name,
            base_model.val_preprocess,
            location=args.data_location,
            batch_size=1,
        )
        dataloader = get_dataloader(dataset, is_train=False, args=args)

        with torch.no_grad():
            if args.analysis:
                avg_scores = {}
                for k in module_names:
                    avg_scores[k] = torch.zeros(len(DATASETS), device=device)

                avg_state_dict = OrderedDict()
                for k, v in temp_model.state_dict().items():
                    avg_state_dict[k] = torch.zeros_like(v)

                data_per_task = {
                    'scores': avg_scores,
                    'state_dict': avg_state_dict
                }

            correct = 0. 
            n = 0
            with tqdm(dataloader, unit=f'batch(1)') as tepoch:
                for data in tepoch:
                    tepoch.set_description(target_dataset_name)

                    data = maybe_dictionarize(data)
                    x = data['images'].to(device)
                    y = data['labels'].to(device)

                    target_task_id = DATASETS.index(target_dataset_name)
                    base_model.load_state_dict(pre_state_dict)

                    if args.analysis:
                        last_feature, scores, merged_state_dict = get_last_feature_sw(
                            args, 
                            device, 
                            x, 
                            base_model, 
                            ft_state_dicts, 
                            memory_bank,
                            target_task_id
                        )
                    else:
                        last_feature = get_last_feature_sw(
                            args, 
                            device, 
                            x, 
                            base_model, 
                            ft_state_dicts, 
                            memory_bank,
                            target_task_id
                        )

                    if args.normalize:
                        last_feature = last_feature / last_feature.norm(dim=-1, keepdim=True)
                    logits = heads[target_dataset_name](last_feature)
                    pred = logits.argmax(dim=1, keepdim=True)

                    correct += pred.eq(y.view_as(pred)).sum().item()
                    n += y.size(0)

                    tepoch.set_postfix(acc=correct / n)

                    if args.analysis:
                        for k, v in scores.items():
                            data_per_task['scores'][k] += v
                        for k, v in merged_state_dict.items():
                            data_per_task['state_dict'][k] += v

                if args.analysis:
                    for k in data_per_task['scores'].keys():
                        data_per_task['scores'][k] /= n
                    for k in data_per_task['state_dict'].keys():
                        data_per_task['state_dict'][k] /= n

                    analysis_data[target_dataset_name] = data_per_task

            top1 = correct / n

        print(f"{target_dataset_name} Top-1 accuracy: {top1:.4f}")

        results = {'top1': top1}
        with open(save_result_path, 'w') as f:
            json.dump(results, f)
        print(f'Results saved to {save_result_path}.')

        if args.analysis:
            torch.save(analysis_data, save_analysis_path)


if __name__ == '__main__':
    args = parse_arguments_for_merge()
    if args.num_tasks == 8:
        DATASETS = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    elif args.num_tasks == 14:
        DATASETS = [
            'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN', 
            'CIFAR100', 'STL10', 'Flowers102', 'OxfordIIITPet', 'PCAM', 'FER2013',
        ]
    elif args.num_tasks == 20:
        DATASETS = [
            'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN', 
            'CIFAR100', 'STL10', 'Flowers102', 'OxfordIIITPet', 'PCAM', 'FER2013',
            'EMNIST', 'CIFAR10', 'Food101', 'FashionMNIST', 'RenderedSST2', 'KMNIST'
        ]
    elif args.num_task == 30:
        DATASETS = [
            'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN',
            'CIFAR100', 'STL10', 'Flowers102', 'OxfordIIITPet', 'PCAM', 'FER2013',
            'EMNIST', 'CIFAR10', 'Food101', 'FashionMNIST', 'RenderedSST2', 'KMNIST',
            "Vegetables", "Kvasir", "IntelImages", "Weather", "CatsDogs",
            "MangoLeafBD", "Beans", "Landscape", "Garbage", "Fruits360"
        ]
    else:
        raise NotImplementedError
    merge_over_two_ckpts(args)