"""
Train activity recognition models on CLIP/DINOv3/VideoMAE features
and multimodal skeleton+RGB fusion.

Matches keypoints_train pipeline exactly:
  - Same seed (111), same cudnn settings, same epoch-based sampling
  - Same optimizer (Adam), scheduler (StepLR)
  - Same splits (sub_train/sub_test), same epochs (50)
  - Same logging (file + console), same evaluation/reporting

Modes:
  keypoint:      Keypoints only → standalone 1D-CNN classifier
  feature:       Pre-extracted CLIP/DINOv3/VideoMAE features → temporal head
  fusion:        Pre-extracted features + keypoints → fusion model

Usage (config file — preferred):
  python train.py --config configs/normal/dinov3_meanpool.py
  python train.py --config configs/normal/dinov3_cnn1d_fusion.py

Usage (CLI args — for quick experiments or overrides):
  python train.py --config configs/normal/dinov3_meanpool.py --batch_size 64
  python train.py --mode feature --model meanpool --ann_file ... --feature_dir ...
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from dataset import build_dataloaders
from models import build_model


def load_config(config_path):
    """Load a Python config file (like keypoints_train ConfigParser)."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Class labels (0-indexed, after subtracting 1 from pkl labels)
LABEL_NAMES = {
    'normal': [
        'stand', 'stand_activity', 'walk', 'sit', 'sit_activity',
        'sitting_down', 'getting_up', 'bend', 'unstable', 'fall',
        'lie_down', 'lying_down', 'reach', 'run', 'jump',
    ],
    'wheelchair': [
        'sit', 'propel', 'pick_place', 'sit_activity', 'bend',
        'getting_up', 'exercise', 'sitting_down', 'prepare_transfer',
        'transfer', 'fall', 'lie_down', 'lying_down', 'get_propelled',
        'stand',
    ],
}

# Mode → what keys the batch contains
MODE_KEYS = {
    'keypoint': ['keypoints'],
    'feature': ['features'],
    'fusion': ['features', 'keypoints'],
}


def print_and_log(msg, log=True):
    """Print to console and optionally log to file."""
    print(msg)
    if log:
        logging.info(msg)


def seed_everything(seed=111):
    """Identical to keypoints_train seed_everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _forward(model, batch, mode, device):
    """Forward pass dispatched by mode."""
    if mode == 'keypoint':
        return model(batch['keypoints'].to(device))
    elif mode == 'feature':
        return model(batch['features'].to(device))
    elif mode == 'fusion':
        return model(batch['features'].to(device),
                     batch['keypoints'].to(device))


def train_one_epoch(model, loader, optimizer, criterion, device, mode,
                    save_logs=True, iter_every=50,
                    qmf_histories=None, qmf_lambda=0.1,
                    ogm_alpha=0.0):
    """Train for one epoch.

    Args:
        ogm_alpha: OGM-GE modulation strength. When > 0, applies on-the-fly
            gradient modulation (Peng et al., CVPR 2022). Only the fused CE
            loss is backpropagated; per-modality logits are used only for
            computing the discrepancy ratio.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_iters = len(loader)

    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    # Per-sample CE for QMF history updates (no reduction)
    if qmf_histories is not None:
        from qmf_utils import rank_loss
        criterion_none = nn.CrossEntropyLoss(reduction='none')

    for i, batch in enumerate(loader):
        labels = batch['label'].to(device).view(-1)

        optimizer.zero_grad()
        logits = _forward(model, batch, mode, device)

        if isinstance(logits, dict) and ogm_alpha > 0 and 'vis_conf' not in logits:
            # OGM-GE: only fused CE is backpropagated (matching original paper)
            loss = criterion(logits['fused'], labels.long())
            loss.backward()

            # Compute per-modality softmax scores for correct class
            with torch.no_grad():
                score_vis = softmax(logits['vis'])[
                    torch.arange(labels.size(0)), labels].sum()
                score_kp = softmax(logits['kp'])[
                    torch.arange(labels.size(0)), labels].sum()

                ratio_vis = score_vis / (score_kp + 1e-8)
                ratio_kp = 1.0 / (ratio_vis + 1e-8)

                if ratio_vis > 1:
                    coeff_vis = 1 - tanh(ogm_alpha * relu(ratio_vis))
                    coeff_kp = 1.0
                else:
                    coeff_kp = 1 - tanh(ogm_alpha * relu(ratio_kp))
                    coeff_vis = 1.0

            # Modulate encoder weight gradients only (not classifier,
            # not biases/normalization). Original paper filters for 4D
            # conv params (len(grad.size())==4). Our 1D-CNN + linear
            # architecture has 3D conv weights and 2D linear weights,
            # so we use dim >= 2 to match the spirit: modulate learned
            # feature transformations, skip 1D biases and BN/LN params.
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if 'classifier' in name:
                    continue
                if param.grad.dim() < 2:
                    continue
                if 'feat_' in name:
                    param.grad = (param.grad * coeff_vis +
                                  torch.zeros_like(param.grad).normal_(
                                      0, param.grad.std().item() + 1e-8))
                elif 'kp_' in name:
                    param.grad = (param.grad * coeff_kp +
                                  torch.zeros_like(param.grad).normal_(
                                      0, param.grad.std().item() + 1e-8))

        elif isinstance(logits, dict) and 'vis_conf' in logits:
            # QMF loss: CE on all three outputs + rank loss
            loss_fused = criterion(logits['fused'], labels.long())
            loss_vis = criterion(logits['vis'], labels.long())
            loss_kp = criterion(logits['kp'], labels.long())
            clf_loss = loss_fused + loss_vis + loss_kp

            # Rank loss to calibrate confidence scores
            idx = batch['idx'].to(device)
            vis_conf = logits['vis_conf']
            kp_conf = logits['kp_conf']

            loss_rank_vis = rank_loss(vis_conf, idx, qmf_histories['vis'])
            loss_rank_kp = rank_loss(kp_conf, idx, qmf_histories['kp'])

            loss = clf_loss + qmf_lambda * (loss_rank_vis + loss_rank_kp)

            # Update histories with per-sample losses
            with torch.no_grad():
                vis_loss_per_sample = criterion_none(
                    logits['vis'], labels.long()).detach()
                kp_loss_per_sample = criterion_none(
                    logits['kp'], labels.long()).detach()
                qmf_histories['vis'].correctness_update(
                    idx, vis_loss_per_sample, vis_conf.detach())
                qmf_histories['kp'].correctness_update(
                    idx, kp_loss_per_sample, kp_conf.detach())
            loss.backward()

        elif isinstance(logits, dict) and 'contrastive_loss' in logits:
            # MMCL: CE(keypoint-only) + λ·contrastive (Liu et al., ACM MM 2024)
            # Paper Table 6: λ₁=0.2 for L_C. L_R omitted (requires BLIP LLM).
            lambda_c = 0.2
            loss_cls = criterion(logits['fused'], labels.long())
            loss_c = logits['contrastive_loss']
            loss = loss_cls + lambda_c * loss_c
            loss.backward()
            if i % iter_every == 0 or i == total_iters - 1:
                print_and_log(
                    f"  L_cls: {loss_cls.item():.4f}  "
                    f"L_C: {loss_c.item():.4f}  "
                    f"(λ·L_C: {(lambda_c * loss_c).item():.4f})",
                    save_logs)

        elif isinstance(logits, dict) and 'pose' in logits:
            # Dual-head loss: α * CE(fused) + (1-α) * CE(pose)
            alpha = 0.7
            loss = (alpha * criterion(logits['fused'], labels.long()) +
                    (1 - alpha) * criterion(logits['pose'], labels.long()))
            loss.backward()
        elif isinstance(logits, dict):
            # OGM-GE with modulation disabled (past epoch window) or
            # other dict-returning models — fused CE only
            loss = criterion(logits['fused'], labels.long())
            loss.backward()
        else:
            loss = criterion(logits, labels.long())
            loss.backward()

        optimizer.step()

        if i % iter_every == 0 or i == total_iters - 1:
            print_and_log(
                f"Iter {i+1}/{total_iters} done. "
                f"Iter loss: {loss.item():.4f}", save_logs)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch['label'].to(device).view(-1)
        logits = _forward(model, batch, mode, device)
        if isinstance(logits, dict):
            logits = logits['fused']
        _, predicted = torch.max(logits.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    return acc, macro_f1, weighted_f1, all_preds, all_labels


def save_confusion_matrix_and_classification_report(
        true_labels, predicted_labels, label_names, save_dir):
    """Save confusion matrix (PNG + CSV) and classification report.
    Matches keypoints_train/utils/test_utils.py.
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'),
                bbox_inches='tight')
    plt.close()

    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        os.path.join(save_dir, 'confusion_matrix.csv'))

    report = classification_report(
        true_labels, predicted_labels, digits=3,
        target_names=label_names, output_dict=True)

    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    mean_class_accuracy = np.nanmean(class_accuracies)
    report['mean_class_accuracy'] = mean_class_accuracy

    report_str = ''
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            metrics_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items())
            report_str += f'{label} {metrics_str}\n'
        else:
            report_str += f'{label}: {metrics:.3f}\n'

    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    report_pretty = classification_report(
        true_labels, predicted_labels, digits=4, target_names=label_names)

    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(report_pretty)

    return report, report_pretty, mean_class_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train activity recognition models')

    # Config file (preferred — all settings in one place)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to Python config file (e.g. configs/dinov3_meanpool.py)')

    # CLI overrides (used when no config, or to override config values)
    parser.add_argument('--mode', type=str, default=None,
                        choices=['keypoint', 'feature', 'fusion'])
    parser.add_argument('--ann_file', type=str, default=None)
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Directory with .npy features (feature/fusion modes)')
    parser.add_argument('--train_split', type=str, default=None)
    parser.add_argument('--test_split', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default=None,
                        choices=['normal', 'wheelchair'])
    parser.add_argument('--preprocess', type=str, default=None,
                        choices=['sequential', 'skip'])
    parser.add_argument('--model_frames', type=int, default=None)
    parser.add_argument('--model', type=str, default=None,
                        choices=['cnn1d', 'cnn1d_tiny', 'meanpool',
                                 'transformer', 'fusion', 'fusion_tiny',
                                 'robust_fusion', 'robust_fusion_tiny',
                                 'qmf_fusion', 'ogm_fusion',
                                 'mmcl_fusion'])
    parser.add_argument('--feat_dim', type=int, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--eta_min', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print_every', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--class_weighted', action='store_true', default=None,
                        help='Use inverse-frequency class weights in loss')

    cli_args = parser.parse_args()

    # Build effective args: config file values as base, CLI overrides on top
    if cli_args.config is not None:
        cfg = load_config(cli_args.config)
        # Extract flat args from config dicts
        args = argparse.Namespace(
            config=cli_args.config,
            # Dataset
            mode=cfg.dataset_cfg['mode'],
            ann_file=cfg.dataset_cfg['ann_file'],
            feature_dir=cfg.dataset_cfg.get('feature_dir'),
            train_split=cfg.dataset_cfg['splits']['train'],
            test_split=cfg.dataset_cfg['splits']['test'],
            dataset_type=cfg.dataset_cfg['dataset_type'],
            preprocess=cfg.dataset_cfg['preprocess'],
            model_frames=cfg.dataset_cfg['model_frames'],
            feat_dim=cfg.dataset_cfg.get('feat_dim', 768),
            keypoint_field=cfg.dataset_cfg.get('keypoint_field', 'keypoint'),
            # Model
            model=cfg.model_cfg['type'],
            num_classes=cfg.model_cfg['args']['num_classes'],
            # Training
            epochs=cfg.train_cfg['epochs'],
            batch_size=cfg.train_cfg['train_settings']['batch_size'],
            lr=cfg.optimizer_cfg['settings']['lr'],
            weight_decay=cfg.optimizer_cfg['settings'].get('weight_decay', 5e-4),
            eta_min=cfg.train_cfg['scheduler'].get('eta_min', 1e-6),
            scheduler_type=cfg.train_cfg['scheduler'].get('type', 'cosine'),
            scheduler_step_size=cfg.train_cfg['scheduler'].get('step_size', 30),
            scheduler_gamma=cfg.train_cfg['scheduler'].get('gamma', 0.1),
            scheduler_milestones=cfg.train_cfg['scheduler'].get('milestones', [90, 100]),
            optimizer_type=cfg.optimizer_cfg.get('type', 'adamw'),
            momentum=cfg.optimizer_cfg['settings'].get('momentum', 0.9),
            nesterov=cfg.optimizer_cfg['settings'].get('nesterov', False),
            warmup_epochs=cfg.train_cfg.get('warmup_epochs', 0),
            num_workers=cfg.train_cfg['train_settings']['num_workers'],
            device=cli_args.device,
            output_dir=cfg.train_cfg['output_dir'],
            seed=cfg.dataset_cfg['seed'],
            print_every=cfg.train_cfg['print_every'],
            save_every=cfg.train_cfg['save_ckpt_every'],
            # Eval batch size (may differ from train)
            eval_batch_size=cfg.train_cfg['test_settings']['batch_size'],
            eval_num_workers=cfg.train_cfg['test_settings']['num_workers'],
            class_weighted=cfg.train_cfg.get('class_weighted', False),
            center_frame_only=cfg.dataset_cfg.get('center_frame_only', False),
        )
        # CLI overrides: any explicitly provided CLI arg takes precedence
        for key in vars(cli_args):
            cli_val = getattr(cli_args, key)
            if cli_val is not None and key != 'config':
                setattr(args, key, cli_val)
    else:
        # No config file — use CLI args with defaults
        args = argparse.Namespace(
            config=None,
            mode=cli_args.mode or 'feature',
            ann_file=cli_args.ann_file,
            feature_dir=cli_args.feature_dir,
            train_split=cli_args.train_split or 'sub_train',
            test_split=cli_args.test_split or 'sub_test',
            dataset_type=cli_args.dataset_type or 'normal',
            preprocess=cli_args.preprocess or 'sequential',
            model_frames=cli_args.model_frames or 16,
            feat_dim=cli_args.feat_dim or 768,
            keypoint_field='keypoint',
            model=cli_args.model or 'transformer',
            num_classes=cli_args.num_classes or 15,
            epochs=cli_args.epochs or 50,
            batch_size=cli_args.batch_size or 128,
            lr=cli_args.lr or 1e-3,
            weight_decay=cli_args.weight_decay or 5e-4,
            eta_min=cli_args.eta_min or 1e-6,
            scheduler_type=getattr(cli_args, 'scheduler_type', None) or 'cosine',
            scheduler_step_size=getattr(cli_args, 'scheduler_step_size', None) or 30,
            scheduler_gamma=getattr(cli_args, 'scheduler_gamma', None) or 0.1,
            optimizer_type=getattr(cli_args, 'optimizer_type', None) or 'adamw',
            num_workers=cli_args.num_workers or 4,
            device=cli_args.device,
            output_dir=cli_args.output_dir or './runs/default',
            seed=cli_args.seed or 111,
            print_every=cli_args.print_every or 5,
            save_every=cli_args.save_every or 5,
            eval_batch_size=cli_args.batch_size or 128,
            eval_num_workers=cli_args.num_workers or 4,
            class_weighted=cli_args.class_weighted or False,
            center_frame_only=False,
        )
        if args.ann_file is None:
            parser.error("--ann_file is required when not using --config")

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate mode-specific args
    if args.mode in ('feature', 'fusion') and args.feature_dir is None:
        parser.error(f"--feature_dir required for mode '{args.mode}'")
    if args.mode == 'keypoint' and not hasattr(args, 'feature_dir'):
        args.feature_dir = None  # Not needed for keypoint-only mode

    # Setup logging
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(args.output_dir, f'train_logs_{timestr}.log')
    logging.basicConfig(
        filename=log_filename, level=logging.INFO,
        format='%(asctime)s %(message)s', filemode='w')
    save_logs = True

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    label_names = LABEL_NAMES[args.dataset_type]

    print_and_log(f"Logging to {log_filename}", save_logs)
    print_and_log(f"Config: {json.dumps(vars(args), indent=2)}", save_logs)

    # Seed
    seed_everything(args.seed)
    device = torch.device(args.device)

    # Keypoint num_frames for 1D-CNN (full window, not subsampled)
    kp_num_frames = 48 if args.preprocess == 'sequential' else 144

    # Build dataloaders
    print_and_log("Building dataloaders...", save_logs)
    dl_kwargs = dict(
        ann_file=args.ann_file,
        mode=args.mode,
        preprocess=args.preprocess,
        model_frames=args.model_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_split=args.train_split,
        test_split=args.test_split,
        eval_batch_size=getattr(args, 'eval_batch_size', args.batch_size),
        eval_num_workers=getattr(args, 'eval_num_workers', args.num_workers),
        keypoint_field=getattr(args, 'keypoint_field', 'keypoint'),
    )
    if args.mode in ('feature', 'fusion'):
        dl_kwargs['feature_dir'] = args.feature_dir
        dl_kwargs['feat_dim'] = args.feat_dim
        dl_kwargs['center_frame_only'] = getattr(args, 'center_frame_only', False)
    train_loader, val_loader, test_loader = build_dataloaders(**dl_kwargs)

    print_and_log(
        f"Train: {len(train_loader.dataset)} clips, "
        f"Val: {len(val_loader.dataset)} clips (windowed), "
        f"Test: {len(test_loader.dataset)} clips (stride-1)", save_logs)

    # Build model
    model = build_model(
        args.model, feat_dim=args.feat_dim, num_classes=args.num_classes,
        num_frames=args.model_frames, kp_num_frames=kp_num_frames)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_and_log(
        f"Model: {args.model}, Trainable: {param_count:,}, "
        f"Total: {total_params:,}", save_logs)

    # Optimizer — only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
            weight_decay=args.weight_decay)
    elif args.optimizer_type == 'sgd':
        nesterov = getattr(args, 'nesterov', False)
        momentum = getattr(args, 'momentum', 0.9)
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr, momentum=momentum, nesterov=nesterov,
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
            weight_decay=args.weight_decay)
    print_and_log(f"Optimizer: {args.optimizer_type}", save_logs)

    if args.scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size,
                           gamma=args.scheduler_gamma)
        print_and_log(
            f"Scheduler: StepLR(step_size={args.scheduler_step_size}, "
            f"gamma={args.scheduler_gamma})", save_logs)
    elif args.scheduler_type == 'multistep':
        milestones = getattr(args, 'scheduler_milestones', [90, 100])
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=args.scheduler_gamma)
        print_and_log(
            f"Scheduler: MultiStepLR(milestones={milestones}, "
            f"gamma={args.scheduler_gamma})", save_logs)
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min)
        print_and_log(
            f"Scheduler: CosineAnnealingLR(eta_min={args.eta_min})", save_logs)

    # Loss — optionally with inverse-frequency class weights
    if args.class_weighted:
        print_and_log("Computing class weights from training set...", save_logs)
        label_counts = Counter()
        train_ds = train_loader.dataset
        for i in range(len(train_ds)):
            _, _, label = train_ds._get_clip_params(i)
            label_counts[label] += 1
        n_samples = sum(label_counts.values())
        n_classes = args.num_classes
        # Use sqrt of inverse frequency to dampen extreme ratios
        # Raw inverse freq has 150x range; sqrt reduces to ~12x
        weights = torch.zeros(n_classes)
        for c in range(n_classes):
            count = label_counts.get(c, 1)
            weights[c] = (n_samples / (n_classes * count)) ** 0.5
        weights = weights / weights.min()  # normalize so min weight = 1
        print_and_log(f"Class weights: {weights.tolist()}", save_logs)
        for i, name in enumerate(label_names):
            print_and_log(f"  {name}: count={label_counts.get(i, 0)}, "
                          f"weight={weights[i]:.4f}", save_logs)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # QMF: initialize per-sample history trackers
    qmf_histories = None
    if args.model == 'qmf_fusion':
        from qmf_utils import History
        n_train = len(train_loader.dataset)
        qmf_histories = {
            'vis': History(n_train),
            'kp': History(n_train),
        }
        print_and_log(f"QMF: initialized History for {n_train} training samples", save_logs)

    best_f1 = 0
    best_epoch = 0
    best_acc = 0

    # Warmup config
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    if warmup_epochs > 0:
        print_and_log(
            f"Warmup: linear ramp over {warmup_epochs} epochs", save_logs)

    print_and_log(f"Starting training.", save_logs)
    print_and_log(f"Number of epochs: {args.epochs}", save_logs)
    print_and_log(
        f"Scores reported every {args.print_every} epochs.", save_logs)
    print_and_log(
        f"Checkpoints saved every {args.save_every} epochs to "
        f"{args.output_dir}", save_logs)
    print_and_log("#" * 80, save_logs)

    for epoch in range(args.epochs):
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Linear warmup: scale LR from 0 → base_lr over warmup_epochs
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        print_and_log(f"Starting Epoch {epoch}.", save_logs)

        # OGM-GE: modulation active for first 50 epochs (matches original paper)
        ogm_alpha = 0.1 if args.model == 'ogm_fusion' and epoch <= 50 else 0.0
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mode,
            save_logs=save_logs,
            qmf_histories=qmf_histories,
            ogm_alpha=ogm_alpha)
        # Skip scheduler step during warmup
        if warmup_epochs == 0 or epoch >= warmup_epochs:
            scheduler.step()

        print_and_log(f"Train Epoch {epoch} done.", save_logs)

        if epoch % args.print_every == 0 or epoch == args.epochs - 1:
            train_acc, _, _, _, _ = evaluate(
                model, train_loader, device, args.mode)
            test_acc, macro_f1, weighted_f1, preds, labels = evaluate(
                model, val_loader, device, args.mode)

            print_and_log(
                f"Epoch [{epoch}/{args.epochs}] - "
                f"Training Accuracy: {train_acc:.1f}%, "
                f"Validation Accuracy: {test_acc:.1f}%, "
                f"Macro-F1: {macro_f1:.1f}%, "
                f"Weighted-F1: {weighted_f1:.1f}%, "
                f"Training Loss: {train_loss:.4f}", save_logs)

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, 'best.pth'))
                print_and_log(
                    f"New best! Macro-F1: {macro_f1:.1f}% at epoch {epoch}",
                    save_logs)

        if epoch != 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f'ckpt_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print_and_log(
                f"Saved checkpoint at epoch {epoch} to {ckpt_path}",
                save_logs)

        print_and_log(f"Epoch {epoch} completed.", save_logs)
        print_and_log("#" * 80, save_logs)

    # Save final checkpoint
    ckpt_path = os.path.join(args.output_dir,
                             f'ckpt_epoch_{args.epochs - 1}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print_and_log(f"Saved final checkpoint to {ckpt_path}", save_logs)
    print_and_log("Training completed.", save_logs)
    print_and_log("#" * 80, save_logs)

    # Final evaluation with best model on stride-1 test set
    print_and_log("=" * 80, save_logs)
    print_and_log(
        f"Best epoch: {best_epoch}, Acc: {best_acc:.1f}%, "
        f"Macro-F1: {best_f1:.1f}%", save_logs)
    print_and_log(
        f"Running final evaluation on stride-1 test set "
        f"({len(test_loader.dataset)} clips)...", save_logs)

    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, 'best.pth'), weights_only=True))
    _, _, _, preds, labels = evaluate(model, test_loader, device, args.mode)

    report, report_pretty, mean_class_acc = \
        save_confusion_matrix_and_classification_report(
            true_labels=labels, predicted_labels=preds,
            label_names=label_names, save_dir=args.output_dir)

    print_and_log("\nClassification Report (Best Model):", save_logs)
    print_and_log(report_pretty, save_logs)
    print_and_log(
        f"Mean Class Accuracy: {mean_class_acc * 100:.2f}%", save_logs)

    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Preprocess: {args.preprocess}\n")
        f.write(f"Dataset type: {args.dataset_type}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Best Macro-F1: {best_f1:.2f}%\n")
        f.write(f"Mean Class Accuracy: {mean_class_acc * 100:.2f}%\n\n")
        f.write(report_pretty)

    print_and_log(f"\nResults saved to {args.output_dir}/", save_logs)


if __name__ == '__main__':
    main()
