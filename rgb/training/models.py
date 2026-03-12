"""
Models for activity recognition from CLIP/DINOv3/VideoMAE features
and multimodal skeleton+RGB fusion.

Model types:
  Keypoint-only (input: keypoints, shape B×68×T):
    - cnn1d:       Standalone 1D-CNN classifier (identical to keypoints_train)

  Feature-based (input: pre-extracted features, shape B×T×D):
    - meanpool:    Mean pool → linear (CLIP/DINOv3/VideoMAE baseline)
    - transformer: 4-layer temporal transformer with CLS token (EVL-style)

  Fusion (input: pre-extracted features + keypoints):
    - fusion:      1D-CNN keypoint encoder + MeanPool → concat → linear
    - fusion_tiny: fusion with tiny keypoint encoder (wheelchair)
    - robust_fusion: fusion + ModDrop (https://doi.org/10.1109/TPAMI.2015.2461544) + dual-head auxiliary loss
    - robust_fusion_tiny: robust_fusion with tiny keypoint encoder (wheelchair)
    - qmf_fusion:  Quality-aware Multimodal Fusion (Zhang & Wu, ICML 2023, https://arxiv.org/abs/2306.02050)
    - ogm_fusion:  OGM-GE gradient modulation (Peng et al., CVPR 2022, https://arxiv.org/abs/2203.15332)
    - mmcl_fusion: Multi-Modality Co-Learning (Liu et al., ACM MM 2024, https://arxiv.org/abs/2407.15706)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Feature-based temporal heads (existing)
# =============================================================================

class MeanPoolHead(nn.Module):
    """Mean pooling + linear classifier (CLIP/DINOv3/VideoMAE baseline)."""

    def __init__(self, feat_dim=768, num_classes=15, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """x: (B, T, D) → logits: (B, num_classes)"""
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


class TransformerHead(nn.Module):
    """Temporal Transformer Encoder with learnable CLS token (EVL-style)."""

    def __init__(self, feat_dim=768, num_classes=15, num_frames=16,
                 num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_frames + 1, feat_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# =============================================================================
# Keypoint 1D-CNN encoder (from keypoints_train)
# =============================================================================

class KeypointCNN1DEncoder(nn.Module):
    """1D-CNN that extracts features from keypoint sequences.

    Identical to KeypointCNN1D from keypoints_train but returns the 128-dim
    feature vector (after fc1) instead of class logits.

    Input: (B, 68, num_frames) — 17 joints × 4 (x, y, dx, dy) with motion
    Output: (B, 128) — feature vector for fusion
    """

    def __init__(self, num_frames=48, motion_info=True):
        super().__init__()
        num_in_channels = 34 * 2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * num_frames, 128)
        self.feat_dim = 128

    def forward(self, x):
        """x: (B, 68, T) → (B, 128)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class KeypointCNN1D(nn.Module):
    """Standalone 1D-CNN classifier for keypoint sequences.

    Identical to keypoints_train/models/keypointcnn_1d.py KeypointCNN1D.

    Input: (B, 68, num_frames) — 17 joints × 4 (x, y, dx, dy) with motion
    Output: (B, num_classes) — class logits
    """

    def __init__(self, num_frames=48, num_classes=15, motion_info=True):
        super().__init__()
        num_in_channels = 34 * 2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * num_frames, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, keypoints):
        """keypoints: (B, 68, T) → logits: (B, num_classes)"""
        x = F.relu(self.bn1(self.conv1(keypoints)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =============================================================================
# Tiny 1D-CNN variants (wheelchair dataset uses smaller model)
# Matches keypoints_train/models/keypointcnn_1d.py KeypointCNN1DTiny
# =============================================================================

class KeypointCNN1DTinyEncoder(nn.Module):
    """Smaller 1D-CNN encoder: conv2=32ch, fc1=64-dim (vs 64ch/128-dim)."""

    def __init__(self, num_frames=48, motion_info=True):
        super().__init__()
        num_in_channels = 34 * 2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * num_frames, 64)
        self.feat_dim = 64

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class KeypointCNN1DTiny(nn.Module):
    """Standalone tiny 1D-CNN classifier (wheelchair baseline)."""

    def __init__(self, num_frames=48, num_classes=15, motion_info=True):
        super().__init__()
        num_in_channels = 34 * 2 if motion_info else 34
        self.conv1 = nn.Conv1d(num_in_channels, 32, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * num_frames, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, keypoints):
        x = F.relu(self.bn1(self.conv1(keypoints)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FeatureFusionTinyModel(nn.Module):
    """Fusion with tiny keypoint encoder (for wheelchair dataset)."""

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256):
        super().__init__()
        self.kp_encoder = KeypointCNN1DTinyEncoder(
            num_frames=num_frames, motion_info=True)
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, features, keypoints):
        feat = features.mean(dim=1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = F.relu(self.feat_proj(feat))
        kp_feat = self.kp_encoder(keypoints)
        kp_feat = F.relu(self.kp_proj(kp_feat))
        fused = torch.cat([feat, kp_feat], dim=1)
        return self.classifier(fused)


# =============================================================================
# Fusion: 1D-CNN + pre-extracted features (DINO/CLIP)
# =============================================================================

class FeatureFusionModel(nn.Module):
    """Multimodal fusion: 1D-CNN (keypoints) + MeanPool (DINO/CLIP features).

    Architecture:
      - Keypoint branch: KeypointCNN1DEncoder → 128 → proj → proj_dim
      - Feature branch: MeanPool + LayerNorm → feat_dim → proj → proj_dim
      - Fusion: concatenate (proj_dim * 2) → Linear classifier
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256):
        super().__init__()
        self.kp_encoder = KeypointCNN1DEncoder(
            num_frames=num_frames, motion_info=True)
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, features, keypoints):
        """
        features: (B, T, D) — pre-extracted DINO/CLIP features (16 frames)
        keypoints: (B, 68, num_frames) — keypoint sequence (48 frames)
        Returns: logits (B, num_classes)
        """
        # Feature branch: mean pool → project to shared dim
        feat = features.mean(dim=1)  # (B, D)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = F.relu(self.feat_proj(feat))  # (B, proj_dim)

        # Keypoint branch → project to shared dim
        kp_feat = self.kp_encoder(keypoints)  # (B, 128)
        kp_feat = F.relu(self.kp_proj(kp_feat))  # (B, proj_dim)

        # Fusion: concatenate and classify
        fused = torch.cat([feat, kp_feat], dim=1)  # (B, proj_dim*2)
        return self.classifier(fused)


# =============================================================================
# ModDrop Fusion (Neverova et al., IEEE TPAMI 2016) + Auxiliary Head
# =============================================================================

class RobustFeatureFusionModel(nn.Module):
    """Fusion with ModDrop (Neverova et al., IEEE TPAMI 2016) + auxiliary head.

    ModDrop randomly drops entire modality channels during training and
    applies a rescaling factor of (1-p) at test time so that the expected
    activation magnitude is preserved (Section 3.2 of the paper).

    An auxiliary keypoint-only classification head provides a secondary
    training signal to maintain independent pose discriminability.

    Reference:
        Neverova, N., Wolf, C., Taylor, G., & Nebout, F. (2016).
        ModDrop: adaptive multi-modal gesture recognition.
        IEEE TPAMI, 38(8), 1692-1706.

    Training loss: α * CE(fused) + (1-α) * CE(pose), α=0.7
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256, visual_dropout_p=0.5):
        super().__init__()
        self.visual_dropout_p = visual_dropout_p

        # Keypoint branch
        self.kp_encoder = KeypointCNN1DEncoder(
            num_frames=num_frames, motion_info=True)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)

        # Visual feature branch
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)

        # Fused classifier
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

        # Auxiliary pose-only classifier
        self.pose_head = nn.Linear(proj_dim, num_classes)

    def forward(self, features, keypoints):
        # Visual branch: mean pool → project
        feat = features.mean(dim=1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = F.relu(self.feat_proj(feat))  # (B, proj_dim)

        # Keypoint branch → project
        kp_feat = self.kp_encoder(keypoints)  # (B, 128)
        kp_feat = F.relu(self.kp_proj(kp_feat))  # (B, proj_dim)

        # ModDrop (Neverova et al., 2016, Sec. 3.2): per-sample modality
        # dropout during training, test-time rescaling by (1-p).
        # Train: feat = feat * mask,  mask ~ Bernoulli(1-p)
        # Test:  feat = feat * (1-p)
        # E[feat_train] = (1-p) * feat = feat_test
        if self.training:
            mask = torch.bernoulli(torch.full(
                (feat.size(0), 1), 1 - self.visual_dropout_p,
                device=feat.device))
            feat = feat * mask
        else:
            feat = feat * (1 - self.visual_dropout_p)

        # Pose-only logits (auxiliary head)
        pose_logits = self.pose_head(kp_feat)

        # Fused logits
        fused = torch.cat([feat, kp_feat], dim=1)  # (B, proj_dim*2)
        fused_logits = self.classifier(fused)

        if self.training:
            return {'fused': fused_logits, 'pose': pose_logits}
        return fused_logits


class RobustFeatureFusionTinyModel(nn.Module):
    """Fusion with ModDrop (tiny variant) for wheelchair dataset.

    Same as RobustFeatureFusionModel (ModDrop, Neverova et al., IEEE TPAMI
    2016) but uses KeypointCNN1DTinyEncoder (conv2=32ch, fc1=64-dim).
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256, visual_dropout_p=0.5):
        super().__init__()
        self.visual_dropout_p = visual_dropout_p

        # Keypoint branch (tiny)
        self.kp_encoder = KeypointCNN1DTinyEncoder(
            num_frames=num_frames, motion_info=True)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)

        # Visual feature branch
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)

        # Fused classifier
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

        # Auxiliary pose-only classifier
        self.pose_head = nn.Linear(proj_dim, num_classes)

    def forward(self, features, keypoints):
        feat = features.mean(dim=1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = F.relu(self.feat_proj(feat))

        kp_feat = self.kp_encoder(keypoints)
        kp_feat = F.relu(self.kp_proj(kp_feat))

        # ModDrop: train with mask, test with (1-p) rescaling
        if self.training:
            mask = torch.bernoulli(torch.full(
                (feat.size(0), 1), 1 - self.visual_dropout_p,
                device=feat.device))
            feat = feat * mask
        else:
            feat = feat * (1 - self.visual_dropout_p)

        pose_logits = self.pose_head(kp_feat)

        fused = torch.cat([feat, kp_feat], dim=1)
        fused_logits = self.classifier(fused)

        if self.training:
            return {'fused': fused_logits, 'pose': pose_logits}
        return fused_logits


# =============================================================================
# QMF: Quality-aware Multimodal Fusion (ICML 2023)
# =============================================================================

class OGMFusionModel(nn.Module):
    """OGM-GE Fusion (Peng et al., CVPR 2022).

    Standard concat-fusion architecture with a shared classifier. Per-modality
    logits are reconstructed by splitting the classifier weight matrix (matching
    the original implementation). Training uses on-the-fly gradient modulation.

    During training: returns dict with 'fused', 'vis', 'kp' logits + encoder
    features for gradient modulation.
    During inference: returns fused logits.
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim

        # Keypoint branch: 1D-CNN encoder → projection
        self.kp_encoder = KeypointCNN1DEncoder(
            num_frames=num_frames, motion_info=True)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)

        # Visual branch: MeanPool + LayerNorm → projection
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)

        # Shared concat classifier (like original OGM-GE ConcatFusion)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

    def forward(self, features, keypoints):
        """
        features: (B, T, D) — pre-extracted visual features
        keypoints: (B, 68, num_frames) — keypoint sequence
        """
        # Visual branch
        feat = features.mean(dim=1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        vis_feat = F.relu(self.feat_proj(feat))    # (B, proj_dim)

        # Keypoint branch
        kp_feat = self.kp_encoder(keypoints)       # (B, 128)
        kp_feat = F.relu(self.kp_proj(kp_feat))    # (B, proj_dim)

        # Fused logits via shared classifier
        fused = torch.cat([vis_feat, kp_feat], dim=1)  # (B, proj_dim*2)
        fused_logits = self.classifier(fused)           # (B, num_classes)

        if self.training:
            # Reconstruct per-modality logits by splitting classifier weights
            # (matching original OGM-GE: fc_out splits into two halves)
            W = self.classifier.weight  # (num_classes, proj_dim*2)
            b = self.classifier.bias    # (num_classes,)
            vis_logits = F.linear(vis_feat, W[:, :self.proj_dim], b / 2)
            kp_logits = F.linear(kp_feat, W[:, self.proj_dim:], b / 2)
            return {
                'fused': fused_logits,
                'vis': vis_logits,
                'kp': kp_logits,
            }
        return fused_logits


class QMFFusionModel(nn.Module):
    """Quality-aware Multimodal Fusion (QMF, Zhang & Wu, ICML 2023).

    Two independent classifier heads (visual + keypoint) with energy-based
    dynamic fusion. At inference, OOD visual features produce diffuse logits
    with low energy scores and are automatically down-weighted.

    Training returns dict with per-modality logits + confidence for QMF loss.
    Inference returns energy-weighted fused logits.
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=48,
                 dropout=0.1, proj_dim=256, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # Keypoint branch: 1D-CNN encoder → projection → classifier
        self.kp_encoder = KeypointCNN1DEncoder(
            num_frames=num_frames, motion_info=True)
        self.kp_proj = nn.Linear(self.kp_encoder.feat_dim, proj_dim)
        self.kp_classifier = nn.Linear(proj_dim, num_classes)

        # Visual branch: MeanPool + LayerNorm → projection → classifier
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.feat_dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Linear(feat_dim, proj_dim)
        self.vis_classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, features, keypoints):
        """
        features: (B, T, D) — pre-extracted visual features (16 frames)
        keypoints: (B, 68, num_frames) — keypoint sequence (48 frames)
        """
        # Visual branch
        feat = features.mean(dim=1)  # (B, D)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = F.relu(self.feat_proj(feat))  # (B, proj_dim)
        vis_logits = self.vis_classifier(feat)  # (B, num_classes)

        # Keypoint branch
        kp_feat = self.kp_encoder(keypoints)  # (B, 128)
        kp_feat = F.relu(self.kp_proj(kp_feat))  # (B, proj_dim)
        kp_logits = self.kp_classifier(kp_feat)  # (B, num_classes)

        # Energy-based confidence: -temperature * logsumexp(logits)
        # Higher logsumexp = more concentrated logits = more confident
        vis_energy = -torch.logsumexp(vis_logits, dim=1)
        kp_energy = -torch.logsumexp(kp_logits, dim=1)
        vis_conf = -self.temperature * vis_energy  # (B,)
        kp_conf = -self.temperature * kp_energy    # (B,)

        # Dynamic fusion: weighted sum of logits
        fused_logits = (vis_logits * vis_conf.unsqueeze(1) +
                        kp_logits * kp_conf.unsqueeze(1))

        if self.training:
            return {
                'fused': fused_logits,
                'vis': vis_logits,
                'kp': kp_logits,
                'vis_conf': vis_conf,
                'kp_conf': kp_conf,
            }
        return fused_logits


# =============================================================================
# MMCL: Multi-Modality Co-Learning (Liu et al., ACM Multimedia 2024)
# =============================================================================

class ContrastiveLoss(nn.Module):
    """Symmetric NT-Xent with intra-modal + inter-modal negatives.

    From MMCL's FAM.py (Contrastive_loss class). Uses both same-modality
    and cross-modality pairs as negatives in the denominator.

    Args:
        tau: Temperature parameter (default 0.4, from MMCL).
    """

    def __init__(self, tau=0.4):
        super().__init__()
        self.tau = tau

    def _cosine_sim(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def _semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self._cosine_sim(z1, z1))
        between_sim = f(self._cosine_sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def forward(self, z1, z2):
        l1 = self._semi_loss(z1, z2)
        l2 = self._semi_loss(z2, z1)
        return ((l1 + l2) * 0.5).mean()


class MMCLFusionModel(nn.Module):
    """MMCL: Multi-Modality Co-Learning (Liu et al., ACM MM 2024).

    Joint classification + contrastive alignment. Classification is
    keypoint-only; visual features are used solely for contrastive
    alignment during training and discarded at inference.

    Training loss: L_cls + 0.2 * L_C  (paper Table 6; L_R omitted)

    Reference:
        Liu, J., Chen, C., & Liu, M. (2024). Multi-Modality Co-Learning
        for Efficient Skeleton-based Action Recognition. ACM MM 2024.
    """

    def __init__(self, feat_dim=768, num_classes=15, num_frames=16,
                 kp_num_frames=48, proj_dim=256, tau=0.4):
        super().__init__()

        # Keypoint encoder + classifier (keypoint-only classification)
        self.kp_encoder = KeypointCNN1DEncoder(
            num_frames=kp_num_frames, motion_info=True)
        self.classifier = nn.Linear(self.kp_encoder.feat_dim, num_classes)

        # FAM: Feature Alignment Module
        # Original MMCL (FAM_Aligh): only the RGB side has an MLP projector.
        # Skeleton features from the GCN encoder are used directly for
        # contrastive alignment (no skeleton projector). We follow the same
        # design: vis_projector maps visual features → kp_encoder.feat_dim,
        # then contrastive loss is computed between raw kp_feat and
        # projected vis_feat.
        self.vis_projector = nn.Sequential(
            nn.Linear(feat_dim, self.kp_encoder.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.kp_encoder.feat_dim, self.kp_encoder.feat_dim),
        )
        self.contrastive_loss = ContrastiveLoss(tau=tau)

    def forward(self, features, keypoints):
        """
        features: (B, T_vis, D) pre-extracted visual features
        keypoints: (B, 68, T) keypoint sequences

        Training: returns {'fused': logits, 'contrastive_loss': scalar}
        Inference: returns logits (B, num_classes)
        """
        kp_feat = self.kp_encoder(keypoints)    # (B, 128)
        logits = self.classifier(kp_feat)       # (B, num_classes)

        if self.training:
            vis_feat = features.mean(dim=1)         # (B, feat_dim)
            vis_emb = self.vis_projector(vis_feat)   # (B, 128)
            c_loss = self.contrastive_loss(kp_feat, vis_emb)
            return {'fused': logits, 'contrastive_loss': c_loss}

        return logits


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_type, feat_dim=768, num_classes=15, num_frames=16,
                kp_num_frames=48, **kwargs):
    """Build model by type.

    Args:
        model_type: One of 'cnn1d', 'cnn1d_tiny', 'meanpool', 'transformer',
                    'fusion', 'fusion_tiny', 'robust_fusion', 'robust_fusion_tiny',
                    'qmf_fusion', 'ogm_fusion', 'mmcl_fusion'
        feat_dim: Feature dimension (for feature-based models)
        num_classes: Number of activity classes
        num_frames: Temporal frames for feature models (16)
        kp_num_frames: Keypoint sequence length for 1D-CNN (48)
    """
    # Standalone keypoint models
    if model_type == 'cnn1d':
        return KeypointCNN1D(
            num_frames=kp_num_frames, num_classes=num_classes)
    elif model_type == 'cnn1d_tiny':
        return KeypointCNN1DTiny(
            num_frames=kp_num_frames, num_classes=num_classes)

    # Feature-based temporal heads
    elif model_type == 'meanpool':
        return MeanPoolHead(feat_dim=feat_dim, num_classes=num_classes)
    elif model_type == 'transformer':
        return TransformerHead(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=num_frames)
    # Feature + keypoint fusion
    elif model_type == 'fusion':
        return FeatureFusionModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)
    elif model_type == 'fusion_tiny':
        return FeatureFusionTinyModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)

    # Robust fusion (modality dropout + dual head)
    elif model_type == 'robust_fusion':
        return RobustFeatureFusionModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)
    elif model_type == 'robust_fusion_tiny':
        return RobustFeatureFusionTinyModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)

    # QMF fusion (energy-based dynamic weighting)
    elif model_type == 'qmf_fusion':
        return QMFFusionModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)

    # OGM-GE fusion (concat + gradient modulation, Peng et al. CVPR 2022)
    elif model_type == 'ogm_fusion':
        return OGMFusionModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=kp_num_frames)

    # MMCL fusion (contrastive co-learning, Liu et al. ACM MM 2024)
    elif model_type == 'mmcl_fusion':
        return MMCLFusionModel(
            feat_dim=feat_dim, num_classes=num_classes,
            num_frames=num_frames, kp_num_frames=kp_num_frames)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
