import torch
from internal_transforms.transforms import Compose
from utils.classification_model import load_model_weights

def load_model_with_transforms(cfg_parser, num_classes, weight_path, device):
    model_class = cfg_parser.get_model_class_from_config()
    model = model_class(num_classes=num_classes, **cfg_parser.model_cfg['args']).to(device)
    model = load_model_weights(model, weight_path=weight_path)
    model.eval()
    transforms = cfg_parser.get_test_transforms()
    transforms = Compose([transform_params['transform_class'](**transform_params['hyperparams']) for transform_params in transforms.values()])
    return model, transforms
