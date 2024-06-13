import torch
import torch.nn.functional as F
import numpy as np

MAIN_LABELS =  ["stand", "stand_activity", "walk", "sit", "sit_activity", "sitting_down", "getting_up",
           "bend", "unstable", "fall", "lie_down", "lying_down", "reach", "run", "jump"]

main_ylabelmap = {i:MAIN_LABELS.index(i) for i in MAIN_LABELS}

WHEELCHAIR_LABELS = [
    'sit', 'propel',
    'pick_place',
    'sit_activity', 
    'bend', 'getting_up',
    'exercise',
    'sitting_down', 
    'prepare_transfer', 'transfer', 'fall', 
    'lie_down', 'lying_down', 
    'get_propelled',
    'stand'
    ]

wheelchair_ylabelmap = {i:WHEELCHAIR_LABELS.index(i) for i in WHEELCHAIR_LABELS}

normal_labels = list(main_ylabelmap.keys())
wheelchair_labels = list(wheelchair_ylabelmap.keys())

def load_model_weights(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cuda:0')
    model.load_state_dict(state_dict)
    return model

def categoryFromOutput(logits, return_confidence=False, label_type = "normal"):
    probabilities = F.softmax(logits, dim=1)
    top_n, top_i = probabilities.topk(1, dim=1)
    
    predicted_idx = top_i[0].item()
    confidence = top_n[0].item()
    
    labels = normal_labels if label_type == "normal" else wheelchair_labels
    
    if (len(labels) > predicted_idx):
        category = labels[predicted_idx]
    else:
        category = ""
    
    if return_confidence:
        return category, confidence
    else:
        return category


def cnn1d_infer(model, transforms, device, input, return_secondary=False, return_confidence=False, label_type="normal"):
    reshaped_data = input.reshape(48, 17, 3)
    keypoints_only = reshaped_data[:, :, :2]
    
    min_x, max_x = 0, 1919
    min_y, max_y = 0, 1079
    keypoints_only[..., 0] = np.clip(keypoints_only[..., 0], min_x, max_x)
    keypoints_only[..., 1] = np.clip(keypoints_only[..., 1], min_y, max_y)
    
    input = torch.from_numpy(keypoints_only)
    input = transforms(input)
    tensor_input = input.to(device)

    model.eval()
    with torch.no_grad():
        tensor_input = tensor_input.unsqueeze(0)
        mainout = model(tensor_input.float())
        action, confidence = categoryFromOutput(mainout, return_confidence=return_confidence, label_type=label_type)

    results = [action]
    if return_confidence:
        results.append(confidence)

    # If return_secondary is True, return the top 3 actions
    if return_secondary:
        _mainout = mainout.detach().clone().cpu()
        _mainout[0, _mainout.argmax(dim=1)] = -float('inf')
        
        # Extract second and third predictions directly from probabilities
        action2, confidence2 = categoryFromOutput(_mainout, return_confidence=return_confidence, label_type=label_type)
        _mainout[0, _mainout.argmax(dim=1)] = -float('inf')
        
        action3, confidence3 = categoryFromOutput(_mainout, return_confidence=return_confidence, label_type=label_type)
        
        results.extend([action2, action3, confidence2, confidence3])

    return tuple(results)
