import torch
from torch.optim.lr_scheduler import StepLR
import logging
import os


def get_segmentation_accuracy(loader, model, device, return_preds=False):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            
            # Transpose outputs to match the shape [batch_size, num_classes, num_frames/height/width]
            outputs = outputs.transpose(1, 2)
            
            _, predicted = torch.max(outputs, 2)  # Get the max along the class dimension
            
            # Flatten labels and predictions for comparison
            predicted_flatten = predicted.flatten()
            labels_flatten = labels.flatten()

            # Create a mask to filter out ignored labels, such as -1
            valid_mask = labels_flatten != -1

            # Apply the mask to filter predictions and labels
            predicted_valid = predicted_flatten[valid_mask]
            labels_valid = labels_flatten[valid_mask]

            # Update correct and total counts based on valid predictions only
            correct_preds = (predicted_valid == labels_valid).sum().item()
            total_valid = labels_valid.size(0)

            correct += correct_preds
            total += total_valid
            
            if return_preds:
                all_preds.extend(predicted_valid.cpu().numpy())
                all_labels.extend(labels_valid.cpu().numpy())

    val_accuracy = 100 * correct / total if total > 0 else 0

    if return_preds:
        return val_accuracy, all_preds, all_labels
    return val_accuracy



def get_classification_accuracy(loader, model, device, return_preds = False):
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.float().to(device), labels.to(device)
            labels = torch.flatten(labels)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if return_preds:
        return 100 * correct / total, all_preds, all_labels         
    return 100 * correct / total



def one_train_segment_epoch(model, loader, device, optimizer, criterion, save_logs, iter_every=50):
    model.train()
    total_iters = len(loader)
    
    epoch_loss = 0.0
    num_batches = 0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()
        
        if i % iter_every == 0 or i == total_iters-1:
            print_and_log(f"Iter {i+1}/{total_iters} done. Iter loss: {loss.item()}", save_logs)
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss/num_batches



def one_train_epoch(model, loader, device, optimizer, criterion, save_logs, iter_every = 50):
    model.train()
    total_iters = len(loader)
    
    epoch_loss = 0.0
    num_batches = 0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        labels = labels.view(-1)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        if i % iter_every == 0 or i == total_iters-1:
            print_and_log(f"Iter {i+1}/{total_iters} done. Iter loss: {loss.item()}", save_logs)
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss/num_batches


def print_and_log(str, log=True):
    print(str)
    if log:
        logging.info(str)


def train_and_validate(num_epochs, model, train_loader, val_loader, optimizer, criterion, scheduler=None, 
                       print_every=10, save_every=100, save_ckpt=False, ckpt_path=None, 
                       save_logs=True, log_filename=None, mode="classification"):
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if save_logs:
        assert log_filename is not None, "Please specify the path to save the logs"
        logging.basicConfig(filename=log_filename, 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
        logger=logging.getLogger()
        
        print_and_log(f"Logging to {log_filename}", save_logs)
        print_and_log(f"Starting training.", save_logs)
        print_and_log(f"Number of epochs: {num_epochs}", save_logs)
        print_and_log(f"The scores will be reported every {print_every} epochs.", save_logs)
        print_and_log(f"Checkpoints will be saved every {save_every} epochs to {os.path.dirname(ckpt_path)}", save_logs)
        print_and_log(f"#################################################################################", save_logs)
        
    
    device = next(model.parameters()).device
    
    if save_ckpt:
        assert ckpt_path is not None, "Please specify a checkpoint path to save the model"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    train_losses = []
    train_accuracies = [0.0]
    val_accuracies = [0.0]

    for epoch in range(num_epochs):
        train_loader.dataset.set_epoch(epoch)
        
        print_and_log(f"Starting Epoch {epoch}.", save_logs)
        
        if mode == "classification":
            average_train_loss = one_train_epoch(model, train_loader, device, optimizer, criterion, save_logs)
        else:
            average_train_loss = one_train_segment_epoch(model, train_loader, device, optimizer, criterion, save_logs)
        
        scheduler.step()

        print(f"Train Epoch {epoch} done.")

        if (epoch) % print_every == 0 or epoch==num_epochs-1:
            train_losses.append(average_train_loss)

            if mode == "classification":
                train_acc = get_classification_accuracy(train_loader, model, device)
                val_acc = get_classification_accuracy(val_loader, model, device)
            else:
                train_acc = get_segmentation_accuracy(train_loader, model, device)
                val_acc = get_segmentation_accuracy(val_loader, model, device)
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            print_and_log(f"Epoch [{epoch}/{num_epochs}] - Training Accuracy: {train_acc}%, Validation Accuracy: {val_acc}%, Training Loss: {average_train_loss}",
                           save_logs)
            
        if epoch!=0 and (epoch) % save_every == 0 and save_ckpt:
            ckpt_path_curr =os.path.join(os.path.dirname(ckpt_path), os.path.basename(ckpt_path).split('.')[0] + f"_epoch_{epoch}.pt")
            
            torch.save(model.state_dict(), ckpt_path_curr)
            print_and_log(f"Saved checkpoint at epoch {epoch} to {ckpt_path_curr}", save_logs)
        
        print_and_log(f"Epoch {epoch} completed.", save_logs)
        print_and_log(f"#################################################################################", save_logs)
    
    
    ckpt_path_curr =os.path.join(os.path.dirname(ckpt_path), os.path.basename(ckpt_path).split('.')[0] + f"_epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path_curr)
    print_and_log(f"Saved checkpoint at epoch {epoch} to {ckpt_path_curr}", save_logs)
    print_and_log(f"Epoch {epoch} completed.", save_logs)
    print_and_log(f"#################################################################################", save_logs)
    print_and_log(f"Training completed.", save_logs)

    return train_losses, train_accuracies, val_accuracies

