import importlib
import importlib.util as importlib_util


def class_from_name(module_name, class_name):
    """
    Given a module name and class name, returns the Python class
    """
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c

class ConfigParser():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.get_config_from_file(config_path)
        
        self.dataset_cfg = self.config.dataset_cfg
        self.optimizer_cfg = self.config.optimizer_cfg    
        self.model_cfg = self.config.model_cfg
        self.train_cfg = self.config.train_cfg
        self.loss_cfg = self.config.loss_cfg
        self.log_cfg = self.config.log_cfg
        
        

    def get_config_from_file(self, config_path):
        spec = importlib_util.spec_from_file_location("module.name", config_path)
        config = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    

    def get_optimizer_from_config(self, model):
        optimizer_class = class_from_name("torch.optim", self.optimizer_cfg['type'])
        return optimizer_class(model.parameters(), **self.optimizer_cfg['settings'])
    
    def get_model_class_from_config(self):
        return class_from_name("models", self.model_cfg['type'])
    
    def get_dataset_class_from_config(self):
        return class_from_name("datasets", self.dataset_cfg['dataset_class'])
    
    def get_test_dataset_class_from_config(self):
        return class_from_name("datasets", self.dataset_cfg['test_dataset_class'])
    
    def get_loss_class_from_config(self):
        return class_from_name("torch.nn", self.loss_cfg['type'])
    
    # Transform settings
    def get_transforms(self, config_type, config_settings="train_settings"):
        transforms_dict = config_type[config_settings]['transforms']
        for transform_name, transform_params in transforms_dict.items():
            transform_param_clone = transform_params.copy()
            transform_params.clear()
            transform_params['hyperparams'] = transform_param_clone
            transform_params['transform_class'] = class_from_name("internal_transforms", transform_name)
        return transforms_dict
    
    def get_train_transforms(self):
        return self.get_transforms(config_type=self.train_cfg, config_settings="train_settings")
    
    def get_val_transforms(self):
        return self.get_transforms(config_type=self.train_cfg, config_settings="val_settings")
        
    def get_test_transforms(self):
        return self.get_transforms(config_type=self.train_cfg, config_settings="test_settings")
    

    # Dataloader settings
    def get_dataloader_settings(self, config_type, config_settings="train_settings"):
        dataloader_settings = config_type[config_settings]
        dataloader_settings = {key: value for key, value in dataloader_settings.items() if key in ['batch_size', 'shuffle', 'num_workers']}
        return dataloader_settings
    
    
    # All settings
    def get_train_dataloader_settings(self):
        return self.get_dataloader_settings(config_type=self.train_cfg, config_settings="train_settings")
    
    def get_val_dataloader_settings(self):
        return self.get_dataloader_settings(config_type=self.train_cfg, config_settings="val_settings")
    
    def get_test_dataloader_settings(self):
        return self.get_dataloader_settings(config_type=self.train_cfg, config_settings="test_settings")
    
    
    
    
    
    

