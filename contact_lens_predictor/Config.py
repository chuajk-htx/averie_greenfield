from dataclasses import dataclass, field
import os
import torch

@dataclass
class Config:
    seed: int = 42
    num_classes: int = 3
    class_names: tuple = ('No', 'Clear', 'Cosmetic')
    image_size: tuple = (224, 224)
    n_splits: int = 5 # For K-fold
    split_ratio: float = 0.2 # For train/test split
    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 0.0003682004202434062
    weight_decay: float = 7.250013898492824e-05
    momentum: float = 0.90
    scheduler_step: int = 3
    scheduler_gamma: float = 0.5
    dropout: float = 0.3
    patience: int = 5
    save_interval: int = 10 # Save model every N epochs
    
    # Path configurations
    #csv_path: str = "/Users/averie/Databases/NDCLD13/NDCLD13_LG4000_all.csv"
    #dataset_path: str = '/Users/averie/Documents/Algos/Tests/Image test/NDCLD13/LG4000/sclera'
    #root: str = "/Users/averie/Documents/Algos/Open Source Code Tests/Modified D-NetPAD"
#    csv_path: str = "/Users/averie/Databases/NDCLD15/copy of NDCLD15-with-Segmentation.csv"
#    dataset_path: str = "/Users/averie/Documents/Algos/Tests/Image test/NDCLD15/sclera"
    #pt_path: str = "/Users/averie/Documents/Algos/Open Source Code Tests/LastStep/Tensors"
    #root: str = "/Users/averie/Documents/Algos/Open Source Code Tests/Modified D-NetPAD"
    csv_path: str = "/Users/averie/Databases/Databases_full_TrainingDataSetFinal.csv"
    dataset_path: str = "/Users/averie/Documents/Algos/Tests/Image test/FullDataset"
#    root: str = "/Users/averie/Documents/Algos/Open Source Code Tests/LastStep"
    root: str = "/Users/averie/Documents/Algos/Open Source Code Tests/FinalTraining"

    results_dir: str = None
    
    # Model options
    model_names: list = field(default_factory=lambda: [
        #['resnet18','torchvision'],
        
        #['resnet152','torchvision'],
        ['mobilenetv3_large_100','timm'],
        #['tf_efficientnetv2_b0','timm'],
        #['densenet161','torchvision'],
        #['resnext101_32x8d','timm'],
        #['resnet50','torchvision'],
        #['resnet152','torchvision']

        #['densenet121','timm'], 
        #['densenet161','timm'],
        #['densenet169','timm'],
        #['resnext101_32x8d','timm']
        #['tf_efficientnetv2_s','timm'],
        ])
                              
                              #[['efficientnet_b7','torchvision'],['tf_efficientnet_b7','timm'],['efficientnet_v2_s','torchvision'],['efficientnet_v2_m','torchvision'],['efficientnet_v2_l','torchvision'],
                              # ['tf_efficientnetv2_s','timm'],['tf_efficientnetv2_m','timm'],['tf_efficientnetv2_l','timm'],['tf_efficientnetv2_b0','timm'],
                              #                         ['shufflenet_v2_x1_0','torchvision'],['resnext50_32x4d','torchvision'],['resnext101_32x8d','torchvision'],['resnext50_32x4d','timm'],['resnext101_32x4d','timm'],['resnext101_32x8d','timm'],
                              #                         ['mobilenet_v2','torchvision'],['mobilenet_v3_large','torchvision'],['mobilenet_v3_small','torchvision'],
                              #                         ['mobilenetv2_100','timm'],['mobilenetv3_large_100','timm'],['mobilenetv3_small_100','timm'],['mobilenetv4_conv_small','timm'],['mobilenetv4_conv_large','timm'],['tf_mobilenetv3_large_100','timm'],['tf_mobilenetv3_large_minimal_100','timm'],['tf_mobilenetv3_small_100','timm'],['tf_mobilenetv3_small_minimal_100','timm'],
                              #                         ['densenet121','torchvision'],['densenet161','torchvision'],['densenet169','torchvision'],['densenet201','torchvision'],
                              #                         ['densenet121','timm'],['densenet161','timm'],['densenet169','timm'],['densenet201','timm'],
                              #                         ['resnet18','torchvision'],['resnet34','torchvision'],['resnet50','torchvision'],['resnet101','torchvision'],['resnet152','torchvision'],
                              #                         ['resnet18','timm'],['resnet34','timm'],['resnet50','timm'],['resnet101','timm'],['resnet152','timm']
                              #                         ])

    attention_map: dict = field(default_factory=lambda: {
        'mobilenetv3_large_100': [["all"]], #["layer1"], ["layer2"],["layer3"]],#[["layer1", "layer2"], ["layer2", "layer3"],["layer1", "layer3"]],#[["all"], ["layer1"], ["layer2"]],[["layer3"]],
        #'tf_efficientnetv2_b0': [["all"],["layer1" ], ["layer2"],["layer3"]], #["layer1", "layer2"], ["layer2", "layer3"],["layer1", "layer3"]], #["layer0_layer1"], ["layer1_layer2"], ["layer0_layer2"]],#[["layer1"]]#["all"]],
        #'densenet161': [["all"],["layer1"], ["layer2"], ["layer3"], ["layer4"], ["layer1", "layer2"], ["layer2", "layer3"],["layer3", "layer4"], ["layer1", "layer4"], ["layer2", "layer4"]],
        #'resnext101_32x8d': [["all"]],
        
        #'resnet18': [["all"],["layer1"]]#, ["layer2"], ["layer3"], ["layer4"], ["layer1", "layer2"], ["layer2", "layer3"],["layer3", "layer4"], ["layer1", "layer4"], ["layer2", "layer4"]],
        #'resnet50': [["layer1"], ["layer2"], ["layer3"], ["layer4"]], #[["all"]],
        #'resnet152': [["layer1"], ["layer2"], ["layer3"], ["layer4"]]#[["layer1"]],#[["all"]],

        #'densenet121': [["all"]],#[["layer1"], ["layer2"], ["layer3"], ["layer1", "layer2"], ["layer2", "layer3"]],
        #'densenet161': [["all"], ["layer1"], ["layer2"], ["layer3"], ["layer4"], ["layer1", "layer2"], ["layer2", "layer3"], ["layer1", "layer3"]],
        #'densenet169': [["all"]],
        #'resnext101_32x8d': [["all"]]
        #'densenet': [["denseblock1"], ["denseblock2"], ["denseblock1", "denseblock2"]],
        #'mobilenet': [["block3"], ["block6"], ["block10"], ["block3", "block6"]],
        #'efficientnet': False,
    })

    use_attention: bool = True

    attention_type: str = 'ChannelAttention'  # in case you want to swap SE/CBAM later


#    model_weights_path: str = '/Users/averie/Documents/Algos/Open Source Code Tests/FinalTraining/results/20250827_003031/final_model.pth'
    #'/Users/averie/Documents/Algos/Open Source Code Tests/FinalTraining/results/20250826_205142/final_model.pth' #'/Users/averie/Documents/Algos/Open Source Code Tests/Modified D-NetPAD/results/20250726_182636/densenet121_timm/final_model.pth'
    
#    test_csv: str = '/Users/averie/Documents/Algos/Open Source Code Tests/Modified D-NetPAD/results/20250715_170424/train_test_split.csv'

#    save_path: str = '/Users/averie/Documents/Algos/Tests/Image test/NDCLD13/AD100_try'


    def __post_init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if self.results_dir is None:
            self.results_dir = os.path.join(self.root, "results")