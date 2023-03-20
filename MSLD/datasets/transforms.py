import torchvision.transforms as T

def build_transforms(cfg, is_train=0):
    mean = [104 / 255.0, 117 / 255.0, 128 / 255.0]
    std = [1.0 / 255, 1.0 / 255, 1.0 / 255]

    normalize_transform = T.Normalize(mean=mean, std=std)
    if is_train==1: 
        transform = T.Compose([
            T.Resize(size=256),
            T.RandomResizedCrop(scale=[0.16,1],size=227),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_transform,
        ])
    else: 
        transform = T.Compose([
            T.Resize(size=256),
            T.CenterCrop(227),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
