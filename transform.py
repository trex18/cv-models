import torchvision.transforms as T

def get_transform(phase, image_size):
    if phase == 'train':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])