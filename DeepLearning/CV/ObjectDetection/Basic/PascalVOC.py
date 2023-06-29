from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms

img_folder = r'G:\DeepLearningDataset\VOC2012'



train_dataset = VOCDetection(img_folder, year='2012', image_set='train', download=False, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_dataset = VOCDetection(img_folder, year='2012', image_set='val', download=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

