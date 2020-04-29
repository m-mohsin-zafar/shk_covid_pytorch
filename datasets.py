import os
from PIL import Image
from torch.utils.data import Dataset


class CovidDataset(Dataset):
    def __init__(self, root, inp_df, transformations=None):
        self.img_names = [n for n in inp_df.x]
        self.img_paths = [os.path.join(root, f) for f in self.img_names]
        self.img_labels = [la for la in inp_df.y]
        self.transformations = transformations

    def __len__(self):
        count = len(self.img_names)
        return count

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = self.img_paths[item]
        img_label = self.img_labels[item]

        image = Image.open(img_path).convert('RGB')

        if self.transformations is not None:
            # Apply all transformations
            image = self.transformations(image)

        return img_name, image, img_label
