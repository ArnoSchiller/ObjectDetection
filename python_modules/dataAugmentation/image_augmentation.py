import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmenters.color import Grayscale

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import glob
import os

class ImageAugmentation():
    with_visualisation = True
    save_results = True

    augmentation_modes = {
        "rotate":               "Rotation", 
        "flip_horizontal":      "Spiegelung (horizontal)", 
        "flip_vertical":        "Spiegelung (vertikal)",        
        "gaussian_noise":       "Rauschen (Gaussian)",
        "brightness_darker":    "Helligkeit (dunkler)",
        "brightness_brighter":  "Helligkeit (heller)",
        "grayscale":            "Grayscale",
    }
    file_extentions = {
        "rotate":               "_rot", 
        "flip_horizontal":      "_fliphr", 
        "flip_vertical":        "_flipvr",        
        "gaussian_noise":       "_noiseg",
        "brightness_darker":    "_darker",
        "brightness_brighter":  "_brighter",
        "grayscale":            "_gray",
    }

    def __init__(self):
        pass

    def load_basic_image(self, image_path):
        image = imageio.imread(image_path)
        #print(image.shape)
        if image.shape[2] > 3:
            image = image[:,:,0:3]
            self.save_new_image(image_path, "", image)
            print(image.shape)

        xml_path = image_path.split(".")[0] + ".xml"
        bbs = self.load_image_bbs(xml_path, image.shape)
        
        if self.with_visualisation:
            ia.imshow(bbs.draw_on_image(image, size=2))
        return image, bbs

    def load_image_bbs(self, xml_path, img_shape):

        parser = etree.XMLParser()
        xmltree = ElementTree.parse(xml_path, parser=parser).getroot()

        boxes = [] 
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax))
       
        return BoundingBoxesOnImage(boxes, shape=img_shape)

    def save_new_image(self, basic_img_path, filename_ext, image):
        new_file_path = basic_img_path.split(".")[0] + filename_ext + ".png"
        imageio.imwrite(new_file_path, image)
        if os.path.exists(new_file_path):
            return new_file_path
        return False


    def save_new_image_label(self, basic_xml_path, filename_ext, bounding_boxes):

        parser = etree.XMLParser()
        xmltree = ElementTree.parse(basic_xml_path, parser=parser).getroot()

        ## changing the name of the annotated class for every object
        for index, object_iter in enumerate(xmltree.findall('object')):
            bndbox = object_iter.find('bndbox')
            bndbox.find('xmin').text = str(bounding_boxes[index].x1)
            bndbox.find('xmax').text = str(bounding_boxes[index].x2)
            bndbox.find('ymin').text = str(bounding_boxes[index].y1)
            bndbox.find('ymax').text = str(bounding_boxes[index].y2)
        
        new_file_path = basic_xml_path.split(".")[0] + filename_ext + ".xml"
        ## save the file 
        self.save_xml(xmltree, new_file_path)

        if os.path.exists(new_file_path):
            return new_file_path
        return False

    def augmentate_image(self, image, bbs, augmentation_mode):
        # rotate
        if augmentation_mode == self.augmentation_modes["grayscale"]:
            gray=iaa.Grayscale(alpha=1.0)
            augmented_image, augmented_bbs = gray(image=image, bounding_boxes=bbs)

        if augmentation_mode == self.augmentation_modes["rotate"]:
            rotate=iaa.Affine(rotate=(-30, 20))
            augmented_image, augmented_bbs = rotate(image=image, bounding_boxes=bbs)
        # flip_horizontal
        if augmentation_mode == self.augmentation_modes["flip_horizontal"]:
            flip_hr=iaa.Fliplr(p=1.0)
            augmented_image, augmented_bbs = flip_hr(image=image, bounding_boxes=bbs)
        # flip_vertical
        if augmentation_mode == self.augmentation_modes["flip_vertical"]:
            flip_vr=iaa.Flipud(p=1.0)
            augmented_image, augmented_bbs = flip_vr(image=image, bounding_boxes=bbs)
        # gaussian_noise
        if augmentation_mode == self.augmentation_modes["gaussian_noise"]:
            gaussian_noise = iaa.AdditiveGaussianNoise(10,20)
            augmented_image, augmented_bbs = gaussian_noise(image=image, bounding_boxes=bbs)
        # brightness_darker
        if augmentation_mode == self.augmentation_modes["brightness_darker"]:
            bright_aug = iaa.AddToBrightness((-60, -20))
            augmented_image, augmented_bbs = bright_aug(image=image, bounding_boxes=bbs)
        # brightness brighter
        if augmentation_mode == self.augmentation_modes["brightness_brighter"]:
            bright_aug = iaa.AddToBrightness((20, 40))
            augmented_image, augmented_bbs = bright_aug(image=image, bounding_boxes=bbs)

        if self.with_visualisation:
            ia.imshow(augmented_bbs.draw_on_image(augmented_image, size=2))

        return augmented_image, augmented_bbs

    def augmentate_image_file(self, image_path, show_results=False, show_bbs=True):
        self.with_visualisation = False
        image, bboxes = self.load_basic_image(image_path)
        images = [image]
        boxes = [bboxes]
        aug_modes = ["Original"]

        aug_keys = list(self.augmentation_modes.keys())
        aug_values = list(self.augmentation_modes.values())
        augmentations = list(self.augmentation_modes.values())
        augmentations.remove(self.augmentation_modes["rotate"])
        augmentations.remove(self.augmentation_modes["grayscale"])
        # augmentations = [self.augmentation_modes["grayscale"]]
        created_file_paths = []

        for aug_mode in augmentations:
            
            aug_img, aug_bbs = self.augmentate_image(image, bboxes, aug_mode)

            # save image and new label file
            if self.save_results:
                xml_path = image_path.split(".")[0] + ".xml"
                aug_key = aug_keys[aug_values.index(aug_mode)]
                path = self.save_new_image_label(xml_path, self.file_extentions[aug_key], aug_bbs)
                if path is not False:
                    created_file_paths.append(path)
                path = self.save_new_image(image_path, self.file_extentions[aug_key], aug_img)
                if path is not False:
                    created_file_paths.append(path)
            images.append(aug_img)
            boxes.append(aug_bbs)
            aug_modes.append(aug_mode)

        if show_results:
            self.show_images(images=images, titles=aug_modes, bounding_boxes=boxes)
        return created_file_paths
        
    def show_saved_results(self,basic_image_path):
        self.with_visualisation = False
        filename_path = basic_image_path.split(".")[0]
        png_files = glob.glob(filename_path + "*.png")

        images = []
        exts = []
        bbs = []

        for image_path in png_files:
            image, boxes = self.load_basic_image(image_path)
            images.append(image)

            file_extention = image_path.split(".")[0].split("_")[-1]
            if image_path == basic_image_path:
                file_extention = "Original"
            exts.append(file_extention)
            bbs.append(boxes)
        
        self.show_images(images=images, titles=exts, bounding_boxes=bbs)


    def show_images(self, images, titles=None, bounding_boxes=None):
        img_per_row = 3
        num_cols = min(img_per_row, len(images)) 
        num_rows = int(round(len(images)/img_per_row + .5))
        sub_idxs = list(range(num_cols * num_rows))
        fig, ax = plt.subplots(num_rows, num_cols)

        for index, image in enumerate(images):
            col = index % img_per_row
            row = num_cols = int((index)/img_per_row)
            sub_idxs.remove(index)
            if not bounding_boxes is None:
                ax[row, col].imshow(bounding_boxes[index].draw_on_image(image, size=6))
            else:
                ax[row, col].imshow(image)
            
            ax[row, col].axis("off")
            if not titles is None:
                ax[row, col].set_title(titles[index])

        # clear not used subplots
        for index in sub_idxs:
            fig.delaxes(ax.flatten()[index])

        fig.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),"Augmentation.png"))
        plt.show()

    def prettify_xml(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True).replace("  ".encode(), "\t".encode())

    def save_xml(self, root, targetFile):
        out_file = None
        
        out_file = codecs.open(targetFile, 'w')

        prettifyResult = self.prettify_xml(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

if __name__ == "__main__":
    import os
    img_aug = ImageAugmentation()

    input_dir = os.path.dirname(__file__)
    file_path = os.path.join(input_dir, "Shrimps.png")
    img_aug.save_results = False
    img_aug.augmentate_image_file(file_path, show_results=True)