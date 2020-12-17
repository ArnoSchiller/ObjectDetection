from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import os

class XMLHandler():

    def __init__(self):
        pass
    
    def create_detection_xml(self, output_filepath,
                                    img_filepath,
                                    boxes,
                                    scores, 
                                    classes,
                                    img_size):

        root = ElementTree.Element("annotation") 

        sub = ElementTree.SubElement(root, "folder")
        sub.text=os.path.split(os.path.split(img_filepath)[0])[-1] 
        
        sub = ElementTree.SubElement(root, "filename") 
        sub.text=os.path.split(img_filepath)[-1]
        
        sub = ElementTree.SubElement(root, "path") 
        sub.text=img_filepath
        
        key = ElementTree.SubElement(root, "source") 
        sub = ElementTree.SubElement(key, "database") 
        sub.text="Unknown"

        key = ElementTree.SubElement(root, "size") 
        sub = ElementTree.SubElement(key, "width") 
        sub.text=str(img_size[0])
        sub = ElementTree.SubElement(key, "height") 
        sub.text=str(img_size[1])
        sub = ElementTree.SubElement(key, "depth") 
        sub.text=str(img_size[2])

        sub = ElementTree.SubElement(root, "segmented") 
        sub.text=str(0)

        for idx, box in enumerate(boxes):
            obj = ElementTree.SubElement(root, "object") 
            
            sub = ElementTree.SubElement(obj, "name")
            sub.text = classes[idx]
            
            sub = ElementTree.SubElement(obj, "pose")
            sub.text = 'Unspecified'
            
            sub = ElementTree.SubElement(obj, "truncated")
            sub.text = '1'
            
            sub = ElementTree.SubElement(obj, "difficult")
            sub.text = '0'

            bb = ElementTree.SubElement(obj, "bndbox")
            sub = ElementTree.SubElement(bb, "xmin")
            sub.text=str(box[0])
            sub = ElementTree.SubElement(bb, "ymin")
            sub.text=str(box[1])
            sub = ElementTree.SubElement(bb, "xmax")
            sub.text=str(box[2])
            sub = ElementTree.SubElement(bb, "ymax")
            sub.text=str(box[3])

        tree = ElementTree.ElementTree(root).getroot()
      
        self.save(tree, output_filepath)


    def change_xml_file(self, file_path, key, value):
        new_file_path = file_path
        parser = etree.XMLParser()
        xmltree = ElementTree.parse(file_path, parser=parser).getroot()
        
        if key == "class_name":
            ## changing the name of the annotated class for every object
            for object_iter in xmltree.findall('object'):
                name = object_iter.find('name')
                name.text = str(value)
        elif key == "file_name":
            filename = xmltree.find('filename')
            filename.text = str(value)
        elif key == "file_path":
            filepath = xmltree.find('path')
            filepath.text = str(value)
        elif key == "folder":
            folder = xmltree.find('folder')
            folder.text = str(value)
        self.save(xmltree, new_file_path)

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True).replace("  ".encode(), "\t".encode())

    def save(self, root, targetFile):
        out_file = None
        
        out_file = codecs.open(targetFile, 'w')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

h = XMLHandler()
h.create_detection_xml("out.xml", "path/to/file.png", [[0.5, 0.1, 0.6, 0.2]], [0.87], ['shrimp'], (640, 640, 3))