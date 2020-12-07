import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Convert xml files to csv file.')
parser.add_argument('--train_test_split', action='store_true', 
                    help='if selected a train_test_split will be done (80/20).')
parser.add_argument('--splitted', action='store_true', 
                    help='if selected a eval_train_test_split will be done (10/(80/20)).')
parser.add_argument('--random_choice', action='store_true', default=False, 
                    help='if selected the split will be random, else it will be like beginning|end.')
parser.add_argument('--xml_files_dir', type=str, default="TEMP_DOWNLOADS",
                    help='directory of the xml files, default: TEMP_DOWNLOADS.')


args = parser.parse_args()

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
        """
        if len(root.findall('object')) == 0:
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     " ",0,0,0,0
                     )
            xml_list.append(value)
        """
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main(args):
    
    directory = os.path.join(os.path.dirname(__file__), args.xml_files_dir)
    xml_df = xml_to_csv(directory)
    xml_df.to_csv('images.csv', index=0)
    print('Successfully converted xml to csv.')

    if args.splitted:
        create_eval_test_train_split()
    elif args.train_test_split:
        create_test_train_split()

def create_eval_test_train_split():
    base_file = os.path.join(os.path.dirname(__file__), 'images.csv')
    labels = pd.read_csv(base_file)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    eval_percentage = 0.1
    test_percentage = 0.2
    num_labels = len(labels)
    num_eval = int(eval_percentage * num_labels)
    num_training = num_labels - num_eval

    num_test = int(test_percentage * num_training)
    num_train = num_training - num_test

    print("Splitting into eval ({}), train ({}) and test data ({})".format(num_eval, num_train, num_test))

    eval_labels = []
    training_labels = []

    #print(labels)
    if args.random_choice: 
        labels_list = [lbl for _,lbl in labels.iterrows()]
        rand_index = np.random.randint(low=0, high=num_labels-1, size=num_eval).tolist()
        eval_labels = [labels_list[i] for i in rand_index]
        training_labels = [labels_list[i] for i in range(len(labels_list)) if i not in rand_index]
    else:
        c = 0
        for index, label in labels.iterrows():
            split_bool = True if index < num_eval else False

            if split_bool:
                eval_labels.append(label)
                c+=1
            else:
                training_labels.append(label)

    test_labels = []
    train_labels = []
    rand_index = np.random.randint(low=0, high=num_training-1, size=num_test).tolist()

    if args.random_choice: 
        rand_index = np.random.randint(low=0, high=num_training-1, size=num_test).tolist()
        test_labels = [training_labels[i] for i in rand_index]
        train_labels = [training_labels[i] for i in range(len(training_labels)) if i not in rand_index]
    else:
        for index, label in enumerate(training_labels):
            split_bool = True if index < num_test else False

            if split_bool:
                test_labels.append(label)
                c+=1
            else:
                train_labels.append(label)

    print("Splitted into eval ({}), train ({}) and test data ({})".format(len(eval_labels), len(train_labels), len(test_labels)))
    
    eval_df = pd.DataFrame(eval_labels, columns=column_name)
    eval_df.to_csv('eval.csv', index=0)

    test_df = pd.DataFrame(test_labels, columns=column_name)
    test_df.to_csv('test.csv', index=0)

    train_df = pd.DataFrame(train_labels, columns=column_name)
    train_df.to_csv('train.csv', index=0)

    print("created train-test-split")


def create_test_train_split():
    base_file = os.path.join(os.path.dirname(__file__), 'images.csv')
    labels = pd.read_csv(base_file)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    test_percentage = 0.2
    num_labels = len(labels)
    num_test = int(test_percentage * num_labels)
    num_train = num_labels - num_test

    print("Splitting into eval ({}), train ({}) and test data ({})".format(0, num_train, num_test))
    print(num_labels)
    test_labels = []
    train_labels = []

    if args.random_choice: 
        labels_list = [lbl for _,lbl in labels.iterrows()]
        rand_index = np.random.randint(low=0, high=num_labels-1, size=num_test).tolist()
        print(len(rand_index))
        test_labels = [labels_list[i] for i in rand_index]
        train_labels = [labels_list[i] for i in range(len(labels_list)) if i not in rand_index]
    else:
        c = 0
        for index, label in labels.iterrows():
            split_bool = True if index < num_test else False

            if split_bool:
                test_labels.append(label)
                c+=1
            else:
                train_labels.append(label)

    print("Splitted into eval ({}), train ({}) and test data ({})".format(0, len(train_labels), len(test_labels)))
    
    test_df = pd.DataFrame(test_labels, columns=column_name)
    test_df.to_csv('test.csv', index=0)

    train_df = pd.DataFrame(train_labels, columns=column_name)
    train_df.to_csv('train.csv', index=0)

    print("created train-test-split")



main(args)