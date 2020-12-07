import boto3 
from tabulate import tabulate       # pip install tabulate
import pandas as pd                 # pip install pandas

from dataset_handler import DatasetHandler

class DatasetStatistics:
    
    bucket_names = []

    def __init__(self):

        self.dataset_handler = DatasetHandler()
        self.bucket_names = self.dataset_handler.bucket_names

    def summery(self):

        matrix = []
        for bucket_name in self.bucket_names:
            num_complete = len(self.dataset_handler.get_complete_dataset_list(bucket_name))
            num_labels = len(self.dataset_handler.get_all_label_names(bucket_name))
            num_images = len(self.dataset_handler.get_all_image_names(bucket_name))
            matrix.append((bucket_name, num_complete, num_labels, num_images))
        df = pd.DataFrame(matrix, columns=["Name", "gelabelte Bilder", "Anzahl Label", "Anzahl Bilder"])
        
        return tabulate(df, headers='keys', tablefmt='psql')

if __name__ == "__main__":
    stats = DatasetStatistics()
    print(stats.summery())