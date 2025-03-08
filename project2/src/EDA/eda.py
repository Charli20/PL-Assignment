import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class EDA:
    """
    Exploratory Data Analysis
    """
    
    @staticmethod
    def count_images(directory):
        return len(os.listdir(directory))
    
    @staticmethod
    def plot_image_counts():
        categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

        train_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\Training"
        test_dir = r"C:\Users\chali\OneDrive\Documents\planing computer project\project2\datasets\Testing"
        
        img_counts = {}

        for category in categories:
            train_path = os.path.join(train_dir, category)
            test_path = os.path.join(test_dir, category)

            
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_count = EDA.count_images(train_path)
                test_count = EDA.count_images(test_path)
                img_counts[category] = {'Training': train_count, 'Testing': test_count}
                print(f"{category} - Training: {train_count} images, Testing: {test_count} images")
            else:
                print(f"Directory not found for category '{category}'")

        
        df_counts = pd.DataFrame(img_counts).T

        
        if not df_counts.empty:
            df_counts.plot(kind='bar', figsize=(10, 6), colormap='viridis')
            plt.xlabel("Tumor Type")
            plt.ylabel("number of images")
            plt.title("Count of brain tumor images in training and testing")
            plt.xticks(rotation=45)
            plt.legend(title="dataset")
            plt.show()
        else:
            print(".")

