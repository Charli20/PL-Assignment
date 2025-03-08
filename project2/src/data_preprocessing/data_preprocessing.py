import os
import cv2
import imutils
import shutil
import random

class CropImg:
    """Class to crop MRI images by detecting contours around the brain region."""

    @staticmethod
    def crop_img(image):
        """Crops the brain MRI image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
      v if len(cnts) == 0:
            return image  

        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        return new_image

    @staticmethod
    def process_images(base_dir, save_dir, categories):
        """Processes and saves cropped images in training and testing data."""
        os.makedirs(save_dir, exist_ok=True)

        for category in categories:
            input_folder = os.path.join(base_dir, category)
            output_folder = os.path.join(save_dir, category)

            os.makedirs(output_folder, exist_ok=True)

            for img_name in os.listdir(input_folder):
                img_path = os.path.join(input_folder, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f" Skipping {img_name}, unable to read image.")
                    continue

                cropped_img = CropImg.crop_img(image)
                save_path = os.path.join(output_folder, img_name)
                cv2.imwrite(save_path, cropped_img)

            print(f"Cropped images saved in {output_folder}")

class DataSplitting:
    """split testing images into Test and Validation sets."""

    @staticmethod
    def split_testing_images(test_dir, validation_dir, categories, split_size=300):
        """
        data spliting  the testing folder into test and validation
        """
        os.makedirs(validation_dir, exist_ok=True)
        for category in categories:
            test_category_path = os.path.join(test_dir, category)

            validation_category_path = os.path.join(validation_dir, category)           
            os.makedirs(validation_category_path, exist_ok=True)
            images = os.listdir(test_category_path)
            random.shuffle(images)
            validation_images = images[:split_size]

            for img in validation_images:
                src_path = os.path.join(test_category_path, img)
                dest_path = os.path.join(validation_category_path, img)                
                shutil.move(src_path, dest_path)

            print(f"moved {split_size} images from {category} to validation.")

        print("Data Splitting Completed!")        