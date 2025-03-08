import os

class RenameImg:
    """
        Renames all images in both training and testing.
        
    """
    
    @staticmethod
    def rename_images(folder, prefix):
        
        count = 1

        for filename in os.listdir(folder):
            source = os.path.join(folder, filename)
            ext = filename.split('.')[-1] 
            destination = os.path.join(folder, f"{prefix}_{count}.{ext}") 

            os.rename(source, destination)
            count += 1

        print(f"All files renamed in {folder}")

