import numpy as np
import cv2
from skimage import measure

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        preds = preds / np.max(preds)  # normalize output to 0-1
        predits  = np.array(preds> 0.5).astype('int64')
        labelss = np.array(labels).astype('int64')


        image = measure.label(predits, connectivity=2)  
        coord_image = measure.regionprops(image)        
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)   
            self.image_area_total.append(area_image)     

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))   
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)  
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)  
                    self.image_area_match.append(area_image)   

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]   
        self.dismatch_pixel +=np.sum(self.dismatch)
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        return Final_PD, float(Final_FA)
