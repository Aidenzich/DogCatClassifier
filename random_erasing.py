
import random
import math
from PIL import Image


class RandomErasing(object):
    def __init__(
        self, EPSILON_PROPORTION = 0.77, sl = 0.02, sh = 0.4, r1 = 0.3, mean_arr=[0.4914, 0.4822, 0.4465]
        ):
        self.EPSILON_PROPORTION = EPSILON_PROPORTION
        self.mean_arr = mean_arr
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON_PROPORTION:
            return img

        for _ in range(100):            
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(
                math.sqrt(target_area * aspect_ratio))
                )
            w = int(
                round(math.sqrt(target_area / aspect_ratio))
                )

            if w < img.size()[2] and h < img.size()[1]:
                x = random.randint(0, img.size()[1] - h)
                y = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    
                    img[0, x: x+h, y: y+w] = self.mean_arr[0]
                    img[1, x: x+h, y: y+w] = self.mean_arr[1]
                    img[2, x: x+h, y: y+w] = self.mean_arr[2]
                    
                else:
                    img[0, x: x+h, y: y+w] = self.mean_arr[1]
                    
                return img

        return img

def showcase():
    img = Image.open('test/98.jpg')
    import torchvision.transforms as transforms 
    # use transform to transform the image to a specific formula
    
    trans = transforms.Compose([        
        transforms.ToTensor()      
    ])
    trans_back = transforms.ToPILImage()

    trans_img = trans(img)
    re = RandomErasing(1)
    re_img = re(trans_img)
    return trans_back(re_img )