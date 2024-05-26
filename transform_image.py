from DnDMetric import Metric,xy_to_polar,polar_to_xy
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.image import imread,imsave
import numpy as np

# pos is the observer's position on the image mapped from
# -1 to 1 in x and y. (so 0,0 is the center of the image)
def transform_image(image,metric,pos = [0,0]):
    height,width = image.shape[0:2]
    x_orig, y_orig = np.arange(width), np.arange(height)
    X_orig, Y_orig = np.meshgrid(x_orig,y_orig)

    #X_new = np.zeros(X_orig.shape())
    #Y_new = np.zeros(Y_orig.shape())
    coord_pixels = []

    for i in range(len(x_orig)):
        for j in range(len(y_orig)):
            x = x_orig[i] - width // 2
            y = y_orig[j] - height // 2
            v = np.array([x,y])
            
            # Where the magic happens
            r_new = np.sqrt(metric(v,v,x,y))
            
            if np.isnan(r_new):r_new = 0
            
            _,phi = xy_to_polar(x,y)
            x_new,y_new = polar_to_xy(r_new,phi)
            #print(x_new,y_new," from ",r_new,phi)
            #X_new[j,i] = x_new
            #Y_new[j,i] = y_new

            coord_pixels.append([x_new,y_new,*image[j,i,:]])
    
    coord_pixels = np.array(coord_pixels)
    nb_color_channels = coord_pixels.shape[1] - 2
    interps = [] # holds the interpolators
    for i in range(nb_color_channels):
        coord = coord_pixels[:,0:2]
        pixel = coord_pixels[:,i + 2]
        interp = CloughTocher2DInterpolator(list(zip(coord[:,0],coord[:,1])),pixel)
        interps.append(interp)

    #X_new_lims = [int(np.ceil(np.min(X_new))),int(np.ceil(np.max(X_new)))]
    #Y_new_lims = [int(np.ceil(np.min(Y_new))),int(np.ceil(np.max(Y_new)))]
    
    x_new_lims = [int(np.ceil(np.min(coord_pixels[:,0]))),int(np.ceil(np.max(coord_pixels[:,0])))]
    y_new_lims = [int(np.ceil(np.min(coord_pixels[:,1]))),int(np.ceil(np.max(coord_pixels[:,1])))]

    image_new = np.zeros((y_new_lims[1] - y_new_lims[0] ,
                          x_new_lims[1] - x_new_lims[0] ,
                          nb_color_channels),dtype = np.uint8)

    height_new,width_new = image_new.shape[0:2]
    x_new = np.arange(width_new)
    y_new = np.arange(height_new)
    
    for j in range(image_new.shape[0]):
        for i in range(image_new.shape[1]):
            for channel in range(nb_color_channels):
                try:
                    image_new[j,i,channel] = int(interps[channel](x_new[i] - width_new //2,
                                                      y_new[j] - height_new //2))
                except :
                    image_new[j,i,channel] = 0
                #print(image_new[j,i,channel],x_new[i] - width_new,y_new[j] - height_new,channel)
    return image_new


if __name__ == "__main__":
    image_path = "room.jpg"
    image = imread(image_path)
    
    def f1(x,y):
        phi = np.arctan2(y,x)
        return (1 + np.abs(np.cos(2 * phi)))/2
    ##print(image)
    m = Metric(f1,0,0,f1)
    ##print("======")
    ##print("======")
    new_image = transform_image(image,m)
    ##print(new_image)
    imsave("trf_room.png",new_image)
    


