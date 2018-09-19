import os
from scipy import misc
from skimage.transform import resize
folder = './../data/scribble_clean_new/scribbles/soccer'
output_folder = './clean_soccer_scribbles'
#print(os.path.exists(folder))
#folder = './../pytorch-hed/cartoon_portrait'
#output_folder = './../pytorch-hed/cartoon_portrait_clean'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

counter=0
for path,subdirs,files in os.walk(folder):
    for name in sorted(files):
        img = misc.imread( os.path.join(path,name), mode='RGBA' )
        img = resize(img,(256,256))
        misc.imsave(os.path.join(output_folder,str(counter) + '.png'),img)
        counter=counter+1
