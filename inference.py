import cv2
import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
import sys
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
output_folder = "./test_output_alpha"
output_image_folder = "./test_output_images"
output_edge_folder = "./test_output_edges"

def rgba2rgb(img):
	return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def get_contour_only(image,color=(0,255,0)):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j,0]== color[0] and image[i,j,1]== color[1] and image[i,j,2]== color[2]:
                image[i,j]=(0,0,0)
            else:
                 image[i,j]=(255,255,255)
    return image

def draw_contour(img,color=(0,255,0)):
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	img = cv2.drawContours(img, contours, -1, color , 3)
	return img

def main(args):
        output_folder = args.output_folder
        output_image_folder = args.output_image_folder
        output_edge_folder = args.output_edge_folder
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

        if not os.path.exists(output_image_folder):
		os.mkdir(output_image_folder)

        if not os.path.exists(output_edge_folder):
		os.mkdir(output_edge_folder)


	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_fraction)
	with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
		saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
		image_batch = tf.get_collection('image_batch')[0]
		pred_mattes = tf.get_collection('mask')[0]

		if args.rgb_folder:
			rgb_pths = os.listdir(args.rgb_folder)
			for rgb_pth in rgb_pths:
				rgb = misc.imread(os.path.join(args.rgb_folder,rgb_pth))
                                if args.edge_folder:
                                    edge = misc.imread(os.path.join(args.edge_folder,rgb_pth.replace('jpg','png')))
				if rgb.shape[2]==4:
					rgb = rgba2rgb(rgb)
				origin_shape = rgb.shape
				rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

				feed_dict = {image_batch:rgb}
				pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
				final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
				image = cv2.imread(os.path.join(args.rgb_folder,rgb_pth))
				final_alpha=final_alpha/255.0
				if args.edge_folder:
					edge = cv2.imread(os.path.join(args.edge_folder,rgb_pth.replace('jpg','png')))
				for i in range(origin_shape[0]):
					for j in range(origin_shape[1]):
						if final_alpha[i,j]<0.2 and args.edge_folder :
							edge[i,j]=[255,255,255]
						if final_alpha[i,j]>=0.2:
						    final_alpha[i,j]=1
                                                else:
                                                    final_alpha[i,j]=0
                                                    image[i,j] = [255,255,255]
				cv2.imwrite(os.path.join(output_image_folder,rgb_pth),image)
				misc.imsave(os.path.join(output_folder,rgb_pth),final_alpha)
				final_alpha_cv2= cv2.imread( os.path.join(output_folder,rgb_pth) ) #cv2.imread(os.path.join(args.rgb_folder,rgb_pth) )
				final_alpha_contours=draw_contour(final_alpha_cv2)
				final_alpha_contours_only=get_contour_only(final_alpha_contours)
				if args.edge_folder:
					cv2.imwrite(os.path.join(output_edge_folder,rgb_pth.replace('jpg', 'png')),edge)
				else:
					cv2.imwrite(os.path.join(output_edge_folder,rgb_pth.replace('jpg', 'png')),final_alpha_contours_only)
		else:
			rgb = misc.imread(args.rgb)
			if rgb.shape[2]==4:
				rgb = rgba2rgb(rgb)
			origin_shape = rgb.shape[:2]
			rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

			feed_dict = {image_batch:rgb}
			pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
			final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
			misc.imsave(os.path.join(output_folder,'alpha.png'),final_alpha)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--rgb', type=str,
		help='input rgb',default = None)
	parser.add_argument('--rgb_folder', type=str,
		help='input rgb',default = None)
    	parser.add_argument('--edge_folder', type=str,
		help='input rgb',default = None)
        parser.add_argument('--gpu_fraction', type=float,
		help='how much gpu is needed, usually 4G is enough',default = 1.0)
        parser.add_argument('--output_folder', type  = str, default ="./test_output_alpha" )
        parser.add_argument('--output_image_folder', type  = str, default ="./test_output_images" )
        parser.add_argument('--output_edge_folder', type  = str, default ="./test_output_edges" )

	return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
