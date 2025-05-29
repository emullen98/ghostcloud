from _nlc_image_utils import image_to_binary_array, corr_func

file_name = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/useful/2012-12-30--03-56-29--421_id=1.png'

my_arr = image_to_binary_array(file=file_name, thresh=100, fill_holes=True, label_clusters=True)[3]

corr_func(labeled_lattice=my_arr, frac=0.01)
