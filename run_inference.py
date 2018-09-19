import os
directory = "/home/arnab/tinkering-projects/data/extract_output/original_images"
output_base = "/home/arnab/tinkering-projects/data/extract_output/"
for dir in os.listdir(directory):
    rgb_folder = os.path.join( directory, dir )
    output_folder = os.path.join( output_base, "masks" , dir )
    output_images = os.path.join( output_base, "images" , dir )
    output_edges = os.path.join( output_base, "scribbles" , dir )

    cmd = "python inference.py --rgb_folder "+ rgb_folder + " --output_folder  " + output_folder + " --output_image_folder  " + output_images + " --output_edge_folder  " + output_edges
    print(cmd)
    os.system(cmd)
