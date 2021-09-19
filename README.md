The main file is loadScene.py
Open it inside of Blender, adjust input and output directory and set n and m.
It will then create the folders n, n+1, ..., m inside of the output directory. 
Each subfolder will contain one rendered scene with related files.

In order to generate the training data please use process_exr.py
It will require the paths to your pbr_dir, your output_dir from above (the one containing the subfolders with the scenes) and it will require a 
third directory where it will store all the extracted images and masks.

I recommend using one output directory and one training directory per blender instance to make sure that they won't overwrite their outputs.
