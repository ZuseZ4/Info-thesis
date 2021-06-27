import bpy
import os
import sys
import glob
import random
import math
import time
import queue
import shutil
import subprocess

dir = "/media/MX500/CS_BA_Data"
if not dir in sys.path:
    sys.path.append(dir)
print("path: ", sys.path)
import loadPBR

# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(loadPBR)

# this is optional and allows you to call the functions without specifying the package name
from loadPBR import *


class Obj_loader:
    compositor_exists = False
    n = 10 # will load n random objects
    square_root = math.ceil(math.sqrt(n))
    square = square_root**2 # next larger square number
    num_loaded_obj = 0
    annotations = dict()

    # Model directory
    base_dir = dir
    assert os.path.isdir(base_dir)
    
    collection_name = "lab_obj_collection"
    
    #model_dir = os.path.join(base_dir,'small','models')
    model_dir = os.path.join(base_dir,'large','models')
    texture_dir = os.path.join(base_dir,'large','textures')
    background_dir = os.path.join(base_dir, 'backgrounds')
    tex_name = os.path.join(base_dir, 'small','textures',"0ab478614f29657a.jpg")
    picture_path = os.path.join(base_dir, "cam_out.jpg")
    render_path = os.path.join(base_dir, "render_output")
    
    assert os.path.isfile(tex_name)
    assert os.path.isdir(texture_dir)
    assert os.path.isdir(model_dir)
    assert os.path.isdir(background_dir)
    
    pbr_parent_dir = os.path.join(base_dir, "SamplePBR")
    num_PBR = len(list(os.walk(pbr_parent_dir)))
    
    # Specify files
    model_files = glob.glob(model_dir + "/*.obj")
    texture_files = glob.glob(texture_dir + "/*.jpg")
    background_files = glob.glob(background_dir + "/*.hdr")

    pbr = pbr_loader(base_dir)
    print(pbr)
    
    def __load_or_reuse(self, img_path):
        img_name = bpy.path.display_name_from_filepath(img_path)
        img = bpy.data.images.get(img_name)
        if img == None:
            img = bpy.data.images.load(img_path)
        return img
    
    def load_obj(self):
        obj_list = random.choices(self.model_files, k=self.n)
        print("samples", obj_list)
        
        col = bpy.data.collections.get(self.collection_name)
        if col == None:
            col = bpy.data.collections.new(self.collection_name) 
            bpy.context.scene.collection.children.link(col)
              
        while(self.num_loaded_obj < self.n):
            obj_path = obj_list[self.num_loaded_obj]
            print(obj_path)
            obj_name = bpy.path.display_name_from_filepath(obj_path)
            bpy.ops.import_scene.obj(filepath=obj_path)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                
            for ob in bpy.context.selected_objects:
                #print("ob:",obj_name)
                col.objects.link(ob)
            
                # Add texture
                applied_mat_name = self.pbr.apply_random(ob, self.num_loaded_obj)
                self.annotations.setdefault(applied_mat_name, []) # add key only if not inside
                #self.annotations[applied_mat_name].append(self.num_loaded_obj)
                self.annotations[applied_mat_name].append(ob)
                
                ob.dimensions = (1,1,1)
                ob.rotation_euler = [0,0,math.radians(random.randint(0,360))]
                ob.location = (random.uniform(-5., 5.), random.uniform(-5.,5.), 1)
            self.num_loaded_obj += 1
        
    def load_background(self):
        bg_path = random.choice(self.background_files)
        
        cam = bpy.context.scene.camera
        if cam == None:
            self.set_camera()
            cam = bpy.context.scene.camera
            
        cam.data.show_background_images = True
        cam.data.background_images.clear()
        
        bg = cam.data.background_images.new()
        bg.image = self.__load_or_reuse(bg_path)
    
    def load_floor(self):
        cube = bpy.data.objects.get("ground")
        if cube != None:
            return
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.selected_objects[0]
        cube.name = "ground"
        cube.location = (0,0,0)
        cube.dimensions = (10,10,0.)

    def set_camera(self):
        if bpy.context.scene.camera == None:
            print("create new scene camera")
            if bpy.data.cameras.get('Camera') == None:
                print("creater new camera")
                camera_data = bpy.data.cameras.new(name='Camera')
            else:
                camera_data = bpy.data.cameras.get('Camera')
            cam = bpy.data.objects.new('Camera', camera_data)
            bpy.context.scene.camera = cam
            bpy.context.scene.collection.objects.link(cam)
        else:
            cam = bpy.context.scene.camera
            
        if bpy.context.scene.collection.objects.get('Camera') == None:
            bpy.context.scene.collection.objects.link(cam)
        
        grid_width = self.square_root
        #cam_loc = (grid_width / 2, grid_width / 2, 10)
        cam_loc = (0,0,30)
        cam = bpy.context.scene.camera
        bpy.context.scene.camera = cam
        cam.location = cam_loc
        cam.rotation_euler = [0,0,0]
        
    def clean(self):
        bpy.data.meshes.data.orphans_purge() # delte unused meshes (they accumulate)
        bpy.data.materials.data.orphans_purge() # delte unused materials (they accumulate)
        bpy.data.images.data.orphans_purge()
        if bpy.data.collections.get(self.collection_name) == None:
            return # we haven't run load_obj() yet
        obj = bpy.data.collections.get(self.collection_name).objects
        while obj:
            bpy.data.objects.remove(obj[0], do_unlink=True)
    
    def create_or_reuse_compositor(self):
        # check if nodes exist, don't create multiples
        if self.compositor_exists:
            return
        else:
            print("creating compositor")
        scene = bpy.context.scene
        print("scene", scene)
        scene.render.use_compositing = True
        scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes.values() # from last iteration, remvoe
        for node in nodes:
            bpy.context.scene.node_tree.nodes.remove(node)
            
        renderNode = scene.node_tree.nodes.new('CompositorNodeRLayers')
        normalizeNode = scene.node_tree.nodes.new('CompositorNodeNormalize')
        compositeNode = scene.node_tree.nodes.new('CompositorNodeComposite')
        OutputNode = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        scene.node_tree.links.new(normalizeNode.inputs['Value'], renderNode.outputs['Depth'])
        scene.node_tree.links.new(compositeNode.inputs['Image'], normalizeNode.outputs['Value'])
        scene.node_tree.links.new(compositeNode.inputs['Alpha'], renderNode.outputs['Alpha'])
        self.compositor_exists = True
        
    def iterate_diff(self):
        if os.path.isdir(self.render_path):
            shutil.rmtree(self.render_path)
        os.mkdir(self.render_path)

        # TODO: iterate self.annotations
        col = bpy.data.collections.get(self.collection_name).objects
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".png")
        bpy.ops.render.render(write_still = True)
        print(len(col))
        keys = self.annotations.keys()
        for i, key in enumerate(keys):
            objects = self.annotations[key]
            print("iterate diff: ", key, objects)
            positions = queue.Queue()
            for obj in objects:
                positions.put(obj.location.copy())
                obj.location = (100,100,100)
            bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_img_" + str(i) + ".png")
            bpy.ops.render.render(write_still = True)
            for obj in objects:
                obj.location = positions.get()
                
    def create_labels(self):
        # I'm too lazy to get PIL running inside of Blender
        subprocess.run(['python3', '/media/MX500/CS_BA_Data/label.py'])

loader = Obj_loader()
loader.clean()
loader.load_floor()
loader.load_obj() # creates multiple textures and materials
loader.set_camera() # only creates one camera (by all means)
loader.load_background() # only creates one bg (per bg image)
loader.create_or_reuse_compositor()
loader.iterate_diff()
loader.create_labels()