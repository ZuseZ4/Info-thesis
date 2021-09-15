import bpy
import bpy_extras
import os
import sys
import glob
import random
import math
from mathutils import Vector
import time
import queue
import shutil
import subprocess
import numpy as np
import time

dir = "/home/zuse/prog/CS_BA_Data"
if not dir in sys.path:
    sys.path.append(dir)
print("path: ", sys.path)
import loadPBR
import VesselGenerator

# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(loadPBR)
imp.reload(VesselGenerator)

# this is optional and allows you to call the functions without specifying the package name
from loadPBR import *
from VesselGenerator import *

class Obj_loader:
    bpy.context.scene.render.engine = 'CYCLES' #later
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.feature_set = "SUPPORTED"
    bpy.context.scene.render.film_transparent = False
    
    #bpy.context.scene.render.film_transparent = True
    #bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    #bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    #bpy.context.scene.render.image_settings.use_zbuffer = True
        
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
    
    model_dir = os.path.join(base_dir,'large','models')
    background_dir = os.path.join(base_dir, 'backgrounds')
    render_path = os.path.join(base_dir, "render_output")
    pbr_parent_dir = os.path.join(base_dir, "SamplePBR")
    
    assert os.path.isdir(pbr_parent_dir)
    assert os.path.isdir(model_dir)
    assert os.path.isdir(background_dir)
    assert os.path.isdir(render_path)
    
    # Specify files
    num_PBR = len(list(os.walk(pbr_parent_dir)))
    model_files = glob.glob(model_dir + "/*.gltf")
    background_files = glob.glob(background_dir + "/*.hdr")
    pbr = pbr_loader(base_dir)
    
    print("#models: ", len(model_files))
    print("pbr: ", pbr)
    
    
    
    def iteration(self, n):
        self.render_path = os.path.join(self.base_dir, "render_output", str(n))
        if not os.path.isdir(self.render_path):
            os.mkdir(self.render_path)
    
    def __load_or_reuse(self, img_path):
        img_name = bpy.path.basename(img_path)
        img = bpy.data.images.get(img_name)
        if img == None:
            print("creating", img_name)
            img = bpy.data.images.load(img_path)
        return img
    
    def load_obj(self):
        #num_vessels = np.random.randint(0, self.n / 2) # we currently don't use them
        num_vessels = 0 # don't crash my pc please
        num_objects = self.n - num_vessels
        obj_list = random.choices(self.model_files, k=num_objects)
        
        col = bpy.data.collections.get(self.collection_name)
        if col == None:
            col = bpy.data.collections.new(self.collection_name) 
            bpy.context.scene.collection.children.link(col)
        assert(col != None, "col is None")      
        for obj_path in obj_list:
            obj_name = bpy.path.display_name_from_filepath(obj_path)
            bpy.ops.import_scene.gltf(filepath=obj_path)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                
            for ob in bpy.context.selected_objects:
                col.objects.link(ob)
            
                # Add texture
                applied_mat_name = self.pbr.apply_random(ob, self.num_loaded_obj)
                self.annotations.setdefault(applied_mat_name, []) # add key only if not inside
                self.annotations[applied_mat_name].append(ob)
            
                scale = random.uniform(0.7, 2)
                ob.dimensions = (scale, scale, scale)
                ob.rotation_mode = "XYZ"
                ob.rotation_euler = (math.radians(random.randint(0,360)),math.radians(random.randint(0,360)),math.radians(random.randint(0,360)))
                print(ob.rotation_euler)
                ob.location = (random.uniform(-5., 5.), random.uniform(-5.,5.), random.uniform(-3.,3.))
            self.num_loaded_obj += 1
        print("vessels: ", num_vessels)
        for i in range(num_vessels):
            vessel_name = "Vessel_"+str(i)
            content_name = "Content_"+str(i)
            print(vessel_name)
            
            # create objects
            AddVessel(VesselName=vessel_name,ContentName=content_name,Col=col,ScaleFactor=np.random.rand()*0.01 + 0.02)
            vessel = bpy.data.objects.get(vessel_name)
            content = bpy.data.objects.get(content_name)
            assert(vessel != None)
            assert(content != None)
            
            # apply textures
            #applied_mat_name = self.pbr.apply_random(vessel, self.num_loaded_obj)
            #self.pbr.apply_mat_to_obj(bpy.data.materials.get("mat-"+str(self.num_loaded_obj)), content)
            print(vessel_name)
            VesselMaterial=AssignMaterialToVessel(vessel_name)
            #AssignMaterialToVessel(vessel_name)
            #self.annotations.setdefault(applied_mat_name, []) # add key only if not inside

            # move objects to random position
            #self.annotations[applied_mat_name].append(vessel)
            #self.annotations[applied_mat_name].append(content)
            v = Vector((random.uniform(-5., 5.), random.uniform(-5., 5.), 0))
            vessel.location += v
            content.location += v
            self.num_loaded_obj += 1
            
            
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
        scene = bpy.context.scene
        if scene.camera == None:
            print("create new scene camera")
            if bpy.data.cameras.get('Camera') == None:
                print("creater new camera")
                camera_data = bpy.data.cameras.new(name='Camera')
            else:
                camera_data = bpy.data.cameras.get('Camera')
            cam = bpy.data.objects.new('Camera', camera_data)
            scene.camera = cam
            scene.collection.objects.link(cam)
        else:
            cam = scene.camera
            
        if scene.collection.objects.get('Camera') == None:
            scene.collection.objects.link(cam)
        
        grid_width = self.square_root
        #cam_loc = (grid_width / 2, grid_width / 2, 10)
        cam_loc = (0,0,28)
        cam = scene.camera
        scene.camera = cam
        cam.location = cam_loc
        cam.rotation_euler = [0,0,0]
    ##############################################################################################################################

    #                   Add Camera to scene

    ##############################################################################################################################  
    def SetCamera(self, name="Camera", lens = 32, location=(0,0,0),rotation=(0, 0, 0),shift_x=0,shift_y=0):
       
        #=================Set Camera================================
       
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
        bpy.context.object.name = name    
        bpy.context.object.data.lens = lens

        bpy.context.scene.camera = bpy.context.object
        bpy.context.scene.camera.location=location
        bpy.context.scene.camera.rotation_euler=rotation
        bpy.context.scene.camera.data.type = 'PERSP'
        bpy.context.scene.camera.data.shift_x=shift_x
        bpy.context.scene.camera.data.shift_y=shift_y

    ########################################################################################################

    # Randomly set camera position (so it will look at the vessel)

    #################################################################
    def RandomlySetCameraPos(self,name,MinDist):
        print("Randomly set camera position")
        #MinDist=np.max([VesWidth,VesHeight])
        R=np.random.rand()*MinDist*4+MinDist*3  # Random radius (set range that give good images)
        print('R='+str(R)+"  MinDist="+str(MinDist))
        Ang=(1.0-1.1*np.random.rand()*np.random.rand())*3.14/2  # Random angle
        x=0
        y=np.sin(Ang)*R #+np.random.rand()*VesWidth-VesWidth/2
        z=np.cos(Ang)*R #+VesHeight*np.random.rand()
        rotx=Ang
        rotz=3.14159
        roty=(0.5*np.random.rand()-0.5*np.random.rand())*np.random.rand() # Random roll
        Focal=50 #(np.random.rand()*5+2)*R/np.max([VesWidth,VesHeight])
        shift_x=0.2-np.random.rand()*0.4
        shift_y=0.2-np.random.rand()*0.4
        self.SetCamera(name="Camera", lens = Focal, location=(x,y,z),rotation=(rotx, roty, rotz),shift_x=shift_x,shift_y=shift_y)
        
        
    def clean(self):
        if bpy.data.collections.get(self.collection_name) == None:
            return # we haven't run load_obj() yet
        col = bpy.data.collections.get(self.collection_name)    
        obj = col.objects
        while obj:
            bpy.data.objects.remove(obj[0], do_unlink=True)
        bpy.data.collections.remove(col)
        
        self.annotations = {}
        self.num_loaded_obj = 0
        
        bpy.data.meshes.data.orphans_purge() # delte unused meshes (they accumulate)
        bpy.data.materials.data.orphans_purge() # delte unused materials (they accumulate)
        bpy.data.images.data.orphans_purge()
    
    def set_background(self):
        bg_path = random.choice(self.background_files)
        print("Background: ", bg_path)
        assert(bg_path != None, "Couldn't find background!")
        bg_img = self.__load_or_reuse(bg_path)
        
        tree_nodes  = bpy.context.scene.world.node_tree.nodes
        node_environment = tree_nodes.get("Environment Texture") # Add Environment Texture node
        node_environment.image = bg_img
        print("background size: ", node_environment.width, node_environment.height)
        node_environment.location = -300,0
    
    def create_or_reuse_compositor(self):
        # check if nodes exist, don't create multiples
        if self.compositor_exists:
            return
        else:
            print("creating compositor")
        scene = bpy.context.scene
        scene.render.use_compositing = True
        scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes.values() # from last iteration, remove
        for node in nodes:
            bpy.context.scene.node_tree.nodes.remove(node)
            
        renderNode = scene.node_tree.nodes.new('CompositorNodeRLayers')
        compositeNode = scene.node_tree.nodes.new('CompositorNodeComposite')

        scene.node_tree.links.new(compositeNode.inputs['Image'], renderNode.outputs['Image'])
        scene.node_tree.links.new(compositeNode.inputs['Z'], renderNode.outputs['Depth'])
        self.compositor_exists = True
        
        node_tree  = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes
        tree_nodes.clear() # Clear all nodes
        node_background = tree_nodes.new(type='ShaderNodeBackground') # Add Background node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment') # Add Environment Texture node
        #node_environment.image = bpy.data.images.load(bg_path) 
        #node_environment.location = -300,0
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld') # Add Output node
        #node_output.location = 200,0

        # Link all nodes
        links = node_tree.links
        link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

        
        
    def iterate_diff(self):
    
        if bpy.data.collections.get(self.collection_name) == None:
            print("couldn't find ", self.collection_name)
            sys.exit(1)
            return # we haven't run load_obj() yet
        col = bpy.data.collections.get(self.collection_name)
        
        for f in glob.glob(os.path.join(self.render_path, "*.*")):
            os.remove(f)        
        
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.use_zbuffer = False
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".png")
        bpy.ops.render.render(write_still = True)
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".exr")
        bpy.ops.render.render(write_still = True)

        keys = self.annotations.keys()
        positions = queue.Queue()
        for i, key in enumerate(keys):
            objects = self.annotations[key]
            for obj in objects:
                positions.put(obj.location.copy())
                obj.location = (1000,1000,1000)
                
        # render empty
        #bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        #bpy.context.scene.render.image_settings.use_zbuffer = True
        #bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_empty.exr")
        #bpy.ops.render.render(write_still = True)
                
        for i, key in enumerate(keys):
            objects = self.annotations[key]          
            num = 0      
            for obj in objects:
                obj.location = positions.get()   

                bpy.context.scene.render.image_settings.file_format = 'PNG'
                bpy.context.scene.render.image_settings.use_zbuffer = False
                bpy.context.scene.render.filepath = os.path.join(self.render_path, key.split("/")[-1] + "_" + str(num) + ".png")
                bpy.ops.render.render(write_still = True)
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
                bpy.context.scene.render.image_settings.use_zbuffer = True
                bpy.context.scene.render.filepath = os.path.join(self.render_path, key.split("/")[-1] + "_" + str(num) + ".exr")
                bpy.context.scene.render.filepath = os.path.join(self.render_path, key.split("/")[-1] + "_" + str(num) + ".exr")
                bpy.ops.render.render(write_still = True)
                num += 1
                
                #for obj in objects: # simply remove obj instead of hiding them again
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_empty.exr")
        bpy.ops.render.render(write_still = True)


        

#CleanScene()
loader = Obj_loader()

startTime = time.time()
loader.create_or_reuse_compositor()
for i in range(0,20):
    loader.clean()
    #CleanScene()
    loader.set_background()
    #loader.set_camera() # only creates one camera (by all means)
    loader.RandomlySetCameraPos("Camera", 6)
    loader.iteration(i)
    #loader.load_floor()
    loader.load_obj() # creates multiple textures and materials
    loader.iterate_diff()
loader.clean()
CleanScene()
#loader.clean()
#loader.create_labels()


executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))


# mask R - CNN
# Segmantic segmentation
# Wednesday

# 730 seconds for 100 folders / scenes (without generating labels)
# 2900 s for 800 folders.

# notes: don't rotate camera - might break bounding boxes of objects (they don't adjust to camera). Instead rotate everything else