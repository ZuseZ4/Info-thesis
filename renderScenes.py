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

# dir = "/home/sf3203tr4/hdd/ManuelDataset/"
dir = "/home/zuse/prog/CS_BA_Data"
if not dir in sys.path:
    sys.path.append(dir)
print("path: ", sys.path)
import loadPBR
#import VesselGenerator

# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(loadPBR)
#imp.reload(VesselGenerator)

# this is optional and allows you to call the functions without specifying the package name
from loadPBR import *
#from VesselGenerator import *

class Obj_loader:
    def __init__(self, gltf_dir, pbr_dir, hdri_dir, output_dir):    
        bpy.context.scene.render.engine = 'CYCLES' #later
        bpy.context.scene.cycles.samples = 900
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.feature_set = "SUPPORTED"
        bpy.context.scene.render.film_transparent = False
        
        #bpy.context.scene.render.film_transparent = True
        #bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        #bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        #bpy.context.scene.render.image_settings.use_zbuffer = True
            
        self.compositor_exists = False
        self.n = 10 # will load n random objects
        self.num_loaded_obj = 0
        self.annotations = dict()

        self.collection_name = "lab_obj_collection"
        
        self.model_dir = gltf_dir
        self.background_dir = hdri_dir
        self.pbr_parent_dir = pbr_dir
        
        self.render_base_dir = output_dir 
        
        assert os.path.isdir(self.pbr_parent_dir)
        assert os.path.isdir(self.model_dir)
        assert os.path.isdir(self.background_dir)
        assert os.path.isdir(self.render_base_dir)
        
        # Specify files
        self.num_PBR = len(list(os.walk(self.pbr_parent_dir)))
        self.model_files = glob.glob(self.model_dir + "/*.gltf")
        self.background_files = glob.glob(self.background_dir + "/*.hdr")
        self.pbr = pbr_loader(self.pbr_parent_dir)
        
        print("#models: ", len(self.model_files))
        print("pbr: ", self.pbr)
        
    
    
    def iteration(self, n):
        self.render_path = os.path.join(self.render_base_dir, str(n))
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
                applied_mat_name, _ = self.pbr.apply_random(ob, self.num_loaded_obj)
                self.annotations.setdefault(applied_mat_name, []) # add key only if not inside
                self.annotations[applied_mat_name].append(ob)
            
                scale = random.uniform(0.7, 3)
                ob.dimensions = (scale, scale, scale)
                ob.rotation_mode = "XYZ"
                ob.rotation_euler = (math.radians(random.randint(0,360)),math.radians(random.randint(0,360)),math.radians(random.randint(0,360)))
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
            print(vessel_name)
            VesselMaterial=AssignMaterialToVessel(vessel_name)
            AssignMaterialToContent(content_name, self.pbr, self.num_loaded_obj)
            #AssignMaterialToVessel(vessel_name)
            #self.annotations.setdefault(applied_mat_name, []) # add key only if not inside

            # move objects to random position
            #self.annotations[applied_mat_name].append(vessel)
            #self.annotations[applied_mat_name].append(content)
            v = Vector((random.uniform(-5., 5.), random.uniform(-5., 5.), 0))
            vessel.location += v
            content.location += v
            self.num_loaded_obj += 1
            
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
        location = (x,y,z)
        rotation = (rotx, roty, rotz)
       
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
        bpy.context.object.name = name    
        bpy.context.object.data.lens = 50 #(np.random.rand()*5+2)*R/np.max([VesWidth,VesHeight])

        bpy.context.scene.camera = bpy.context.object
        bpy.context.scene.camera.location=location
        bpy.context.scene.camera.rotation_euler=rotation
        bpy.context.scene.camera.data.type = 'PERSP'
        bpy.context.scene.camera.data.shift_x=0.2-np.random.rand()*0.4
        bpy.context.scene.camera.data.shift_y=0.2-np.random.rand()*0.4
        
        
    def cleanAll(self):
        bpy.data.materials.data.orphans_purge()
        bpy.data.images.data.orphans_purge()
        
        
    ###############################################################################################################################

    ##==============Clean secene remove  all objects currently on the schen============================================

    ###############################################################################################################################

    def CleanScene(self):
        self.annotations = {}
        self.num_loaded_obj = 0
        
        for bpy_data_iter in (
                bpy.data.objects,
                bpy.data.meshes,
                bpy.data.cameras,
        ):
            for id_data in bpy_data_iter:
                bpy_data_iter.remove(id_data)
        print("=================Cleaning scene deleting all objects==================================")     
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        allMeshes=[]
        for mes in bpy.data.meshes:
           allMeshes.append(mes)
           print("Deleting Mesh:")
           print(mes)
        for mes in allMeshes:
            bpy.data.meshes.remove(mes)
        
    
    def set_background(self):
        bg_path = random.choice(self.background_files)
        print("Background: ", bg_path)
        assert(bg_path != None, "Couldn't find background!")
        bg_img = self.__load_or_reuse(bg_path)
        
        node_environment = bpy.context.scene.world.node_tree.nodes.get("Environment Texture") # Add Environment Texture node
        node_environment.image = bg_img
        print("background size: ", node_environment.width, node_environment.height)
    
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
        compositeNode.use_alpha = False
        blackNode = scene.node_tree.nodes.new('CompositorNodeRGB') # for depth render passes, so we can hide the scene
        blackNode.color = (255,255,255)
        self.renderNode = renderNode
        self.blackNode  = blackNode
        self.compositeNode = compositeNode

        scene.node_tree.links.new(compositeNode.inputs['Image'], renderNode.outputs['Image'])
        scene.node_tree.links.new(compositeNode.inputs['Z'], renderNode.outputs['Depth'])
        self.compositor_exists = True
        
        node_tree  = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes
        tree_nodes.clear() # Clear all nodes
        node_background = tree_nodes.new(type='ShaderNodeBackground') # Add Background node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment') # Add Environment Texture node
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld') # Add Output node

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

        # Not working optimizations
        # Connecting the scene to the renderer such that we render the scene
        # bpy.context.scene.cycles.max_bounces = 12
        # bpy.context.scene.node_tree.links.new(self.compositeNode.inputs['Image'], self.renderNode.outputs['Image'])
        # bpy.context.scene.render.resolution_percentage = 100
        
        bpy.context.scene.cycles.samples = 900 # for the real image
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.use_zbuffer = False
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".png")
        bpy.ops.render.render(write_still = True)
        
        # Not working optimizations        
        # Now we hide the scene to speed up the collection of depth informations
        # bpy.context.scene.node_tree.links.new(self.compositeNode.inputs['Image'], self.blackNode.outputs['RGBA'])
        # bpy.context.scene.cycles.max_bounces = 0
        # bpy.context.scene.render.resolution_percentage = 1
        
        # We only need the background image for our primary rgb image, not for collecting depth infos. It's compute heavy, so drop it.
        node_environment = bpy.context.scene.world.node_tree.nodes.get("Environment Texture")
        node_environment.image = None
        
        bpy.context.scene.cycles.samples = 1 # We only need depth info for the rest
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
                
        for i, key in enumerate(keys):
            objects = self.annotations[key]          
            num = 0
            for obj in objects:
                obj.location = positions.get()   

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



def main():
    
    # input_dir = str(sys.argv[1])
    # output_dir = str(sys.argv[2])
    # n = int(sys.argv[3])
    # m = int(sys.argv[4])

    if True:
        gltf_dir   = "/home/zuse/prog/CS_BA_Data/large/models"
        pbr_dir    = "/home/zuse/prog/CS_BA_Data/SamplePBR"
        hdri_dir   = "/home/zuse/prog/CS_BA_Data/backgrounds"
        output_dir = "/home/zuse/prog/CS_BA_Data/render_output"
    else:
        gltf_dir   = "/home/sf3203tr4/hdd/VirtualDataSet/Data/ObjectGTLF" 
        pbr_dir    = "/home/sf3203tr4/hdd/VirtualDataSet/Data/2K_PBR"
        hdri_dir   = "/home/sf3203tr4/Desktop/Cloud/HDRI_POLYHAVEN/Poly Haven Assets/HDRIs/4k"
        output_dir = "/home/sf3203tr4/hdd/ManuelDataset/render_output" 
    n = 0
    m = 2 # 1000

    loader = Obj_loader(gltf_dir, pbr_dir, hdri_dir, output_dir)

    startTime = time.time()

    loader.create_or_reuse_compositor()
    for i in range(n,m):
        loader.CleanScene()
        loader.cleanAll()
        loader.set_background()
        loader.RandomlySetCameraPos("Camera", 6)
        loader.iteration(i)
        loader.load_obj() # creates multiple textures and materials
        loader.iterate_diff()
    loader.CleanScene()
    loader.cleanAll()

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == "__main__":
    main()
