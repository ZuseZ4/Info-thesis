import bpy
import bpy_extras
import os
import sys
import glob
import random
import math
import time
import queue
import shutil
import subprocess
import numpy as np
import time

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
    #bpy.context.scene.render.engine = 'CYCLES' #later
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.use_zbuffer = True
        
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
    assert os.path.isdir(render_path)
    
    pbr_parent_dir = os.path.join(base_dir, "SamplePBR")
    num_PBR = len(list(os.walk(pbr_parent_dir)))
    
    # Specify files
    model_files = glob.glob(model_dir + "/*.obj")
    texture_files = glob.glob(texture_dir + "/*.jpg")
    background_files = glob.glob(background_dir + "/*.hdr")

    pbr = pbr_loader(base_dir)
    print(pbr)
    
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
        obj_list = random.choices(self.model_files, k=self.n)
        
        col = bpy.data.collections.get(self.collection_name)
        if col == None:
            col = bpy.data.collections.new(self.collection_name) 
            bpy.context.scene.collection.children.link(col)
              
        while(self.num_loaded_obj < self.n):
            obj_path = obj_list[self.num_loaded_obj]
            obj_name = bpy.path.display_name_from_filepath(obj_path)
            bpy.ops.import_scene.obj(filepath=obj_path)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                
            for ob in bpy.context.selected_objects:
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
    
    def create_or_reuse_compositor(self):
        # check if nodes exist, don't create multiples
        if self.compositor_exists:
            return
        else:
            print("creating compositor")
        scene = bpy.context.scene
        scene.render.use_compositing = True
        scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes.values() # from last iteration, remvoe
        for node in nodes:
            bpy.context.scene.node_tree.nodes.remove(node)
            
        renderNode = scene.node_tree.nodes.new('CompositorNodeRLayers')
        compositeNode = scene.node_tree.nodes.new('CompositorNodeComposite')
        scene.node_tree.links.new(compositeNode.inputs['Image'], renderNode.outputs['Image'])
        scene.node_tree.links.new(compositeNode.inputs['Z'], renderNode.outputs['Depth'])
        scene.node_tree.links.new(compositeNode.inputs['Alpha'], renderNode.outputs['Alpha'])
        self.compositor_exists = True

    #def __get_bounding_box(obj):
        
    def __clamp(self, x, minimum, maximum):
        return max(minimum, min(x, maximum))

    def __camera_view_bounds_2d(self, me_ob):
        """
        Returns camera space bounding box of mesh object.

        Negative 'z' value means the point is behind the camera.

        Takes shift-x/y, lens angle and sensor size into account
        as well as perspective/ortho projections.

        :arg scene: Scene to use for frame size.
        :type scene: :class:`bpy.types.Scene`
        :arg obj: Camera object.
        :type obj: :class:`bpy.types.Object`
        :arg me: Untransformed Mesh.
        :type me: :class:`bpy.types.Mesh´
        :return: a Box object (call its to_tuple() method to get x, y, width and height)
        :rtype: :class:`Box`
        """
        scene = bpy.context.scene
        cam_ob = scene.camera

        mat = cam_ob.matrix_world.normalized().inverted()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_eval = me_ob.evaluated_get(depsgraph)
        me = mesh_eval.to_mesh()
        me.transform(me_ob.matrix_world)
        me.transform(mat)

        camera = cam_ob.data
        frame = [-v for v in camera.view_frame(scene=scene)[:3]]
        camera_persp = camera.type != 'ORTHO'

        lx = []
        ly = []

        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if camera_persp:
                if z == 0.0:
                    lx.append(0.5)
                    ly.append(0.5)
                # Does it make any sense to drop these?
                # if z <= 0.0:
                #    continue
                else:
                    frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        min_x = self.__clamp(min(lx), 0.0, 1.0)
        max_x = self.__clamp(max(lx), 0.0, 1.0)
        min_y = self.__clamp(min(ly), 0.0, 1.0)
        max_y = self.__clamp(max(ly), 0.0, 1.0)

        mesh_eval.to_mesh_clear()

        r = scene.render
        fac = r.resolution_percentage * 0.01
        dim_x = r.resolution_x * fac
        dim_y = r.resolution_y * fac

        # Sanity check
        if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
            return (0, 0, 0, 0)

        return (
            round(min_x * dim_x),            # X
            round(dim_y - max_y * dim_y),    # Y
            round((max_x - min_x) * dim_x),  # Width
            round((max_y - min_y) * dim_y)   # Height
        )

    # Print the result
    #print(camera_view_bounds_2d(context.scene, context.scene.camera, context.object))      
        
        
    def iterate_diff(self):
    
        if bpy.data.collections.get(self.collection_name) == None:
            return # we haven't run load_obj() yet
        col = bpy.data.collections.get(self.collection_name)    

        
        for f in glob.glob(os.path.join(self.render_path, "*.*")):
            os.remove(f)
        f = open(os.path.join(self.render_path, "labels.txt"), "w")

        print("samples: ", bpy.context.scene.eevee.taa_samples)
        bpy.context.scene.eevee.taa_samples = 16
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.use_zbuffer = False
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".png")
        bpy.ops.render.render(write_still = True)
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_full" + ".exr")
        bpy.ops.render.render(write_still = True)
        # the next one could improve perf, but has no effect
        #bpy.context.scene.eevee.taa_samples = 1 # depth information are exact after the first run

        keys = self.annotations.keys()
        positions = queue.Queue()
        for i, key in enumerate(keys):
            objects = self.annotations[key]
            for obj in objects:
                positions.put(obj.location.copy())
                obj.location = (100,100,100)
                
        # render empty
        #bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        #bpy.context.scene.render.image_settings.use_zbuffer = True
        #bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_empty.exr")
        #bpy.ops.render.render(write_still = True)
                
        for i, key in enumerate(keys):
            f.write(bpy.path.basename(key))
            objects = self.annotations[key]                
            for obj in objects:
                obj.location = positions.get()
                bb = self.__camera_view_bounds_2d(obj) # (x,y,width,height)
                f.write(" " + str(bb[0]) + "," + str(bb[1]) + "," + str(bb[2]) + "," + str(bb[3]))            
                
            #bpy.context.scene.render.image_settings.file_format = 'PNG'
            #bpy.context.scene.render.image_settings.use_zbuffer = False
            #bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_img_" + str(i) + ".png")
            #bpy.ops.render.render(write_still = True)
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
            bpy.context.scene.render.image_settings.use_zbuffer = True
            bpy.context.scene.render.filepath = os.path.join(self.render_path, "render_img_" + str(i) + ".exr")
            bpy.ops.render.render(write_still = True)
                    
            for obj in objects: # simply remove obj instead of hiding them again
                bpy.data.objects.remove(obj, do_unlink=True)
            f.write("\n")
        f.close()
        

                
    def create_labels(self):
        # I'm too lazy to get PIL/exr running inside of Blender
        subprocess.run(['python3', '/media/MX500/CS_BA_Data/label.py'])




loader = Obj_loader()

startTime = time.time()
for i in range(200, 210):
    print("annotations ", loader.annotations)
    loader.set_camera() # only creates one camera (by all means)
    loader.iteration(i)
    loader.load_floor()
    loader.load_obj() # creates multiple textures and materials
    loader.load_background() # only creates one bg (per bg image)
    loader.create_or_reuse_compositor()
    loader.iterate_diff()
    loader.clean()
loader.clean()
#loader.create_labels()


executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))


# mask R - CNN
# Segmantic segmentation
# Wednesday

# 730 seconds for 100 folders / scenes (without generating labels)
# 2900 s for 800 folders.

# notes: don't rotate camera - might break bounding boxes of objects (they don't adjust to camera). Instead rotate everything else