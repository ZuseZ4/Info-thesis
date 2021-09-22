import bpy
import os
import sys
import glob
import random
import math
  
class pbr_loader:
    def __init__(self, pbr_parent_path):
        assert os.path.isdir(pbr_parent_path)
        self.pbr_dirs = [f.path for f in os.scandir(pbr_parent_path) if f.is_dir()]


    def __load_or_reuse(self, img_path):
        img_name = bpy.path.basename(img_path)
        img = bpy.data.images.get(img_name)
        if img == None:
            print("creating PBR", img_name)
            img = bpy.data.images.load(img_path)
        return img

    def __add_normalNode(self, mat, normal_path):
        self.normal = True
        normalMap = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        normalMap.inputs[0].default_value = 8.0
        normalNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
        normalNode.image = self.__load_or_reuse(normal_path)
        normalNode.image.colorspace_settings.name = 'Non-Color'
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(normalMap.inputs['Color'], normalNode.outputs['Color'])
        mat.node_tree.links.new(bsdf.inputs['Normal'], normalMap.outputs['Normal'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(normalNode.inputs['Vector'], scaleNode.outputs['Vector'])
                
    def __add_colorNode(self, mat, tex_path):
        self.color = True
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = self.__load_or_reuse(tex_path)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(texImage.inputs['Vector'], scaleNode.outputs['Vector'])
    
    # Displacement ignores bsdf, as it's better added directly to the material output node
    def __add_displacementNode(self, mat, displacement_path):
        self.displacement = True
        displacementMap = mat.node_tree.nodes.new('ShaderNodeDisplacement')
        displacementNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
        displacementNode.image = self.__load_or_reuse(displacement_path)
        displacementNode.image.colorspace_settings.name = 'Non-Color'
        mat_output = mat.node_tree.nodes.get("Material Output")
        mat.node_tree.links.new(displacementMap.inputs['Height'], displacementNode.outputs['Color'])
        mat.node_tree.links.new(mat_output.inputs['Displacement'], displacementMap.outputs['Displacement'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(displacementNode.inputs['Vector'], scaleNode.outputs['Vector'])
    
    def __add_roughnessNode(self, mat, roughness_path):
        self.rough = True
        roughnessImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        roughnessImage.image = self.__load_or_reuse(roughness_path)
        roughnessImage.image.colorspace_settings.name = "Non-Color"
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Roughness'], roughnessImage.outputs['Color'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(roughnessImage.inputs['Vector'], scaleNode.outputs['Vector'])

    def __add_metallic(self, mat, metal_path):
        self.metal = True
        metalImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        metalImage.image = self.__load_or_reuse(metal_path)
        metalImage.image.colorspace_settings.name = "Non-Color"
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Metallic'], metalImage.outputs['Color'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(metalImage.inputs['Vector'], scaleNode.outputs['Vector'])


    def __add_scaleNode(self, mat):
        foo = mat.node_tree.nodes.new('ShaderNodeMapping')
        bar = mat.node_tree.nodes.new('ShaderNodeTexCoord')
        
        foo.inputs[3].default_value = (0.2, 0.2, 0.2) # would depend on texture / obj combination, but good enough in avg.
        
        mat.node_tree.links.new(foo.inputs['Vector'], bar.outputs['UV'])
       
        
    def apply_mat_to_obj(self, mat, obj):
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            for i in range(0,len(obj.data.materials.values())):
                obj.data.materials[i] = mat

    def load_mat_from_folder(self, pbr_dir, num):
        base_name = os.path.basename(pbr_dir).split("-JPG")[0]
        material_name = 'mat-' + base_name
        
        pbr_mat = bpy.data.materials.get(material_name) 
        if pbr_mat != None: # Did we already create it?
            #print("Reusing material:", material_name)
            return pbr_mat
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        
        self.__add_scaleNode(mat)
        self.metal = False

        for Fname in os.listdir(pbr_dir):
            lower_fname = Fname.lower()
            file_path = os.path.join(pbr_dir, Fname)

            if ("color." in lower_fname) or ("ao." in lower_fname):
                self.__add_colorNode(mat, file_path)
            elif ("roughness." in lower_fname) or ("rough." in lower_fname):
                self.__add_roughnessNode(mat, file_path)
            elif ("normal." in lower_fname) or ("norm." in lower_fname) or ("normalgl." in lower_fname):
                self.__add_normalNode(mat, file_path)
            elif ("height." in lower_fname) or ("displacement." in lower_fname) or ("disp." in lower_fname):
                self.__add_displacementNode(mat, file_path)
            elif ("metallic." in lower_fname) or ("metalness." in lower_fname) or ("metal." in lower_fname) or ("metalic." in lower_fname):
                self.__add_metallic(mat, file_path)
        # Default values, if not given:
        if self.metal != True:
            mat.node_tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = 0.0 # expected by most devs?

        return mat




    def apply_random(self, obj, num):
        pbr_dir = random.choice(self.pbr_dirs)
        
        mat = self.load_mat_from_folder(pbr_dir, num)
        
        self.apply_mat_to_obj(mat, obj)

        return pbr_dir, mat
        
