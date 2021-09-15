import bpy
import os
import sys
import glob
import random
import math
  
class pbr_loader:
    def __init__(self, base_dir):
        pbr_parent_path = os.path.join(base_dir, "SamplePBR")
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
        normalMap = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        normalNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
        normalNode.image = self.__load_or_reuse(normal_path)
        normalNode.image.colorspace_settings.name = 'Non-Color'
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(normalMap.inputs['Color'], normalNode.outputs['Color'])
        mat.node_tree.links.new(bsdf.inputs['Normal'], normalMap.outputs['Normal'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(normalNode.inputs['Vector'], scaleNode.outputs['Vector'])
                
    def __add_colorNode(self, mat, tex_path):
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = self.__load_or_reuse(tex_path)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(texImage.inputs['Vector'], scaleNode.outputs['Vector'])
    
    def __add_displacementNode(self, mat, displacement_path):
        displacementMap = mat.node_tree.nodes.new('ShaderNodeDisplacement')
        displacementNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
        displacementNode.image = self.__load_or_reuse(displacement_path)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat_output = mat.node_tree.nodes.get("Material Output")
        mat.node_tree.links.new(displacementMap.inputs['Height'], displacementNode.outputs['Color'])
        mat.node_tree.links.new(mat_output.inputs['Displacement'], displacementMap.outputs['Displacement'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(displacementNode.inputs['Vector'], scaleNode.outputs['Vector'])
    
    def __add_roughnessNode(self, mat, roughness_path):
        roughnessImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        roughnessImage.image = self.__load_or_reuse(roughness_path)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Roughness'], roughnessImage.outputs['Color'])
        scaleNode = mat.node_tree.nodes.get('Mapping')
        mat.node_tree.links.new(roughnessImage.inputs['Vector'], scaleNode.outputs['Vector'])


    def __add_scaleNode(self, mat):
        foo = mat.node_tree.nodes.new('ShaderNodeMapping')
        bar = mat.node_tree.nodes.new('ShaderNodeTexCoord')
        
        foo.inputs[3].default_value = (0.2, 0.2, 0.2) # would depend on texture / obj combination, but good enough in avg.
        
        mat.node_tree.links.new(foo.inputs['Vector'], bar.outputs['UV'])
        
        
    def apply_mat_to_obj(self, mat, obj):
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            #obj.data.materials = [mat]
            for i in range(0,len(obj.data.materials.values())):
                obj.data.materials[i] = mat        # before
                
                

    def apply_random(self, obj, num):
        pbr_dir = random.choice(self.pbr_dirs)
        
        material_name = 'mat-' + str(num)
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        
        self.__add_scaleNode(mat)
        base_name = os.path.basename(pbr_dir).split("-JPG")[0] + "_"
        color_path = os.path.join(pbr_dir, base_name + "Color.jpg")
        self.__add_colorNode(mat, color_path)
        normal_path = os.path.join(pbr_dir, base_name + "NormalGL.jpg")
        self.__add_normalNode(mat, normal_path)
        roughness_path = os.path.join(pbr_dir, base_name + "Roughness.jpg")
        self.__add_roughnessNode(mat, roughness_path)
        displacement_path = os.path.join(pbr_dir, base_name + "Displacement.jpg")
        self.__add_displacementNode(mat, displacement_path)
        
        self.apply_mat_to_obj(mat, obj)
        
        return pbr_dir            
        