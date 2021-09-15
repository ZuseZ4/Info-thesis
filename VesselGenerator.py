
# Produce vessel object and 5 content objects
# See "Main" section in line 586

###############################Dependcies######################################################################################3

import bpy
import math
import numpy as np
import bmesh
import os
import shutil
import random
import json
#####################################################################################################################
def RandPow(n):
    r=1
    for i in range(int(n)):
        r*=np.random.rand()
    return r


###############################################################################################################################

##==============Clean secene remove  all objects currently on the schen============================================

###############################################################################################################################

def CleanScene():
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            #bpy.data.cameras,
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
###########################333#############################################################

# Create random slope curve  (r vs z) for vessel region 

#########################################################################################################
def CreateSlope(slope0):
    pr=np.random.rand();
 
    a=b=y=Rad=Drad=0
    print(pr)
    if pr<0.20:
        Mode="Linear"
        slope=0
        a=0
        b=0
        print("++++++++++++++++")
    elif pr<0.38:
        Mode="Linear"
        slope=np.random.rand()*3-1.5
        a=0
        b=0
    elif pr<0.56:
        Mode="Linear"
        slope=slope0
        a=np.random.rand()*0.3-0.15
        b=0
    elif pr<0.78:
        Mode="polynomial"
        slope=slope0
        a=np.random.rand()*0.3-0.15
        b=np.random.rand()*0.2-0.1
    elif pr<1.1:
        Mode="Sin"
        print("SIN")
        a=2*RandPow(2)
        Rad=np.random.rand()*3.14
        Drad=RandPow(2)#3
        slope=0
        b=0
    y=0
    
    return slope,a,b,y,Mode,Rad,Drad
    
###################################################################################################################

# CreateRadius Array for vessel profile (Curve that determine how the vessel radius change a long the Z axis)

##################################################################################################################
def CreateRadiusArray(MinH,MaxH,MinR,MaxR): 
        print("========================Creating Radius Array=====================================================")
        h=np.random.randint(MaxH-MinH)+MinH # Height of vessel (number of layers
        
        MaterialTopHeight=int((np.random.rand()*(h-1))+2)
        
        MaterialInitHeight=int(np.random.rand()*(MaterialTopHeight-1))
        MaterialTopHeight=h

        
        if MaterialInitHeight>=MaterialTopHeight: MaterialInitHeight= MaterialTopHeight-1
        
     
    #------------------------------------------------------------------
        r=np.random.rand()*50+4 # Radius of the vessel at start
        r0=r
        rl=[r]# array of radiuses in each layer of  the vessel
#        dslop=np.random.rand()*0.4-0.2 # diffrential of slope
#        if np.random.rand()<-0.7: 
#             Mode="Linear"
#        else:
#             Mode="Sin"
#             Rad=np.random.rand()*3.14
#             Drad=np.random.rand()*0.3
#             
        slope,a,b,y,Mode,Rad,Drad=CreateSlope(0)
        dslop=0
        swp=np.random.rand()*3 # Probability for replacement
        for i in range(h): # go layer by layer and change vessel raius
            while(True):  
                 if Mode=="Linear":
                     dslop=a+b*y
                     slope+=dslop#-np.random.rand()*0.4
                     y+=1
                 if Mode=="Sin":
                     Rad+=Drad                 
                     #slope=a*np.sin(Rad)
                     slope=a*np.sin(Rad)
    #                  
#                 if slope<-9 and dslop<0:
#                     slope,a,b,y,Mode,Rad,Drad=CreateSlope(slope)
#                     continue
#                 
#                 if slope>9  and dslop>0: 
#                     slope,a,b,y,Mode,Rad,Drad=CreateSlope(slope)
#                     continue
                 if (r+slope)>MaxR and dslop>0:
                     slope,a,b,y,Mode,Rad,Drad=CreateSlope(slope)
                     continue
                 if (r+slope)<MinR and dslop<0:
                        slope,a,b,y,Mode,Rad,Drad=CreateSlope(slope)
                        continue
                 if np.random.rand()<swp/h:
                     slope,a,b,y,Mode,Rad,Drad=CreateSlope(slope)
                     print("SwitcH")
                     print(Mode)
                 break
  
            r+=slope 
            if r>MaxR: r=MaxR
            if r<MinR: r=MinR      
            rl.append(r)  
           #print("h="+str(h))
           #print("MaterialInitHeight="+str(MaterialInitHeight))
           #print("MaterialTopHeight="+str(MaterialTopHeight))
        return rl, MaterialTopHeight, MaterialInitHeight,h   
##################################################################################################################333

#                          Create vessel Object 

####################################################################################################################
def AddVessel(VesselName="Vessel",ContentName="MaterialContent",Col=bpy.context.collection,MinH=4,MaxH=80,MinR=4,MaxR=40,ScaleFactor=0.1):   
    print("=================Create Vessel Mesh object and add to scene==================================")     
    #--------------------Create random shape Material assign parameters----------------------------------------- 
    if np.random.rand()<0.5: 
        ModeThikness="Solidify" # Mode in which vessel thikness will be generated solidify modifier
    else:
         ModeThikness="DoubleLayer"  # Mode in which vessel thikness will be generated double layer wall
#--------------------------------------------------------------------------------------
  
    #------------------------Create vessel random information---------------------------------------------------- 
    #Vnum = np.random.randint(50)+3 #Number vertex in a layer/ring
    Vnum = np.random.randint(5)+3 #Number vertex in a layer/ring
    Vinc = (math.pi*2)/(Vnum) # difference of angle between vertex
    
    #--------------Generate information of vessel profile radius vs height---------------------------------
    rl, MaterialTopHeight, MaterialInitHeight,VesselHeight=CreateRadiusArray(MinH,MaxH,MinR,MaxR)      
     
     #----------------Set thiknesss for vessel wall/surface---------------------------------------
    VesselWallThikness=1000
    while(VesselWallThikness>np.max(rl)):
        if np.random.rand()<1:
              VesselWallThikness=(np.max(rl)/(11))*(np.random.rand()+0.1)
        else:
              VesselWallThikness=np.min(rl)*(np.random.rand()+0.1)/4# # Vessel thikness
    
    #-----------------------Set floor/bottum------------------------------------
    
    VesselFloorHeight=0
    if np.random.rand()<0.2:
           VesselFloorHeight= np.random.randint(np.max([1,np.min([int(MaterialInitHeight-VesselWallThikness),int(VesselHeight/9)])]))
           
           
               
           #VesselFloorHeight=int(VesselHeight/2)
           print("VesselFloorHeight"+str(VesselFloorHeight))
    
    #-------------------Scale Vessel-------------------------------------------------------
    VesselWallThikness*=ScaleFactor  
    VesselFloorHeight*=ScaleFactor      
    
    
    #-----------Strech deform of vessel out of cylindrical----------------------------------------------------------------------------------- 
    stx=sty=1   
#    if np.random.rand()<0.04:
#        if np.random.rand()<0.5:
#             stx=np.random.rand()*0.8+0.2
#        else: 
#             sty=np.random.rand()*0.8+0.2
    #----------------------Content size this is the initial shape/mesh of the liquid inside the vessel------------------------------------------------------------

    MatX_RadRatio=0.97
    MatY_RadRatio=0.97
    MaterialInitHeight = VesselFloorHeight+1
    
     
#======================Add Vertex for vessel/ vessel opening/ and conenten meshes==================================   
    #---------------------------Vessel openining vertex-----------------------------------------------
    Openverts = [] # openning in vessel mouth
    Openfaces = []
    Openedges = []
    #---------------------------Material/content meshes---------------------------------------------------------------------
    Matverts = {}
    Matfaces = {}
    Matedges = {}
    MaterialTopHeight={}
    for i in range(5):
          Matverts[i]=[]
          Matfaces[i]=[]
          Matedges[i]=[]
          MaterialTopHeight[i]=VesselHeight*(np.random.rand()*0.9+0.1)
        

    #-----------------------Create vertex object and faces arrays for vessel------------------------------------------------------------------
    verts = []
    faces = []
    edges = []
    
    
    
    
    #=============================Add vertexesx=======================================================
    
    MaxXY=0

    MaxZ=0
    for fz in range(len(rl)):
        for j in range(0,Vnum ):
            theta=j*Vinc
            r1=rl[fz]
            x = (r1 * math.cos(theta))*stx*ScaleFactor
            y = (r1 * math.sin(theta))*sty*ScaleFactor
            z = fz*ScaleFactor #scale * (r2 * math.sin(phi))
            MaxZ=np.max([z,MaxZ])
            MaxXY=np.max([x,y,MaxXY])
    
         #   print(x)
            vert = (x,y,z)  # Add Vertex
            verts.append(vert) # Add to vessel vertexes
            Mvert = (x* MatX_RadRatio,y* MatY_RadRatio,z)  # 
            for i in range(5):
                    if fz<=MaterialTopHeight[i] and fz>=MaterialInitHeight: # Material/content inside vessel
                          Matverts[i].append(Mvert)
            if fz==len(rl)-1: # opening of vessel
                          Openverts.append(vert)
# ...................Inner walll if the vessel surface is double layered-----------------------------
    if  ModeThikness=="DoubleLayer":
        if VesselFloorHeight==0:  VesselFloorHeight=1
        for fz in range(len(rl)-1,np.max([int(VesselFloorHeight-1),1]),-1):
            for j in range(0,Vnum ):
                theta=j*Vinc
                r1=rl[fz]-VesselWallThikness
                x = (r1 * math.cos(theta))*stx*ScaleFactor
                y = (r1 * math.sin(theta))*sty*ScaleFactor
                z = fz*ScaleFactor #scale * (r2 * math.sin(phi))
                MaxZ=np.max([z,MaxZ])
                MaxXY=np.max([x,y,MaxXY])
                vert = (x,y,z)  # Add Vertex
                verts.append(vert)
    
    #---------------------------------------------------------------------------------------------------------------
    #        Add faces / combine vertex into faces
    #----------------------------------------------------------------------------------------------------------------
    #----------------------------vessel wall----------------------------------    
    for k in range(len(verts)-Vnum):
        if not k%Vnum==(Vnum-1): 
            face = (k,k+1,k+Vnum+1,k+Vnum)
        else: # the last point in the ring is connected to the first point in the ring
            face = (k,k-Vnum+1,k+1,k+Vnum)
        faces.append(face) 
        for i in range(5):
           if k+Vnum<len(Matverts[i]):
                Matfaces[i].append(face) 
    #    #print(k)
    #------------Vessel floor as single face-------------------------------------------        
    if np.random.rand()<0.85:
        face = (0,)
        faceTop=(VesselFloorHeight*Vnum,) # face of top floor
        for k in range(1,Vnum):
            face += (k,)
            faceTop += (k+VesselFloorHeight*Vnum,)
        if VesselFloorHeight>0: faces.append(faceTop) 
        faces.append(face) 
        for i in range(5):
           Matfaces[i].append(face)
        Openfaces.append(face)
        
    #------------content top as as a single single face-------------------------------------------
    for i in range(5):        
        face = (len(Matverts[i])-Vnum-1,)
        for k in range(len(Matverts[i])-Vnum,len(Matverts[i])):
            face += (k,)
        Matfaces[i].append(face) 
#***************************************************************************
                
#*******************************************************************************
  
    #---------------------------------------------------------------------------------------------------------------
    #  Create vessel mesh and object and add to scene
    #----------------------------------------------------------------------------------------------------------------
    #create mesh and object
    mymesh = bpy.data.meshes.new(VesselName)
    myobject = bpy.data.objects.new(VesselName,mymesh)
     
    #set mesh location
    myobject.location=(0,0,0)
    # bpy.context.collection.objects.link(myobject) # before
    Col.objects.link(myobject)
    #bpy.context.scene.objects.link(myobject)
    #"create mesh from python data"
    print("create mesh from python data")
    mymesh.from_pydata(verts,edges,faces)
    mymesh.update(calc_edges=True)

    bpy.data.objects[VesselName].select_set(True)
    #bpy.context.scene.objects.active = bpy.data.objects["Vessel"]
    bpy.context.view_layer.objects.active = bpy.data.objects[VesselName]
    #bpy.context.object=bpy.data.objects['supershape'
    #-----------------------------------------------------------------------------------------------------------------
    #              Add modifiers to vessel
    #----------------------Add modifiers to vessel to smooth and i----------------------------------------------------------
    SubdivisionLevel=-1
    Smooth=False   
    if np.random.rand()<0.9:
        bpy.ops.object.modifier_add(type='SUBSURF') # add more polygos (kind of smothing
        SubdivisionLevel=np.random.randint(4)
        SubdivisionRenderLevel=np.random.randint(4)
        bpy.context.object.modifiers["Subdivision"].levels = SubdivisionLevel
        bpy.context.object.modifiers["Subdivision"].render_levels = SubdivisionRenderLevel
    if np.random.rand()<0.9:    
        bpy.ops.object.shade_smooth() # smooth 
        Smooth=True
    if  ModeThikness=="Solidify": # Add vessel thikness usiing solidify modifier
         bpy.ops.object.modifier_add(type='SOLIDIFY')# Set Vessel thikness 
         bpy.context.object.modifiers["Solidify"].thickness = VesselWallThikness
     
    #===================================================================================
    #-------------------------------------Add content object------------------------------------------------------------------------------------     
    #create mesh and object
    for i in range(5):   
        mymesh = bpy.data.meshes.new(ContentName+str(i))
        myobject = bpy.data.objects.new(ContentName,mymesh)
         
        #set mesh location
        myobject.location=(0,0,0)
        bpy.context.collection.objects.link(myobject)
        #bpy.context.scene.objects.link(myobject)
         
        #create material mesh from python data

        print("create material mesh from python data")
        mymesh.from_pydata(Matverts[i],Matedges[i],Matfaces[i])
        mymesh.update(calc_edges=True)

        bpy.data.objects[ContentName].select_set(True)
    #...............................Add modifier to content object........................................
    #bpy.context.scene.objects.active = bpy.data.objects["Vessel"]
        bpy.context.view_layer.objects.active = bpy.data.objects[ContentName]
        if SubdivisionLevel>0: # Smooth
            bpy.ops.object.modifier_add(type='SUBSURF') # add more polygos (kind of smothing
            bpy.context.object.modifiers["Subdivision"].levels = SubdivisionLevel
            bpy.context.object.modifiers["Subdivision"].render_levels = SubdivisionRenderLevel
        if Smooth: bpy.ops.object.shade_smooth() # smooth 
    #===================================================================================
#    #-------------------------------------Add Vessel opening plate as an object------------------------------------------------------------------------------------     
#    mymesh = bpy.data.meshes.new("VesselOpenning")
#    myobject = bpy.data.objects.new("VesselOpenning",mymesh)
#     
#    #set mesh location
#    myobject.location=(0,0,0)
#    bpy.context.collection.objects.link(myobject)
#    #bpy.context.scene.objects.link(myobject)
#     
#    #create mesh from python data


#    mymesh.from_pydata(Openverts,Openedges,Openfaces)
#    mymesh.update(calc_edges=True)

#    bpy.data.objects[ContentName].select_set(True)
#    #bpy.context.scene.objects.active = bpy.data.objects["Vessel"]
#    bpy.context.view_layer.objects.active = bpy.data.objects["VesselOpenning"]
#    if SubdivisionLevel>0:
#        bpy.ops.object.modifier_add(type='SUBSURF') # add more polygos (kind of smothing
#        bpy.context.object.modifiers["Subdivision"].levels = SubdivisionLevel
#        bpy.context.object.modifiers["Subdivision"].render_levels = SubdivisionRenderLevel
#    if Smooth: bpy.ops.object.shade_smooth() # smooth 
    #===================================================================================

    return MaxXY,MaxZ,VesselFloorHeight,VesselWallThikness

###################################################################################################################################

# Transform BSDF Mateiral to dictionary (use to save materials properties)

####################################################################################################################################
def BSDFMaterialToDictionary(Mtr):
    bsdf=Mtr.node_tree.nodes["Principled BSDF"]
    dic={}
    dic["TYPE"]="Principled BSDF"
    dic["Name"]=Mtr.name
    dic["Base Color"]=(bsdf.inputs[0].default_value)[:]## = (0.0892693, 0.0446506, 0.137255, 1)
    dic["Subsurface"]=bsdf.inputs[1].default_value## = 0
    dic["Subsurface Radius"]=str(bsdf.inputs[2].default_value[:])
    dic["Subsurface Color"]=bsdf.inputs[3].default_value[:]# = (0.8, 0.642313, 0.521388, 1)
    dic["Metalic"]=bsdf.inputs[4].default_value# = 5
    dic["Specular"]=bsdf.inputs[5].default_value# = 0.804545
    dic["Specular Tint"]=bsdf.inputs[6].default_value# = 0.268182
    dic["Roughness"]=bsdf.inputs[7].default_value# = 0.64
    dic["Anisotropic"]=bsdf.inputs[8].default_value# = 0.15
    dic["Anisotropic Rotation"]=bsdf.inputs[9].default_value# = 0.236364
    dic["Sheen"]=bsdf.inputs[10].default_value# = 0.304545
    dic["Sheen tint"]=bsdf.inputs[11].default_value# = 0.304545
    dic["Clear Coat"]=bsdf.inputs[12].default_value# = 0.0136364
    dic["Clear Coat Roguhness"]=bsdf.inputs[13].default_value #= 0.0136364
    dic["IOR"]=bsdf.inputs[14].default_value# = 3.85
    dic["Transmission"]=bsdf.inputs[15].default_value# = 0.486364
    dic["Transmission Roguhness"]=bsdf.inputs[16].default_value# = 0.177273
    dic["Emission"]=bsdf.inputs[17].default_value[:]# = (0.170604, 0.150816, 0.220022, 1)
    dic["Emission Strengh"]=bsdf.inputs[18].default_value
    dic["Alpha"]=bsdf.inputs[19].default_value
   # dic["bsdf Blender"]=bsdf.inputs
    return dic

###################################################################################################################################

# Transform glosst transparent Mateiral to dictionary (use to save materials properties)

####################################################################################################################################
def GlassMaterialToDictionary(Mtr):
    print("Creating glass material dictionary")
    GlassBSDF = bpy.data.materials["TransparentLiquidMaterial"].node_tree.nodes["Glass BSDF"]
    VolumeAbsorbtion = bpy.data.materials["TransparentLiquidMaterial"].node_tree.nodes["Volume Absorption"]
    dic={}
    dic["TYPE"]="Glass Transparent"
    dic["Name"]=Mtr.name
    dic["Distribution"]=GlassBSDF.distribution# = 'BECKMANN'
    dic["Base Color"]=GlassBSDF.inputs[0].default_value[:]# = (1, 0.327541, 0.225648, 1)
    dic["Roughness"]=GlassBSDF.inputs[1].default_value
    dic["IOR"]=GlassBSDF.inputs[2].default_value
    
    dic["VolumeAbsorbtion Color"]=VolumeAbsorbtion.inputs[0].default_value[:]# = (1, 0.668496, 0.799081, 1)
    dic["Density"]=VolumeAbsorbtion.inputs[1].default_value
    
#    dic["GLASS bsdf Blender"] = GlassBSDF
 #   dic["VolumeAbsorbtion Blender"] = VolumeAbsorbtion
    return dic
     

###############################################################################################################################

##            #Assing  random tansperent material to vessel material ( assume material already exists in the blend file)

###############################################################################################################################
def AssignMaterialToVessel(name):  
  
    print("================= Assign material to vessel "+name+"=========================================================")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[name] 
#------------------------Assing material node to vessel----------------------------------------------------------
    if np.random.rand()<10.8:
       bpy.data.objects[name].data.materials.append(bpy.data.materials['Glass'])
   
#-----------Set random properties for material-----------------------------------------------------
    if np.random.rand()<0.02: # Color
        bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand())
    else:
        rnd=1-np.random.rand()*0.3
        bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (rnd, rnd, rnd, rnd)

    bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[3].default_value = bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[0].default_value 


    if np.random.rand()<0.1: # Subsurface
        bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = np.random.rand()
    else:
        bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 0
   
   
    if np.random.rand()<0.15: #Transmission
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[15].default_value = 1-0.15*RandPow(3) # Transmission
    else:
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[15].default_value = 1 #Transmission
       
       
    if np.random.rand()<0.2: # Roughnesss
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.2*RandPow(3) # Roughness
    else: 
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0# Roughness
  
 
   
       
    if np.random.rand()<0.6:# ior index refraction
         bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[14].default_value = 1.45+np.random.rand()*0.55 #ior index of reflection for transparen objects  
    
    else:
        bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[14].default_value = 1.415+np.random.rand()*0.115 #ior index of reflection for transparen objects  
    #https://pixelandpoly.com/ior.html

    if np.random.rand()<0.3:# transmission rouighness
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[16].default_value = 0.15*RandPow(3) # transmission rouighness
    else: 
        bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[16].default_value = 0 # transmission rouighness
    

    if np.random.rand()<0.12: # Metalic
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[4].default_value = 0.18*RandPow(3)# metalic
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[4].default_value =0# meralic
      
      
    if np.random.rand()<0.12: # specular
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[5].default_value = np.random.rand()# specular
    elif np.random.rand()<0.6:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[5].default_value =0.5# specular
    else:
      ior=bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[14].default_value# specular
      specular=((ior-1)/(ior+1))**2/0.08
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[5].default_value=specular
      
    if np.random.rand()<0.12: # specular tint
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[6].default_value = np.random.rand()# tint specular
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[6].default_value =0.0# specular tint
  
    if np.random.rand()<0.12: # unisotropic
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[8].default_value = np.random.rand()# unisotropic
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[8].default_value =0.0# unisotropic
  
    if np.random.rand()<0.12: # unisotropic
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[9].default_value = np.random.rand()# unisotropic rotation
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[9].default_value =0.0# unisotropic
    
    if np.random.rand()<10.15:
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[16].default_value = 0.25*RandPow(3) # transmission rouighness
    else: 
        bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[16].default_value = 0 # transmission rouighness
    

    if np.random.rand()<0.1: # Clear  coat
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[12].default_value = np.random.rand()
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[12].default_value =0# 

    if np.random.rand()<0.1: # Clear  coat
       bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[13].default_value = np.random.rand()
    else:
      bpy.data.materials['Glass'].node_tree.nodes["Principled BSDF"].inputs[13].default_value =0.03# 
    bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[10].default_value = 0
    bpy.data.materials["Glass"].node_tree.nodes["Principled BSDF"].inputs[11].default_value = 0.5

    return BSDFMaterialToDictionary(bpy.data.materials["Glass"]) # turn material propeties into dictionary (for saving)




################################################################################################################################################################

#                                    Main 

###################################################################################################################################################################

#CleanScene()  # Delete all objects in scence
  
#    #------------------------------Create random vessel object and assign  material to it---------------------------------
#MaxXY,MaxZ,MinZ,VesselWallThikness=AddVessel("Vessel","Content",Col=bpy.context.collection,ScaleFactor=0.01)#np.random.rand()+0.1) # Create Vessel object named "Vessel" and add to scene also create mesh inside the vessel ("Content) which will be transformed to liquid
# MaxXY are the maximal values of XY in the vessesl (basically  max radius) MaxZ  MinZ are the maximal and minimal Z values 
#VesselMaterial=AssignMaterialToVessel("Vessel") # assign random material to vessel object

 
