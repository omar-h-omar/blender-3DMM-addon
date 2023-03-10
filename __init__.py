import bpy
from bpy.props import PointerProperty
from bpy.types import Scene
from .panels import *
from .operators import *
from .properties import *

bl_info = {
    "name": "3D Morphable Model Addon",
    "author": "omar",
    "description": "",
    "blender": (3, 3, 1),
    "version": (0, 0, 1),
    "category": "3D View"
}

classes = (Main_PT_Panel, Animator_PT_Panel, Texture_PT_Panel, Main_OT_Install_Dependencies, Main_OT_Facial_Recognition_Mapper,
           Main_OT_Create_Model, Main_OT_UV_Unwrap_Model, Main_OT_Extract_Texture, Main_OT_Take_Picture, File_Pickers)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    Scene.file_pickers = PointerProperty(type=File_Pickers)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del Scene.file_pickers

if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
