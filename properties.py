from bpy.props import StringProperty
from bpy.types import PropertyGroup

class File_Pickers(PropertyGroup):
    python_path: StringProperty(
        name="Path to Python", description="The path to the python executable on your system.\nThe python executable needs to have the same version as the python in Blender", subtype='FILE_PATH')
    eos_path: StringProperty(
        name="Path to EOS-PY", description="The path to the eos-py repository on your system", subtype='FILE_PATH')
    model_path: StringProperty(
        name="Path to Model", description="The path to the model on your system", subtype='FILE_PATH')
    blendshapes_path: StringProperty(
        name="Path to Blendshapes", description="The path to the model's blendshapes", subtype='FILE_PATH')
    texture_image_path: StringProperty(
        name="Path to Texture Image", description="The path to the image where texture will be extracted from", subtype='FILE_PATH')
    landmarks_mapper_path: StringProperty(
        name="Path to Landmarks Mapper", description="The path to the landmarks mapper", subtype='FILE_PATH')
    edge_topology_path: StringProperty(
        name="Path to Edge Topology", description="The path to the model's edge topology", subtype='FILE_PATH')
    model_contour_path: StringProperty(
        name="Path to the Model Contour", description="The path to the model contour", subtype='FILE_PATH')
    contour_landmarks_mapper_path: StringProperty(
        name="Path to Contour Landmarks Mapper", description="The path to the contour landmarks mapper", subtype='FILE_PATH')