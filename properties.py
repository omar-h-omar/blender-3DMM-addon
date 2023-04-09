from bpy.props import StringProperty, IntProperty, BoolProperty
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
    video_path: StringProperty(
        name="Path to Pre-recorded video", description="The path to a pre-recorded video for animation", subtype='FILE_PATH')

class Animation_Properties(PropertyGroup):
    fitting_iterations: IntProperty(
        name="Fitting Iterations", description="The number of iterations for the fitting process", default=5, min=1, max=300)
    set_number_of_coefficients: BoolProperty(
        name="Set number of coefficients", description="Set the number of shape and expression coefficients of the model", default=False)
    show_more_shape_coefficients: BoolProperty(default=False)
    show_more_blendshape_coefficients: BoolProperty(default=False)

class Coefficients_Collection(PropertyGroup):
  isEnabled: BoolProperty(default=True)
    