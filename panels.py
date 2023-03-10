from bpy.types import Panel

install_dependencies = False
try:
    import cv2
    import mediapipe as mp
    import eos
except:
    install_dependencies = True
    
    
class Main_PT_Panel(Panel):
    bl_label = "Dependencies Installer" if install_dependencies else "Model Creator"
    bl_idname = "MAIN_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "3D Morphable Model Addon"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()
        box = col.box()

        if (install_dependencies):
            # UI to install dependencies
            box.prop(context.scene.file_pickers,
                 "python_path", text="Path to Python")
            box.prop(context.scene.file_pickers, "eos_path", text="Path to EOS-PY")
            col.operator("main.install_dependencies",
                         text="Install Dependencies")
            col.label(
                text="Note: To use this addon, you need to install the python on your system first.")
        else:
            # UI to create the model
            box.prop(context.scene.file_pickers,
                     "model_path", text="Path to Model")
            box.prop(context.scene.file_pickers,
                     "blendshapes_path", text="Path to Blendshapes")
            col.operator("main.create_model", text="Create Model")

class Texture_PT_Panel(Panel):
    bl_label = "Texture Creator"
    bl_idname = "TEXTURE_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "3D Morphable Model Addon"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.active_object and '3DMM' in context.active_object

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()
        box = col.box()
        box.operator("main.uv_unwrap_model", text="UV Unwrap Model")
        box.label(
            text="Take a picture of yourself/Provide a picture path and click on the button below to create the texture.")
        box.operator("main.take_picture", text="Take Picture")
        box.prop(context.scene.file_pickers, "texture_image_path",
                 text="Path to Texture Image")
        col.operator("main.extract_texture", text="Extract Texture")

class Animator_PT_Panel(Panel):
    bl_label = "Animator"
    bl_idname = "ANIMATOR_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "3D Morphable Model Addon"
    bl_options = {"DEFAULT_CLOSED"}

    # Ensures that the panel is only visible when the model is selected
    @classmethod
    def poll(cls, context):
        return context.active_object and '3DMM' in context.active_object

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()
        box = col.box()
        box.prop(context.scene.file_pickers,
                 "landmarks_mapper_path", text="Path to Landmarks Mapper")
        box.prop(context.scene.file_pickers,
                 "edge_topology_path", text="Path to Edge Topology")
        box.prop(context.scene.file_pickers,
                 "model_contour_path", text="Path to Model Contour")
        box.prop(context.scene.file_pickers,
                 "contour_landmarks_mapper_path", text="Path to Contour Landmarks Mapper")
        col.operator("main.facial_recognition_mapper",
                     text="Start Real-Time Facial Animation")
