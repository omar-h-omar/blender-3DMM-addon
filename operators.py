import os
import bpy
import bmesh
from bpy.types import Operator
import sys
import subprocess
import numpy as np
import time
import math
from scipy.spatial import distance
install_dependencies = False
try:
    import cv2
    import mediapipe as mp
    import eos
except:
    install_dependencies = True

from .utils import *

class Main_OT_Install_Dependencies(Operator):
    bl_label = "Install Dependencies"
    bl_idname = "main.install_dependencies"
    bl_description = "Install the dependencies for the addon"

    def execute(self, context):
        try:
            python_path = context.scene.file_pickers.python_path
            eos_path = context.scene.file_pickers.eos_path
            python_path = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
            eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/eos"

            # Updating pip and installing cmake on the system's python
            subprocess.call([python_path, "-m", "pip",
                             "install", "--upgrade", "pip"])
            subprocess.call([python_path, "-m", "pip", "install", "cmake"])

            # Building eos-py wheel
            subprocess.call(
                [python_path, "setup.py", "bdist_wheel", "--plat-name=any"], cwd=eos_path)

            # renaming the wheel
            import os
            os.environ["PYTHONNOUSERSITE"] = "1"
            os.rename(eos_path + '/dist/eos_py-1.3.0-cp310-cp310-any.whl',
                      eos_path + '/dist/eos_py-1.3.0-cp310-none-any.whl')

            # Upgrading pip and Installing eos-py wheel, openCV, cmake and mediapipe
            subprocess.call([sys.executable, "-m", "pip",
                             "install",  "--upgrade", "pip"])
            subprocess.call([sys.executable, "-m", "ensurepip"])
            subprocess.call([sys.executable, "-m", "pip", "install",
                             eos_path + "/dist/eos_py-1.3.0-cp310-none-any.whl"])
            subprocess.call([sys.executable, "-m", "pip",
                             "install", "opencv-python"])
            subprocess.call([sys.executable, "-m", "pip", "install", "cmake"])
            subprocess.call(
                [sys.executable, "-m", "pip", "install", "mediapipe"])

            # Importing the installed modules and toggling the install_dependencies variable
            import eos
            import cv2
            import mediapipe
            global install_dependencies
            install_dependencies = False

        except Exception as e:
            self.report({'ERROR'}, e.toString())
        return {'FINISHED'}

class Main_OT_Create_Model(Operator):
    bl_label = "Create Model"
    bl_idname = "main.create_model"
    bl_description = "Create model using eos-py"

    def execute(self, context):
        # from sklearn.neighbors import KDTree
        # import csv
        # media_pipe_model = [(bpy.data.objects[2].matrix_world @ v.co) for v in bpy.data.objects[2].data.vertices]
        # sfm_model = [(bpy.data.objects[3].matrix_world @ v.co) for v in bpy.data.objects[3].data.vertices]
        # tree = KDTree(media_pipe_model)
        # print(media_pipe_model[0])
        # print(media_pipe_model[:1])
        # print(tree.query([sfm_model[880]], k=1))
        # dist, sfm_ind = tree.query(media_pipe_model, k=1)
        # sfm_flat_ind = [item for sublist in sfm_ind for item in sublist]
        # mapping = {index: value for index, value in enumerate(sfm_flat_ind)}
        # repeats = set([i for i in sfm_flat_ind if sfm_flat_ind.count(i) > 1])
        # for i in repeats:
        #     mappings = [k for k, v in mapping.items() if v == i]
        #     distances  = [dist[m] for m in mappings]
        #     mappings.pop(distances.index(min(distances)))
        #     for m in mappings:
        #         mapping.pop(m)
        # with open('4DFM_conversion_with_no_repeats.csv', 'w', encoding='UTF8', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(mapping.items())
        
        # Getting the paths
        model_path = context.scene.file_pickers.model_path
        blendshapes_path = context.scene.file_pickers.blendshapes_path

        eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/eos"
        model_path = eos_path + "/share/sfm_shape_3448.bin"
        blendshapes_path = eos_path + "/share/expression_blendshapes_3448.bin"
        # model_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/4dfm_head_highres_v1.2_blendshapes_with_colour.bin"

        # Loading the model
        model = load_morphable_model(model_path, blendshapes_path)

        # Checking if the blendshapes are loaded
        if (model.get_expression_model() == None):
            self.report(
                {'ERROR'}, "Blendshapes could not be loaded. Please check that the model has blendshapes or enter a path to the blendshapes file.")
            return {'FINISHED'}

        # blendshape_coeffs = np.zeros(len(model.get_expression_model()))
        # blendshape_coeffs[3] = 1 # Raise left mouth corner
        # sample = model.draw_sample(shape_coefficients=[], expression_coefficients = blendshape_coeffs, color_coefficients=[])
        sample = model.draw_sample([], [])
        mesh = bpy.data.meshes.new('mesh')
        mesh.from_pydata(sample.vertices, [], sample.tvi)
        mesh.update()
        model_object = bpy.data.objects.new('model', mesh)

        # Adding the shape keys
        model_object.shape_key_add(name="Basis")
        shape_coeffs_count = model.get_shape_model().get_num_principal_components()
        blendshape_coeffs_count = len(model.get_expression_model())

        model_object.shape_coefficients.clear()
        for i in range(shape_coeffs_count):
            model_object.shape_coefficients.add()
            shape_coeffs = np.zeros(shape_coeffs_count)
            shape_coeffs[i] = 1
            sample = model.draw_sample(shape_coeffs, [])
            mesh = bpy.data.meshes.new('mesh')
            mesh.from_pydata(sample.vertices, [], sample.tvi)
            new_object = bpy.data.objects.new('model', mesh)
            sk = model_object.shape_key_add(name=f"Shape PCA {i + 1}")
            sk.slider_min = -3
            sk.slider_max = 3
            for i in range(len(sample.vertices)):
                sk.data[i].co = new_object.data.vertices[i].co

        # Adding the blendshape keys
        model_object.blendshape_coefficients.clear()
        for i in range(blendshape_coeffs_count):
            model_object.blendshape_coefficients.add()
            blendshape_coeffs = np.zeros(len(model.get_expression_model()))
            blendshape_coeffs[i] = 1
            sample = model.draw_sample(shape_coefficients=[
            ], expression_coefficients=blendshape_coeffs, color_coefficients=[])
            mesh = bpy.data.meshes.new('mesh')
            mesh.from_pydata(sample.vertices, [], sample.tvi)
            new_object = bpy.data.objects.new('model', mesh)
            sk = model_object.shape_key_add(name=f"Blendshape {i + 1}")
            sk.slider_min = -1
            sk.slider_max = 3
            for i in range(len(sample.vertices)):
                sk.data[i].co = new_object.data.vertices[i].co

        # Creating the material nodes
        material = bpy.data.materials.new(name="material")
        material.use_nodes = True
        bsdf_node = material.node_tree.nodes["Principled BSDF"]
        imageNode = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        material.node_tree.links.new(
            bsdf_node.inputs['Base Color'], imageNode.outputs['Color'])
        model_object.data.materials.append(material)

        context.collection.objects.link(model_object)
        # Adding custom properties to the model object
        model_object['3DMM'] = True
        return {'FINISHED'}

class Main_OT_UV_Unwrap_Model(Operator):
    bl_label = "UV Unwrap Model"
    bl_idname = "main.uv_unwrap_model"
    bl_description = "UV Unwrap the model"

    def execute(self, context):
        # Getting the paths
        mesh = context.active_object.data
        model_path = context.scene.file_pickers.model_path
        blendshapes_path = context.scene.file_pickers.blendshapes_path

        eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/eos"
        model_path = eos_path + "/share/sfm_shape_3448.bin"
        blendshapes_path = eos_path + "/share/expression_blendshapes_3448.bin"
        
        # Loading the morphable model
        model = load_morphable_model(model_path, blendshapes_path)

        # Creating a UV layer and unwrapping the model using the texture coordinates
        bm = bmesh.new()
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.new()
        sample = model.draw_sample([], [])
        for f in bm.faces:
            vertices = sample.tti[f.index] if len(
                sample.tti) != 0 else sample.tvi[f.index]
            f.loops[0][uv_layer].uv = (
                sample.texcoords[vertices[0]][0], 1 - sample.texcoords[vertices[0]][1]) # Flipping the y axis to match the openGL format
            f.loops[1][uv_layer].uv = (
                sample.texcoords[vertices[1]][0], 1 - sample.texcoords[vertices[1]][1])
            f.loops[2][uv_layer].uv = (
                sample.texcoords[vertices[2]][0], 1 - sample.texcoords[vertices[2]][1])
        bm.to_mesh(mesh)
        return {'FINISHED'}

class Main_OT_Take_Picture(Operator):
    bl_label = "Take Picture"
    bl_idname = "main.take_picture"
    bl_description = "Take a picture using the webcam and saves it"

    _cap = None  # Video capture object
    _timer = None  # Timer object

    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
        return {'FINISHED'}

    def modal(self, context, event):
        if (event.type == 'ESC'):
            self.cancel(context)
            return {'CANCELLED'}

        if (event.type == 'TIMER'):
            self.init_camera()
            _, frame = self._cap.read()
            if frame is not None:
                flipped_frame = cv2.flip(frame, 1)
                cv2.putText(flipped_frame, f'Press Space to Take a picture.',
                            (50, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.imshow('Take a Picture', flipped_frame)
                cv2.waitKey(1)

        if (event.type in 'SPACE' or cv2.waitKey(1) == 32):
            _, frame = self._cap.read()
            cv2.imwrite("captured_image.png", frame)
            context.scene.file_pickers.texture_image_path = os.path.join(
                os.path.dirname(__file__), "captured_image.png")
            self.cancel(context)
        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class Main_OT_Extract_Texture(Operator):
    bl_label = "Extract Texture"
    bl_idname = "main.extract_texture"
    bl_description = "Extract Texture From Image and Apply to Model Material"

    def execute(self, context):
        # Getting the paths
        material = context.active_object.active_material
        model_path = context.scene.file_pickers.model_path
        image_path = context.scene.file_pickers.texture_image_path
        landmarks_mapper_path = context.scene.file_pickers.landmarks_mapper_path
        blendshapes_path = context.scene.file_pickers.blendshapes_path
        
        eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/eos"
        model_path = eos_path + "/share/sfm_shape_3448.bin"
        landmarks_mapper_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/mediapipe_to_sfm.txt"
        blendshapes_path = eos_path + "/share/expression_blendshapes_3448.bin"
        
        # Checking if the paths are valid
        if (image_path == ""):
            self.report({'ERROR'}, "Please select an image")
            return {'CANCELLED'}

        # Loading the morphable model and the landmark mapper
        model = load_morphable_model(model_path, blendshapes_path)
        landmark_mapper = eos.core.LandmarkMapper(landmarks_mapper_path)
        
        # Initializing the face detector
        mp_face_detection = mp.solutions.face_mesh
        face_detection = mp_face_detection.FaceMesh(
            min_detection_confidence=0.5, max_num_faces=1, refine_landmarks=True)
        
        # Loading the Image
        image = cv2.imread(image_path)
        landmark_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detecting the face and extracting the landmarks
        results = face_detection.process(landmark_image)
        if results.multi_face_landmarks:
            landmarks_extracted = []
            for i in range(468):
                    landmarks_extracted.append(eos.core.Landmark(str(i + 1), [float(results.multi_face_landmarks[0].landmark[i].x * image.shape[1]), float(results.multi_face_landmarks[0].landmark[i].y * image.shape[0])]))
        else:
            self.report({'ERROR'}, "No face detected in the image")
            return {'CANCELLED'}
        
        # Fitting the model to the face
        (mesh, pose, shape_coeffs, blendshape_coeffs) = fit_shape_and_pose(
            model, landmarks_extracted, landmark_mapper, image.shape[1], image.shape[0])
        
        # Extracting the texture and applying it to the material node
        texture_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA, 4)
        texturemap = eos.render.extract_texture(mesh, pose, texture_image)
        cv2.imwrite("model_texture.png", texturemap)
        material.node_tree.nodes["Image Texture"].image = bpy.data.images.load(
            "model_texture.png")
        return {'FINISHED'}

class Main_OT_Facial_Recognition_Mapper(Operator):
    bl_label = "Facial Recognition Mapper"
    bl_idname = "main.facial_recognition_mapper"
    bl_description = "Map face to model using eos-py"

    _counter = 1  # Counter to keep track of the number of frames
    _prevTime = 0  # Previous timestamp of the loop
    _initialPosition = None  # Initial position of the face. Used for translation
    _cap = None  # Video capture object
    _timer = None  # Timer object

    _model = None
    _landmark_mapper = None
    _shape_coeffs_controller = None
    _blendshape_coeffs_controller = None
    _landmark_points_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336,
                           296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
                           380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
    _face_detection = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, max_num_faces=1, refine_landmarks=False)
    _shape_coeffs = None

    def init_camera(self, context):
        if context.scene.file_pickers.video_path:
            self._cap = cv2.VideoCapture(context.scene.file_pickers.video_path)
        else:
            self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None

    def modal(self, context, event):
        if (event.type in {'RIGHTMOUSE', 'ESC'}):
            self.cancel(context)
            return {'CANCELLED'}

        if (event.type == 'TIMER'):
            if self._cap == None:
                self.init_camera(context)
            _, image = self._cap.read()
            if image is not None:
                image.flags.writeable = False
                input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self._face_detection.process(input_image)
                if results.multi_face_landmarks:
                    # landmarks_extracted = [eos.core.Landmark(str(i), [float(results.multi_face_landmarks[0].landmark[index].x * image.shape[1]), float(results.multi_face_landmarks[0].landmark[index].y * image.shape[0])]) for i, index in enumerate(self._landmark_points_68, start=1)]
                    landmarks_extracted = []
                    for i in range(468):
                        landmarks_extracted.append(eos.core.Landmark(str(i + 1), [float(results.multi_face_landmarks[0].landmark[i].x * image.shape[1]), float(results.multi_face_landmarks[0].landmark[i].y * image.shape[0])]))
                    
                    if self._shape_coeffs is None:
                      (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self._model,
                                                                                                   landmarks_extracted, self._landmark_mapper, image.shape[1], image.shape[0], num_iterations=context.scene.animation_properties.fitting_iterations)
                      self._shape_coeffs = shape_coeffs
                    else:
                      (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self._model,
                                                                                                   landmarks_extracted, self._landmark_mapper, image.shape[1], image.shape[0], num_iterations=context.scene.animation_properties.fitting_iterations, pca_coeffs=self._shape_coeffs)
                    if self._counter == 1:
                        for i in range(len(shape_coeffs)):
                            if (self._shape_coeffs_controller[i] == True):
                                keyblock = context.active_object.data.shape_keys.key_blocks[i + 1]
                                keyblock.value = shape_coeffs[i]
                                keyblock.keyframe_insert("value", frame=self._counter)
                        self._initialPosition = pose.get_translation()

                    for i in range(len(blendshape_coeffs)):
                        if (self._blendshape_coeffs_controller[i] == True):
                            keyblock = context.active_object.data.shape_keys.key_blocks[len(shape_coeffs) + i + 1]
                            keyblock.value = blendshape_coeffs[i]
                            keyblock.keyframe_insert("value", frame=self._counter)

                    # Animating the mesh rotation and translation
                    rotation = pose.get_rotation_euler_angles()
                    context.active_object.rotation_euler = (rotation[0], -rotation[1], -rotation[2])
                    context.active_object.keyframe_insert("rotation_euler", frame=self._counter)
                    translation = self._initialPosition - pose.get_translation()
                    context.active_object.location = (translation[0], -translation[1], translation[2])
                    context.active_object.keyframe_insert("location", frame=self._counter)
                    # Incrementing the counter
                    self._counter += 1

                # Displaying the image and FPS
                currTime = time.time()
                fps = 1 / (currTime - self._prevTime)
                self._prevTime = currTime
                flipped_image = cv2.flip(image, 1)
                cv2.putText(
                    flipped_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
                cv2.imshow('Real Time Face Detection', flipped_image)
        return {'PASS_THROUGH'}

    def execute(self, context):
        eos_path = context.scene.file_pickers.eos_path
        model_path = context.scene.file_pickers.model_path
        blendshapes_path = context.scene.file_pickers.blendshapes_path
        landmarks_mapper_path = context.scene.file_pickers.landmarks_mapper_path

        eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/eos"
        model_path = eos_path + "/share/sfm_shape_3448.bin"
        blendshapes_path = eos_path + "/share/expression_blendshapes_3448.bin"
        # landmarks_mapper_path = eos_path + "/share/ibug_to_sfm.txt"
        landmarks_mapper_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/mediapipe_to_sfm.txt"
        
        self._model = load_morphable_model(model_path, blendshapes_path)
        self._landmark_mapper = eos.core.LandmarkMapper(landmarks_mapper_path)
        self._shape_coeffs_controller = [i.isEnabled for i in context.active_object.shape_coefficients]
        self._blendshape_coeffs_controller = [i.isEnabled for i in context.active_object.blendshape_coefficients]
        # 4DFM
        # eos_path = "/Users/omar/Desktop/Uni/Fourth Year/Project/4DFM"
        # self._model = load_morphable_model("/Users/omar/Desktop/Uni/Fourth Year/Project/4dfm_head_highres_v1.2_blendshapes_with_colour.bin")
        # self._landmark_mapper = eos.core.LandmarkMapper(eos_path + '/ibug68_landmark_mappings.txt')
        # self._landmark_mapper = eos.core.LandmarkMapper(eos_path + '/mediapipe_to_4dfm.txt')

        wm = context.window_manager
        self._timer = wm.event_timer_add(0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class Main_OT_Show_More_Shape_Coefficients(Operator):
    bl_label = "Show More Shape Coefficients"
    bl_idname = "main.show_more_shape_coefficients"

    def execute(self, context):
        context.scene.animation_properties.show_more_shape_coefficients = not context.scene.animation_properties.show_more_shape_coefficients
        return {'FINISHED'}

class Main_OT_Show_More_Blendshape_Coefficients(Operator):
    bl_label = "Show More Blendshape Coefficients"
    bl_idname = "main.show_more_blendshape_coefficients"

    def execute(self, context):
        context.scene.animation_properties.show_more_blendshape_coefficients = not context.scene.animation_properties.show_more_blendshape_coefficients
        return {'FINISHED'}