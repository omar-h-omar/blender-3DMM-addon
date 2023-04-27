# blender-3DMM-addon: A Blender python plugin for 3D Morphable Face Models

This plugin provides basic functionality to use 3D morphable face models (3DMMs) inside the Blender environment. It is powered by [eos](https://github.com/patrikhuber/eos).

Main features:
  * Support for Surrey Face Model (SFM), 4D Face Model (4DFM), Basel Face Model (BFM) 2009 and 2017, and the Liverpool-York Head Model (LYHM) out-of-the-box
  * Creation of Blender face objects from 3DMMs
  * Simple node tree to apply textures to the created Blender objects
  * Fast linear pose, shape and expression fitting
      * Supports images, videos or a live video feed using a webcam
  * Creation of UV layers if the 3DMM has UV-coordinates
  * Texture extraction to obtain a pose-invariant representation of the face texture


## Usage
The plugin was tested on Blender version 3.3.1 and eos version 1.4.0.

### Installation

To install the plugin:
  1. Download the latest release zip file
  2. In Blender go to Edit -> Preferences > Add-ons
  3. Click on Install and select the zip file

The plugin currently has the following dependencies: [eos](https://github.com/patrikhuber/eos), OpenCV, Numpy and Mediapipe.

To install the dependencies:
  1. Download the latest release for eos from [here](https://github.com/patrikhuber/eos/releases) and unzip the downloaded folder
  2. In the Dependencies Installer panel, provide the path to eos folder
  3. Click on Install Dependencies

**Note:** The eos installation is currently experimental. If you're facing difficulties, please download eos and build it for the Blender Python version. Then install the built wheel using `BLENDER-PYTHON-PATH -m pip install BUILT-WHEEL-PATH`.

### Panels

The plugin has three main panels: Model Creator, Animator and Texture Creator.

#### Model Creator Panel

This panel is responsible for creating Blender objects using 3DMMs. To use, simply provide a path to your 3DMM. If your model has a separate expression model, you can also provide a path to it in the panel. The panel will draw a mean sample from the 3DMM and create a Blender object using the resulting mesh data. All Blender objects created using this plugin comes with shape keys that correspond to the 3DMM's shape and expression coefficients. This allows for finer control over the resulting mesh.

#### Animator Panel

This panel allows you to animate your 3DMM. You can animate your Blender objects in real-time using a webcam. During testing, the plugin was able to animate 30 frames per second with the Surrey Face Model! Alternatively, you can provide a path to an existing image or video. There are a few ways to control the animation results. Firstly, you can control the number of fitting iterations. A higher number of iterations will result in a more accurate mesh but will take longer to animate. The second way to control the animation is to select which shape and expression coefficients to animate. Once you're satisfied with the options you've selected, click on `Animate Model` to begin the animation.

During the animation process, an extra window will appear to show you which frame is currently being animated. When a frame is animated your timeline is updated to include the new keyframe. You can stop the animation process at any point in time using the `ESC` key on your keyboard.

#### Texture Creator Panel

With this panel, you can UV unwrap your Blender objects using the UV-coordinates of the 3DMM. This panel also allows you to take a picture using a webcam or provide an existing image for texture extraction.

**Note:** UV unwrapping typically takes several minutes to complete. If you're using the Surrey Face Mode or the 4D Face Model, the `UV.blend` file contains UV unwrapped blender objects of those models. To copy the UV map, select the target object and using `Shift` select the source object. Then go to Object menu -> Make Links… -> Transfer UV Layouts (Shortcut: `Ctrl-L`)
