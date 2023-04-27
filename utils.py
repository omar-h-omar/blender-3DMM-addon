try:
  import eos
except:
  pass

def load_morphable_model(model_path, blendshapes_path=""):
  # Loading the model
  model = eos.morphablemodel.load_model(model_path)

  # Checking if blendshapes need to be loaded
  if (model.get_expression_model_type() == eos.morphablemodel.MorphableModel.ExpressionModelType.none and blendshapes_path != ""):
      blendshapes = eos.morphablemodel.load_blendshapes(blendshapes_path)
      model = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                color_model=model.get_color_model(),
                                                vertex_definitions=None,
                                                texture_coordinates=model.get_texture_coordinates())
  
  return model
