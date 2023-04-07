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

def convert_landmarks(morphablemodel_with_expressions, current_mesh, landmarks, landmark_mapper):
  # The 2D and 3D point correspondences used for the fitting:
  vertex_indices = []  # their vertex indices
  image_points = []  # the corresponding 2D landmark points
  for i in range(len(landmarks)):
      converted_name = landmark_mapper.convert(landmarks[i].name)
      if (converted_name == None):
          continue

      if (morphablemodel_with_expressions.get_landmark_definitions()):
          found_vertex_idx = morphablemodel_with_expressions.get_landmark_definitions(
          ).value().find(converted_name.value())
          if found_vertex_idx != morphablemodel_with_expressions.get_landmark_definitions().value():
              vertex_idx = found_vertex_idx
          else:
              continue
      else:
          vertex_idx = int(converted_name)
      vertex = (current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                current_mesh.vertices[vertex_idx][2], 1)
      vertex_indices.append(vertex_idx)
      image_points.append(landmarks[i].coordinates)
  return vertex_indices, image_points

def fit_shape_and_pose(morphablemodel_with_expressions, landmarks, landmark_mapper, image_width, image_height, pca_shape_coefficients=[]):
    num_shape_coefficients_to_fit = morphablemodel_with_expressions.get_shape_model(
    ).get_num_principal_components()
    if (len(pca_shape_coefficients) > 0):
        current_pca_shape = morphablemodel_with_expressions.get_shape_model(
        ).draw_sample(pca_shape_coefficients)
    else:
        current_pca_shape = morphablemodel_with_expressions.get_shape_model(
        ).draw_sample([0.0] * num_shape_coefficients_to_fit)
    current_combined_shape = current_pca_shape + eos.morphablemodel.draw_sample(
        expression_model=morphablemodel_with_expressions.get_expression_model(), expression_coefficients=[])
    current_mesh = eos.morphablemodel.sample_to_mesh(current_combined_shape, morphablemodel_with_expressions.get_color_model().get_mean(),
                                                     morphablemodel_with_expressions.get_shape_model().get_triangle_list(),
                                                     morphablemodel_with_expressions.get_color_model().get_triangle_list(
    ), morphablemodel_with_expressions.get_texture_coordinates(),
        morphablemodel_with_expressions.get_texture_triangle_indices())

    # The 2D and 3D point correspondences used for the fitting:
    model_points = []  # the points in the 3D shape model
    vertex_indices = []  # their vertex indices
    image_points = []  # the corresponding 2D landmark points
    for i in range(len(landmarks)):
        converted_name = landmark_mapper.convert(landmarks[i].name)
        if (converted_name == None):
            continue

        if (morphablemodel_with_expressions.get_landmark_definitions()):
            found_vertex_idx = morphablemodel_with_expressions.get_landmark_definitions(
            ).value().find(converted_name.value())
            if found_vertex_idx != morphablemodel_with_expressions.get_landmark_definitions().value():
                vertex_idx = found_vertex_idx
            else:
                continue
        else:
            vertex_idx = int(converted_name)
        vertex = (current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                  current_mesh.vertices[vertex_idx][2], 1)
        model_points.append(vertex)
        vertex_indices.append(vertex_idx)
        image_points.append(landmarks[i].coordinates)

    # Initial Pose and Expression Fitting:
    current_pose = eos.fitting.estimate_orthographic_projection_linear(
        image_points, model_points, image_width, image_height)
    rendering_params = eos.fitting.RenderingParameters(
        current_pose, image_width, image_height)
    affine_from_ortho = eos.fitting.get_3x4_affine_camera_matrix(
        rendering_params, image_width, image_height)
    expression_coefficients = eos.fitting.fit_expressions(morphablemodel_with_expressions.get_expression_model(
    ), current_pca_shape, affine_from_ortho, image_points, vertex_indices)
    current_combined_shape = current_pca_shape + eos.morphablemodel.draw_sample(
        expression_model=morphablemodel_with_expressions.get_expression_model(), expression_coefficients=expression_coefficients)
    current_mesh = eos.morphablemodel.sample_to_mesh(current_combined_shape, morphablemodel_with_expressions.get_color_model().get_mean(),
                                                     morphablemodel_with_expressions.get_shape_model().get_triangle_list(),
                                                     morphablemodel_with_expressions.get_color_model().get_triangle_list(
    ), morphablemodel_with_expressions.get_texture_coordinates(),
        morphablemodel_with_expressions.get_texture_triangle_indices())

    for i in range(5):
        current_pose = eos.fitting.estimate_orthographic_projection_linear(
            image_points, model_points, image_width, image_height)
        rendering_params = eos.fitting.RenderingParameters(
            current_pose, image_width, image_height)
        affine_from_ortho = eos.fitting.get_3x4_affine_camera_matrix(
            rendering_params, image_width, image_height)

        #  Estimate the PCA shape coefficients with the current blendshape coefficients:
        if (len(pca_shape_coefficients) == 0):
            mean_plus_expressions = morphablemodel_with_expressions.get_shape_model().get_mean()
            + eos.morphablemodel.draw_sample(expression_model=morphablemodel_with_expressions.get_expression_model(
            ), expression_coefficients=expression_coefficients)
            pca_shape_coefficients = eos.fitting.fit_shape_to_landmarks_linear(
                shape_model=morphablemodel_with_expressions.get_shape_model(), affine_camera_matrix=affine_from_ortho, landmarks=image_points, vertex_ids=vertex_indices,
                base_face=mean_plus_expressions, num_coefficients_to_fit=num_shape_coefficients_to_fit)

        #  Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphablemodel_with_expressions.get_shape_model(
        ).draw_sample(pca_shape_coefficients)
        expression_coefficients = eos.fitting.fit_expressions(morphablemodel_with_expressions.get_expression_model(
        ), current_pca_shape, affine_from_ortho, image_points, vertex_indices)
        current_combined_shape = current_pca_shape + eos.morphablemodel.draw_sample(
            expression_model=morphablemodel_with_expressions.get_expression_model(), expression_coefficients=expression_coefficients)
        current_mesh = eos.morphablemodel.sample_to_mesh(current_combined_shape, morphablemodel_with_expressions.get_color_model().get_mean(),
                                                         morphablemodel_with_expressions.get_shape_model().get_triangle_list(),
                                                         morphablemodel_with_expressions.get_color_model().get_triangle_list(
        ), morphablemodel_with_expressions.get_texture_coordinates(),
            morphablemodel_with_expressions.get_texture_triangle_indices())
    return (current_mesh, rendering_params, pca_shape_coefficients, expression_coefficients)
