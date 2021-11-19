
def scale_dimension(dim,scale):
    return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

def cell_params_iterator(input_features,resolution_levels,resolution_level_to_features,**kwargs):
    for i,curr_level in enumerate(resolution_levels):
        
        prev_level = resolution_levels[i - 1] if i > 0 else 0
        prev_prev_level = resolution_levels[i - 2] if i > 1 else 0 
        scale = (curr_level - prev_level)

        # if we're currently at a greater resolution, than before, scale input up
        # if we're at a lower resolution, or the same one (0 or 1), scale down, or leave the same
        if scale == -1: 
            scale = 2
        elif scale == 0: 
            scale = 1
        elif scale == 1:
            scale = 0.5
        else:
            raise Exception('cannot scale more than one resolution layer')

        features_prev_prev = resolution_level_to_features[prev_prev_level] * 4
        features_prev = resolution_level_to_features[prev_level] * 4
        if i == 1:
            features_prev_prev = input_features
        elif i == 0:
            features_prev_prev = input_features
            features_prev = input_features

        features_to_next = resolution_level_to_features[curr_level]

        yield({
            'c_in_prev_prev': features_prev_prev,
            'c_in_prev': features_prev,
            'c_out': features_to_next,
            'scale': scale,
            **kwargs
        })