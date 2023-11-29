
def get_model_device(model):
    return next(model.parameters()).device


