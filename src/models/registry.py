from src.models.cnn import build_cnn


MODEL_REGISTRY = {
    'cnn': build_cnn,

}

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY.keys():
        raise ValueError(f"Unknown model name. Recieved: {name} Available: {MODEL_REGISTRY.keys()}")
    return MODEL_REGISTRY[name](**kwargs)
