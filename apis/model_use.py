from load_model.model import LOAD_SIAMESE_NETWORK

def model_use():
    AI_model = LOAD_SIAMESE_NETWORK()
    model = AI_model.create_model_siamese()
    model()