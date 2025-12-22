from sklearn.linear_model import Ridge 

def get_model(option):
    if option==1:
        model=Ridge

    return model


