import timm
import model


def getModel():
    num_classes = 10
    model_name = 'repvit_m0_9'
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def getModel_defense(**kwargs):
    num_classes = 10
    model_name = 'repvit_m0_9'
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def getModel_origin():
    num_classes = 10
    model_name = 'repvit_m0_9'
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
