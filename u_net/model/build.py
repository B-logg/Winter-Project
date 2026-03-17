from model.unet import UNet_carbon


def build_model(net, num_class, dropout=False):

    if net == "UNet_carbon":
        model = UNet_carbon(num_classes=num_class, dropout=dropout)

    return model
    