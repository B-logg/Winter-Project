from torchvision.utils import make_grid

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def log_tensorboard(writer, current_logs, i_iter):
    for name, value in current_logs.items():
        writer.add_scalar(f'data/{name}', to_numpy(value), i_iter)

def draw_in_tensorboard(writer, images, i_iter):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image', grid_image, i_iter)

def write_weight_in_tensorboard(writer, model, i_iter):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i_iter)
