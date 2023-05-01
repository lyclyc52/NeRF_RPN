import torch

pi = 3.141592

def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def regular_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    w_regular = torch.where(w > h, w, h)
    h_regular = torch.where(w > h, h, w)
    theta_regular = torch.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


def rectpoly2obb(polys):
    eps=1e-7
    theta = torch.atan2(-(polys[..., 3] - polys[..., 1]), polys[..., 2] - polys[..., 0] + eps)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    Matrix = torch.stack([Cos, -Sin, Sin, Cos], dim=-1)
    Matrix = Matrix.view(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = torch.stack([x, y], dim=-1).unsqueeze(-2)
    center_polys = polys.view(*polys.shape[:-1], 4, 2) - center
    rotate_polys = torch.matmul(center_polys, Matrix.transpose(-1, -2))

    xmin, _ = torch.min(rotate_polys[..., :, 0], dim=-1)
    xmax, _ = torch.max(rotate_polys[..., :, 0], dim=-1)
    ymin, _ = torch.min(rotate_polys[..., :, 1], dim=-1)
    ymax, _ = torch.max(rotate_polys[..., :, 1], dim=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = torch.stack([x, y, w, h, theta], dim=-1)
    return regular_obb(obboxes)


def obb2poly(obboxes):
    """Return the 4-point representation of OBB."""
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat([w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat([-h/2 * Sin, -h/2 * Cos], dim=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat([point1, point2, point3, point4], dim=-1)

def obb2poly_3d(obboxes):
    """Return the 8-point representation of 3d OBB.
        OBB: [x, y, z, w, h, d, theta]"""
    obboxes_2d = torch.cat([obboxes[..., 0:2], obboxes[..., 3:5], obboxes[..., -1::6]], dim=-1)
    polys_2d = obb2poly(obboxes_2d) # [x1, y1, x2, y2, x3, y3, x4, y4]
    point1, point2, point3, point4 = torch.split(polys_2d, [2, 2, 2, 2], dim=-1)

    h_bias = obboxes[..., 5::7] / 2
    z0, z1 = obboxes[..., 2::7] - h_bias, obboxes[..., 2::7] + h_bias
    lower_points = torch.cat([point1, z0, point2, z0, point3, z0, point4, z0], dim=-1)
    upper_points = torch.cat([point1, z1, point2, z1, point3, z1, point4, z1], dim=-1)
    poly_3d = torch.cat([lower_points, upper_points], dim=-1)
    
    return poly_3d


def obb2hbb(obboxes):
    """Return the smallest AABB that contains the OBB."""
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w/2 * Cos) + torch.abs(h/2 * Sin)
    y_bias = torch.abs(w/2 * Sin) + torch.abs(h/2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center-bias, center+bias], dim=-1)

def obb2hbb_3d(obboxes):
    """Return the smallest 3d AABB that contains the 3d OBB."""
    center, z, w, h, d, theta = torch.split(obboxes, [2, 1, 1, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w/2 * Cos) + torch.abs(h/2 * Sin)
    y_bias = torch.abs(w/2 * Sin) + torch.abs(h/2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    z_bias = d / 2
    return torch.cat([center-bias, z-z_bias, center+bias, z+z_bias], dim=-1)

def obb2points_3d(obboxes):
    """ preprocessing for 2d loss """
    # [x, y, z, w, l, h, theta]
    center, w, l, h, theta = torch.split(obboxes, [3, 1, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector = torch.cat([w/2 * Cos -l/2 * Sin, w/2 * Sin + l/2 * Cos, h/2], dim=-1)
    return torch.cat([center-vector, center+vector], dim=0)