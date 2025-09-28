import torch
from einops import rearrange

def num2bits(value:torch.Tensor, precision:int, v_min=0, v_max=1) -> torch.Tensor:
    device = value.device

    _min = torch.full_like(value, v_min, device=device)
    _max = torch.full_like(value, v_max, device=device)
    v_bits = torch.zeros((precision, len(value)), dtype=torch.bool, device=device)
    for p in range(precision):
        mid = (_min + _max) / 2
        mask = value > mid
        v_bits[p] = mask
        _min = torch.where(mask, mid, _min)
        _max = torch.where(mask, _max, mid)
    return v_bits.T


def bits2num(bits:torch.Tensor, precision:int, v_min=0, v_max=1) -> torch.Tensor:
    device = bits.device

    _min = torch.full((bits.shape[0], ), v_min, device=device)
    _max = torch.full((bits.shape[0], ), v_max, device=device)

    for p in range(precision):
        mid = (_min + _max) / 2
        mask = bits[:, p]
        _min = torch.where(mask, mid, _min)
        _max = torch.where(mask, _max, mid)
    
    value = (_min + _max) / 2
    return value

def geohash_encoding(traj: torch.FloatTensor, precision = 40) -> torch.Tensor:

    b, l, c = traj.shape

    traj_lon = traj[:, :, 0].flatten()
    traj_lat = traj[:, :, 1].flatten()
    traj_lon = rearrange(
        num2bits(traj_lon, precision=precision, v_max=1, v_min=0), 
        '(b l) p -> b l p', l=l)
    traj_lat = rearrange(
        num2bits(traj_lat, precision=precision, v_max=1, v_min=0), 
        '(b l) p -> b l p', l=l)

    traj = torch.stack([traj_lon, traj_lat], dim = -1).long()

    return traj # [batch, len, precision, 2]

def geohash_decoding(traj: torch.BoolTensor, precision = 40) -> torch.Tensor:

    b, l, p, _ = traj.shape

    traj_lon = rearrange(traj[:, :, :, 0], 'b l p -> (b l) p')
    traj_lat = rearrange(traj[:, :, :, 1], 'b l p -> (b l) p')

    traj_lon = bits2num(traj_lon, precision=precision, v_max=1, v_min=0)
    traj_lat = bits2num(traj_lat, precision=precision, v_max=1, v_min=0)
    
    traj = torch.stack([rearrange(traj_lon, '(b l p) -> b l p', l=l, b=b)[:, :, 0],
                        rearrange(traj_lat, '(b l p) -> b l p', l=l, b=b)[:, :, 0]], dim = -1)
    
    return traj
