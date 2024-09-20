from copy import deepcopy
import typing as t

import torch

# rho parametrization
# a     b
# b*  1 - a

# measurements parametrization
# (x, y, z, t)

# rho -> m
# (a, 1 - a, 0.5 + Re(b), 0.5 + Im(b))


# m -> rho

# {0, 1}    {0, 2}         {0, 3}
#  x 0     x   z-0.5      x   i(t-0.5)   
#  0 y   z-0.5  1-x    -i(t-0.5)  1-x

#           {1, 2}         {1, 3}
#          1-y  z-0.5    1-y  i(t-0.5)
#         x-0.5   y    -i(t-0.5)   y

# 			               {2, 3}
# 		                   0.5  (z-0.5)+(t-0.5)i
#                         (z-0.5)-(t-0.5)i   0.5 


MEASUREMENT_ID_TO_RHO = {
    0: lambda x: torch.stack([
        torch.stack([x, torch.zeros_like(x)], dim=-1),
        torch.stack([torch.zeros_like(x), 1-x], dim=-1)
    ], dim=-2),
    1: lambda y: torch.stack([
        torch.stack([1-y, torch.zeros_like(y)], dim=-1),
        torch.stack([torch.zeros_like(y), y], dim=-1)
    ], dim=-2),
    2: lambda z: torch.stack([
        torch.stack([torch.zeros_like(z), z-0.5], dim=-1),
        torch.stack([z-0.5, torch.zeros_like(z)], dim=-1)
    ], dim=-2),
    3: lambda t: torch.stack([
        torch.stack([torch.zeros_like(t), 1.j*(t-0.5)], dim=-1),
        torch.stack([1.j*(-t+0.5), torch.zeros_like(t)], dim=-1)
    ], dim=-2)
}


def reconstruct_1qbit_hlp(measurements: torch.Tensor, measurements_ids: t.List[int]):
    # assert no duplicates in measurements_ids
    assert len(measurements_ids) == len(set(measurements_ids)), 'Duplicate measurements'
    current_measurement_ids = deepcopy(measurements_ids)
    current_measurements = measurements.clone().to(torch.complex64)
    if (0 in measurements_ids) and (1 in measurements_ids):
        index_1 = measurements_ids.index(1)
        current_measurement_ids.remove(1)
        current_measurements = torch.cat((current_measurements[..., :index_1], current_measurements[..., index_1+1:]), dim=-1)
    if (0 not in measurements_ids) and (1 not in measurements_ids):
        current_measurement_ids.append(0)
        current_measurements = torch.cat((current_measurements, torch.full_like(current_measurements[..., 0:1], 0.5)), dim=-1)
    rho = torch.stack(
        [
            MEASUREMENT_ID_TO_RHO[measurement_id](current_measurements[..., index])
            for index, measurement_id in enumerate(current_measurement_ids)
        ]
    ).sum(dim=0)
    return rho