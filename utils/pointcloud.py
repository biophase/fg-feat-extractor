import torch

def voxel_hash (pcd_inds:torch.Tensor)->torch.Tensor:
    """Generates a 1D hash from a 3D voxel index array

    Args:
        pcd_inds (torch.Tensor): Nx3 index array (dtype=int32/int64)

    Returns:
        torch.Tensor: N, flat index array
    """

    assert (pcd_inds.dtype == torch.int32) or (pcd_inds.dtype == torch.int64),\
        f"Input should be either int32 or int64 but got {pcd_inds.dtype}"
    dims = pcd_inds.shape[1]
    flattened_inds = torch.zeros(pcd_inds.shape[0],dtype=torch.long).to(pcd_inds.device)
    pcd_inds_max = (pcd_inds).amax(dim=0)+1

    for i in range(dims):
        factor = torch.prod(torch.cat([pcd_inds_max[i+1:]]))
        flattened_inds += pcd_inds[:,i] * factor
    return flattened_inds



def grid_subsample(pcd_points:torch.Tensor, voxel_size:float, pcd_feats:torch.Tensor=None)->torch.Tensor:

    """returns the mean point (barycenter) of each voxel

    Args:
        pcd_points (torch.Tensor): Input points xyz (Nx3)
        pcd_feats (torch.Tensor): Input features    (NxF)
        voxel_size (float): Voxel size

    Returns:
        points (torch.Tensor): Baricenters xyz (Mx3)
        feats  (torch.Tensor): Mean feature    (MxF)
    """


    
    if pcd_feats is not None:
        assert pcd_points.shape[0] == pcd_feats.shape[0],\
            f"pcd_points and pcd_feats should have the same length but got {pcd_points.shape[0]} and {pcd_feats.shape[0]}"
        pcd_points_feats = torch.cat([pcd_points,pcd_feats],dim=1)
    else:
        pcd_points_feats = pcd_points
    
    N, xyzF = pcd_points_feats.shape

    # find 3D voxel index of each point
    # pcd_inds = ((pcd_points)//voxel_size).long()
    pcd_inds = torch.div(pcd_points, voxel_size, rounding_mode='floor').long()
    # 3D voxel array -> 1D voxel array
    flat_inds = voxel_hash(pcd_inds)
    # find occupied voxels
    _, inv_inds, bin_counts = torch.unique(flat_inds, return_inverse=True, return_counts=True)
    binned_points_feats = torch.zeros(size=(bin_counts.shape[0], xyzF)).to(pcd_points.device) # num_occupied_vox x 3
    # sum points in each bin
    binned_points_feats.scatter_add_(
        src=pcd_points_feats,
        index=inv_inds[:,None].expand(-1,xyzF),
        dim=0
    )
    # find the mean of each bin
    binned_points_feats /= bin_counts[:,None]

    return dict(
        points = binned_points_feats[:,:3],
        feats  = binned_points_feats[:,3:] if pcd_feats is not None else None,
        bin_counts = bin_counts,
        inv_inds = inv_inds
    )