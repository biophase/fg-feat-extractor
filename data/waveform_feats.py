import torch
import numpy as np

def compute_shifted_fast(fwf, eps=3_000_000):
    """Shifts the waveform, such that the lowest non-zero value is at 0. 
    The shifted waveform is returned, as well as the shift values for each sample."""
    fwf_shifted = fwf.copy()
    dum = fwf.copy()
    mask = dum==0
    dum[mask] += eps
    shifts = dum.min(axis=1)
    fwf_shifted += (np.ones_like(fwf) * mask) * shifts[:,None]
    fwf_shifted -= shifts[:,None]
    return fwf_shifted, shifts

def find_derivative_batch(x):
    return (x[:, 1:] - x[:, :-1]) / 1

find_derivative = lambda x: (x[1:]-x[:-1])/1


def find_peaks (d1: torch.Tensor, d2: torch.Tensor, min_peak_curvature: float = 0.0):
    d1_shifted = torch.roll(d1,-1)
    # d2 = torch.roll(d2, 1)
    signs_d1 = torch.sign(d1)
    signs_d1_shifted = torch.sign(d1_shifted)
    signs_d1[torch.eq(signs_d1,0)] = 1
    signs_d1_shifted[torch.eq(signs_d1_shifted,0)] = 1
    signs_d1 = signs_d1 != signs_d1_shifted
    # peaks_inds = torch.where(signs[:-1])[0].to(dtype=torch.float32)
    positive_d2 = torch.le(d2,min_peak_curvature)
    peaks_inds = torch.where(torch.logical_and(signs_d1[:-1],positive_d2))[0].to(dtype=torch.float32)+1
    

    return peaks_inds

def find_peaks_batch(d1, d2, min_peak_curvature=0.0):
    """Computes the peaks of the waveform, based on the first and second derivative.
    Peaks which don't reach a required curvature are discarded.

    Args:
        d1 (torch.Tensor): first derivative of the waveform.
        d2 (torch.Tensor): second derivative of the waveform
        min_peak_curvature (float, optional): Curvature threshold. Defaults to 0.0.

    Returns:
        Tuple: (peaks_sample_idx, peaks_pos_idx)
    """

    d1_shifted = torch.roll(d1, -1, dims=1)
    signs_d1 = torch.sign(d1)
    signs_d1_shifted = torch.sign(d1_shifted)
    signs_d1[torch.eq(signs_d1, 0)] = 1
    signs_d1_shifted[torch.eq(signs_d1_shifted, 0)] = 1
    signs_d1 =  torch.not_equal(signs_d1, signs_d1_shifted)
    positive_d2 = torch.le(d2, min_peak_curvature)
    intermid =torch.logical_and(signs_d1[:, :-1], positive_d2)
    peaks_batch = torch.nonzero(intermid,as_tuple=False)
    peaks_sample_idx = peaks_batch[:,0]
    peaks_pos_idx = peaks_batch[:,1] + 1 
    
    return peaks_sample_idx, peaks_pos_idx