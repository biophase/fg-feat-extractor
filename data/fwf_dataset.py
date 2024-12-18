from typing import List, Optional, Dict
from copy import deepcopy

from plyfile import PlyData, PlyElement
import pandas as pd
import numpy as np
from glob import glob
import json
import os
from omegaconf import OmegaConf
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader
import torch


from data.transforms import *

from utils.pointcloud import grid_subsample_simple, BBox



def get_label_structure_from_file(cfg):
    with open(os.path.join(cfg.data.dataset_root,"class_dict.json"),"r") as f:
        class_dict =  json.load(f) 
        return {k: len(class_dict[k]) for k in class_dict.keys() if k in cfg.data.label_names}


class FwfDataset(Dataset):
    def __init__(self,
                 cfg,
                 cfg_transforms,
                 cfg_projDataDescrs,
                 return_projIdx = False,
                 return_resiIdx = False,
                 ):
        
        super(Dataset,self).__init__()
        self.cfg = cfg
        
        # self.proj_query_list = proj_query_list
        # self.return_fields_input = return_fields_input
        # self.return_waveform = return_waveform
        
        
        # keep track of point cloud sizes
        self.proj_lens = []
        self.return_projIdx = return_projIdx
        self.return_resiIdx = return_resiIdx

        # build transforms
        self.transforms_dict = {}
        for feat_key, t_list_cfg in cfg_transforms.items():
            t_list = []
            for transform_name, params in t_list_cfg.items():
                transform_cls = globals().get(transform_name)
                if not transform_cls:
                    raise ValueError(f"Unknown transform: {transform_name}")
                transform = transform_cls(**params)
                t_list.append(transform)
            self.transforms_dict[feat_key] = TransformsList(t_list)
        
        
        # go through search pattern
        counter = 0
        self.projects = list()
        for fwf_proj_dataDescr in cfg_projDataDescrs:

            fwf_proj_fp = os.path.join(cfg.data.dataset_root, fwf_proj_dataDescr['proj_name'])
            fwf_proj_bboxes = fwf_proj_dataDescr['bboxes'] if 'bboxes' in fwf_proj_dataDescr.keys() else 'default'
            
            proj_name = fwf_proj_dataDescr['proj_name']
            

            print(f"Loading '{proj_name}'; Bounding box IDs = {fwf_proj_bboxes if len(fwf_proj_bboxes) else 'default'}")

            # load the data
            pcd = pd.DataFrame(PlyData.read(list(glob(os.path.join(fwf_proj_fp,'labeled','*pointcloud.ply')))[0]).elements[0].data)
            wfm = np.load(list(glob(os.path.join(fwf_proj_fp,'labeled','*waveform.npy')))[0]).astype(np.float32)
            meta = json.load(open(list(glob(os.path.join(fwf_proj_fp,'labeled','*metadata.json')))[0],"r"))
            
            # field names (constants)
            riegl_feat_names = [ 'riegl_reflectance','riegl_amplitude', 'riegl_deviation', 'riegl_targetIndex','riegl_targetCount']
            # label_names = ['labels_0', 'labels_1', 'labels_2', 'labels_3']
            self.label_names:Optional[List[str]] = cfg.data.label_names
            
            
            # convert to numpy
            riegl_feats = pcd[riegl_feat_names].to_numpy()
            rgb = pcd[['Red','Green','Blue']].to_numpy()
            
            # normalize the data
            # FIXME: Statistics calculated on first point cloud only
            wfm = (wfm.astype(np.float32) - 358.35934) / 623.0141 # - mean / std

            riegl_feats -= np.array([[-6.90191949, 25.16398933, 26.45952891,  1.        ,  1.03636612]]) # - means
            riegl_feats /= np.array([[  2.32590898, 2.98518547, 929.71399545, 1., 0.22651793]]) # /std
            
            rgb -= np.array([[0.29689665, 0.3428666,  0.190237]]) # means
            rgb /= np.array([[0.21558372, 0.23351644, 0.21213871]]) # std          
            

            
            # get scan positions
            if 'scanId=000' in meta['scan_positions'].keys():
                sop = np.array([meta['scan_positions'][f'scanId={si:03}']['sop'] for si in range(len(meta['scan_positions']))])
            else:
                # handle case where only one scan position is in the metadata and it's f.s.r. labeled 'scanId=001' instead of 'scanID=000'
                sop = np.array([meta['scan_positions']['scanId=001']['sop']])

            # get full xyz
            xyz_defaultBbox = pcd[['x','y','z']].to_numpy()       

                
            # save project as dict
            if len(fwf_proj_bboxes)==0 or 'bboxes' not in meta.keys():
                # handle default case
            
                self.projects.append(dict(
                    proj_name=f"{proj_name}::defaultBbox",
                    xyz = xyz_defaultBbox,
                    wfm = wfm,
                    sop = sop,
                    rgb = rgb,
                    riegl_feats = riegl_feats,
                    labels = pcd[self.label_names].to_numpy(),
                    sop_ids = pcd['scan_id'].to_numpy(),
                    kd_tree = KDTree(xyz_defaultBbox),
                ))
                
            else:
                # handle region bboxes case
                for bbox_i in fwf_proj_bboxes:
                    bbox_meta = meta['bboxes'][f'bboxId={bbox_i:03}']
                    bbox = BBox(orientation=np.array(bbox_meta['orientation']), width=np.array(bbox_meta['width']))
                    subcloud_mask=bbox.cutout(xyz_defaultBbox)
                    xyz_masked = xyz_defaultBbox[subcloud_mask]
                    self.projects.append(dict(
                        proj_name=f"{proj_name}::bboxId={bbox_i:03}",
                        xyz = xyz_masked,
                        wfm = wfm[subcloud_mask],
                        sop = sop,
                        rgb = rgb[subcloud_mask],
                        riegl_feats = riegl_feats[subcloud_mask],
                        labels = pcd[self.label_names].to_numpy()[subcloud_mask], # labels need to be subsampeld
                        sop_ids = pcd['scan_id'].to_numpy()[subcloud_mask],
                        kd_tree = KDTree(xyz_masked),

                    ))
                    
                    
            # calculate the cumulative sum of the point cloud sizes




    def load_class_weights(self, kind ='full'):
        with open(os.path.join(self.cfg.data.dataset_root,'class_weights.json'),'r') as f:
            self.statistics = json.load(f)[kind]
            
            self.class_weights = deepcopy(self.statistics)

            for level in self.class_weights.keys():
                counts = self.class_weights[level]
                self.class_weights[level] = np.array([(1/(n_k+1))/sum([1/(n_j+1) for n_j in counts]) for n_k in counts])

    def subsample_grid(self, grid_size:float, save_inv=True):
        self.proj_lens = []
        if not grid_size: grid_size = self.cfg.data.query_grid_size
        for i, proj in enumerate(self.projects):
            sub = grid_subsample_simple(proj['xyz'],grid_size, 'cuda' if self.cfg.data.subsample_on_gpu else 'cpu')
            _, sub_ids = proj['kd_tree'].query(sub['points'])
            self.projects[i].update(dict(
                labels_sub = proj['labels'][sub_ids], # labels need to be subsampeld 
                xyz_sub = sub['points'],
                sub_inv = sub['inv_inds'] if save_inv else None
            ))
            self.proj_lens.append(sub['points'].shape[0])
        self.proj_lens_cumsum = np.cumsum(self.proj_lens)

    def subsample_random(self, sample_ratio: float, save_inv=True, weighted=False):

        assert 0 < sample_ratio <= 1, "Sample ratio must be between 0 and 1."
        
        self.proj_lens = []
        for i, proj in enumerate(self.projects):
            num_points = proj['xyz'].shape[0]
            sample_size = int(sample_ratio * num_points)
            # assume uniform distribution for sampling
            if not weighted: 
                sampled_ids = np.random.choice(num_points, sample_size, replace=True)
            # distribute inversely proportional to the class frequency
            else:
                weights = []
                for li, label_name in enumerate(self.cfg.data.label_names):
                    labels = proj['labels'][:,li]
                    weights_level = np.array(self.class_weights[label_name][labels],dtype=np.float64)
                    weights_level /= weights_level.sum()
                    weights.append(weights_level[:, None])
                # average over all levels
                weights = np.concatenate(weights,axis=-1).mean(axis=-1)
                sampled_ids = np.random.choice(np.arange(num_points), sample_size, replace=True, p=weights)

            self.projects[i].update(dict(
                labels_sub = proj['labels'][sampled_ids],  # Subsample labels
                xyz_sub = proj['xyz'][sampled_ids],  # Subsample coordinates
                sub_inv = np.argsort(sampled_ids) if save_inv else None  # Save inverse indices if needed
            ))
            self.proj_lens.append(len(sampled_ids))
        
        self.proj_lens_cumsum = np.cumsum(self.proj_lens)
    
    def compute_neibors_knn(self, k:int, verbose=True):
        for proj in self.projects:
            if verbose: print(f"Computing neibors for '{proj['proj_name']}' @ k={k}")
            dists, neib_ids = proj['kd_tree'].query(proj['xyz_sub'], k)
            proj['neibors'] = neib_ids
    

   
    def compute_normals_knn(self, extract_geometric = None, verbose=True):
        for proj in self.projects:
            if extract_geometric == None:
                extract_geometric = 'geom_feats' in self.cfg.data.scalar_input_fields
            k = proj['neibors'].shape[1]
            if verbose: print(f"Computing normals {'and geom_feats ' if extract_geometric else ''}for '{proj['proj_name']}' @ k={k}")
            neibs_xyz = proj['xyz'][proj['neibors']]

            means = neibs_xyz.mean(axis=1, keepdims=True)
            neibs_xyz -= means
            cov = (neibs_xyz.transpose([0,2,1]) @ neibs_xyz) / (k-1)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            eigenvals = np.clip(eigenvals, a_min=1e-10, a_max=None)
            # get non-flipped normals
            normals = eigenvecs[:, :, 0]

            # upsample normals to full resolution
            normals = normals[proj['sub_inv']]

            # move all points to scanner CS
            points_origin_scanPos = proj['sop'][proj['sop_ids']][:,:3,3]
            xyz_scannerCs = proj['xyz'] - points_origin_scanPos
            signs = np.sign(np.squeeze(xyz_scannerCs[:,None,:] @ normals [:,:,None])) * -1
            normals *= signs[:,None]

            proj['normals'] = normals

            if extract_geometric:
                eigenvals = np.sort(eigenvals)[:,::-1]
                # Compute geometric features
                linearity = (eigenvals[:, 0] - eigenvals[:, 1]) / (eigenvals[:, 0] + 1e-10)
                planarity = (eigenvals[:, 1] - eigenvals[:, 2]) / (eigenvals[:, 0] + 1e-10)
                sphericity = eigenvals[:, 2] / (eigenvals[:, 0] + 1e-10)
                omnivariance = np.cbrt(eigenvals.prod(axis=1))
                anisotropy = (eigenvals[:, 0] - eigenvals[:, 2]) / (eigenvals[:, 0] + 1e-10)
                eigenentropy = -np.sum((eigenvals / (eigenvals.sum(axis=1, keepdims=True) + 1e-10)) *
                                    np.log(eigenvals / eigenvals.sum(axis=1, keepdims=True) + 1e-10), axis=1)
                
                # Combine features into a single matrix (N x 6) by stacking and upsampling
                geom_feats = np.stack([
                    linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy
                ], axis=1)[proj['sub_inv']]
                proj['geom_feats'] = geom_feats
                

       
    def compute_incAngles(self, verbose=True):
        for proj in self.projects:
            if verbose: print(f"Computing incidence angles for '{proj['proj_name']}'")
            xyz_scannerCs = proj['xyz'] - proj['sop'][proj['sop_ids']][:,:3,3]
            proj['incAngles']= np.arccos(np.squeeze((proj['normals'][:,None,:] @ xyz_scannerCs[...,None])) / \
                (np.linalg.norm(xyz_scannerCs,axis=-1) * np.linalg.norm(proj['normals'],axis=-1)))
            proj['distanceFromScanner'] = np.linalg.norm(xyz_scannerCs, axis=-1)
            
            # reshape by adding an additional axis
            proj['incAngles'] = proj['incAngles'][:,None]
            proj['distanceFromScanner'] = proj['distanceFromScanner'][:,None]
            
    
    def __getitem__(self, index):
        # get proj index first
        proj_idx = np.argwhere(index<self.proj_lens_cumsum)[0][0]
        size_prev = self.proj_lens_cumsum[proj_idx-1] if proj_idx > 0 else 0
        residual_idx = index - size_prev
        
        # get neibors of point at index
        neibs = self.projects[proj_idx]['neibors'][residual_idx]

        return_dict = dict(
            features_neibors = np.concatenate(
                [
                    self.transforms_dict[f](self.projects[proj_idx][f][neibs])
                    if f in self.transforms_dict else self.projects[proj_idx][f][neibs]
                    for f in self.cfg.data.scalar_input_fields
                ],
                axis=-1
            ).astype(np.float32),
            features_point = np.concatenate(
                [
                    self.transforms_dict[f](self.projects[proj_idx][f][residual_idx][None,:])
                    if f in self.transforms_dict else self.projects[proj_idx][f][residual_idx][None,:]
                    for f in self.cfg.data.scalar_input_fields
                ],
                axis=-1
            ).astype(np.float32)
        )

        # add waveforms
        return_dict.update(dict(
            wfm_neibors = self.projects[proj_idx]['wfm'][neibs] if self.cfg.data.fw_input_field else None,
            wfm_point = self.projects[proj_idx]['wfm'][residual_idx][None,:] if self.cfg.data.fw_input_field else None,
        )) # type:ignore

        # add labels as separate entries
        for label_i, label_name in enumerate(self.label_names):
            return_dict.update({
                label_name : self.projects[proj_idx]['labels_sub'][residual_idx][label_i]
            })
        
        # return inverse indices for testing
        if self.return_projIdx:
            return_dict.update(dict(
                proj_idx = proj_idx
            ))
        if self.return_resiIdx:
            return_dict.update(dict(
                residual_idx = residual_idx
            ))    


        
        
        return return_dict
        
    
    def __len__(self):
        return np.sum([p['xyz_sub'].shape[0] for p in self.projects])
        


