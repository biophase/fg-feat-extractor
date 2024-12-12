from typing import List, Optional, Dict

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



def get_label_structure(cfg):
    with open(os.path.join(cfg.data.dataset_root,"class_dict.json"),"r") as f:
        class_dict =  json.load(f) 
        return {k: len(class_dict[k]) for k in class_dict.keys() if k in cfg.data.label_names}


class FwfDataset(Dataset):
    def __init__(self,
                 cfg,
                 cfg_transforms,
                 cfg_projDataDescrs,
                 ):
        
        super(Dataset,self).__init__()
        self.cfg = cfg
        
        # self.proj_query_list = proj_query_list
        # self.return_fields_input = return_fields_input
        # self.return_waveform = return_waveform
        
        
        # keep track of point cloud sizes
        self.proj_lens = []

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
                sub = grid_subsample_simple(xyz_defaultBbox,self.cfg.data.query_grid_size, 'cuda' if self.cfg.data.subsample_on_gpu else 'cpu')
                kd_tree = KDTree(xyz_defaultBbox)
                _, sub_ids = kd_tree.query(sub['points'])
                self.projects.append(dict(
                    proj_name=f"{proj_name}::defaultBbox",
                    xyz = xyz_defaultBbox,
                    wfm = wfm,
                    sop = sop,
                    rgb = rgb,
                    riegl_feats = riegl_feats,
                    labels = pcd[self.label_names].to_numpy()[sub_ids], # labels need to be subsampeld 
                    sop_ids = pcd['scan_id'].to_numpy(),
                    kd_tree = kd_tree,
                    xyz_sub = sub['points'],
                    sub_inv = sub['inv_inds']
                ))
                self.proj_lens.append(sub['points'].shape[0])
            else:
                # handle region bboxes case
                for bbox_i in fwf_proj_bboxes:
                    bbox_meta = meta['bboxes'][f'bboxId={bbox_i:03}']
                    bbox = BBox(orientation=np.array(bbox_meta['orientation']), width=np.array(bbox_meta['width']))
                    subcloud_mask=bbox.cutout(xyz_defaultBbox)
                    xyz_masked = xyz_defaultBbox[subcloud_mask]
                    sub = grid_subsample_simple(xyz_masked, self.cfg.data.query_grid_size, 'cuda' if self.cfg.data.subsample_on_gpu else 'cpu')
                    kd_tree = KDTree(xyz_masked)
                    _, sub_ids = kd_tree.query(sub['points'])
                    self.projects.append(dict(
                        proj_name=f"{proj_name}::bboxId={bbox_i:03}",
                        xyz = xyz_masked,
                        wfm = wfm[subcloud_mask],
                        sop = sop,
                        rgb = rgb[subcloud_mask],
                        riegl_feats = riegl_feats[subcloud_mask],
                        labels = pcd[self.label_names].to_numpy()[subcloud_mask][sub_ids], # labels need to be subsampeld
                        sop_ids = pcd['scan_id'].to_numpy()[subcloud_mask],
                        kd_tree = KDTree(xyz_masked),
                        xyz_sub = sub['points'],
                        sub_inv = sub['inv_inds']
                    ))
                    self.proj_lens.append(sub['points'].shape[0])
                    
            # calculate the cumulative sum of the point cloud sizes

            if counter >= 3:
                continue
                break
            else:
                counter += 1
        self.proj_lens_cumsum = np.cumsum(self.proj_lens)
    
    def compute_neibors_knn(self, k:int):
        for proj in self.projects:
            print(f"Computing neibors for '{proj['proj_name']}' @ k={k}")
            dists, neib_ids = proj['kd_tree'].query(proj['xyz_sub'], k)
            proj['neibors'] = neib_ids
    

   
    def compute_normals_knn(self):
        for proj in self.projects:
            k = proj['neibors'].shape[1]
            print(f"Computing normals for '{proj['proj_name']}' @ k={k}")
            neibs_xyz = proj['xyz'][proj['neibors']]

            means = neibs_xyz.mean(axis=1, keepdims=True)
            neibs_xyz -= means
            cov = (neibs_xyz.transpose([0,2,1]) @ neibs_xyz) / (k-1)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
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


            
    def compute_incAngles(self):
        for proj in self.projects:
            print(f"Computing incidence angles for '{proj['proj_name']}'")
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
                label_name : self.projects[proj_idx]['labels'][residual_idx][label_i]
            })


        
        
        return return_dict
        
    
    def __len__(self):
        return np.sum([p['xyz_sub'].shape[0] for p in self.projects])
        


