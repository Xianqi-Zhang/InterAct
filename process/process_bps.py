import os
import argparse
import os.path
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from bps_torch.bps import bps_torch

def resolve_obj_mesh(object_path, obj_name):
    obj_dir = os.path.join(object_path, obj_name)
    p1 = os.path.join(obj_dir, f"{obj_name}.obj")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(obj_dir, "mesh.obj")
    if os.path.exists(p2):
        return p2
    if os.path.isdir(obj_dir):
        cands = [x for x in os.listdir(obj_dir) if x.endswith(".obj")]
        if cands:
            return os.path.join(obj_dir, cands[0])
    return None


# visualize markers motion of smpl model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process object BPS features.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="behave,intercap,grab,omomo,arctic",
        help="Comma-separated dataset names to process.",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Root data directory.")
    args = parser.parse_args()

    # bps 
    bps_torch = bps_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bps_obj = np.load('assets/bps_basis_set_1024_1.npy')
    bps_obj = torch.from_numpy(bps_obj).float().to(device)
   
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    data_root = args.data_root
    for dataset in datasets:
        print(f'Loading {dataset} ...')
        dataset_path = os.path.join(data_root, dataset)
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        OBJECT_BPS_PATH = os.path.join(dataset_path, 'objects_bps')
        if not os.path.isdir(OBJECT_PATH):
            print(f"[WARN] Skip dataset={dataset}: missing objects directory.")
            continue
        
        os.makedirs(OBJECT_BPS_PATH, exist_ok=True)  # create folder if not exist
        data_name = [x for x in os.listdir(OBJECT_PATH) if os.path.isdir(os.path.join(OBJECT_PATH, x))]
        for k, name in tqdm(enumerate(data_name)):
            mesh_path = resolve_obj_mesh(OBJECT_PATH, name)
            if mesh_path is None:
                print(f"[WARN] Missing mesh for dataset={dataset} object={name}; skip.")
                continue
            mesh_obj = trimesh.load(mesh_path, force='mesh')
            obj_verts = mesh_obj.vertices
            obj_verts = (obj_verts)[None, ...]
            torch_obj_verts = torch.from_numpy(obj_verts).float().to(device)
            
            bps_object_geo = bps_torch.encode(x=torch_obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=bps_obj[None,...])['deltas'] # T X N X 3 
            bps_object_geo_np = bps_object_geo.data.detach().cpu().numpy()
            
            obj_bps_dir = os.path.join(OBJECT_BPS_PATH, name)
            os.makedirs(obj_bps_dir, exist_ok=True)

            np.save(os.path.join(obj_bps_dir, f"{name}_1024.npy"), bps_object_geo_np)
            

        
        
