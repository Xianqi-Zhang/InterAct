import os
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonicalize object meshes and update object transforms.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="behave,intercap,grab,omomo,arctic",
        help="Comma-separated dataset names to process, e.g. 'arctic' or 'behave,arctic'.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory.")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    data_root = args.data_root
    for dataset in datasets:
        print("Processing dataset:", dataset)
        obj_dic = {}
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        if not os.path.isdir(MOTION_PATH) or not os.path.isdir(OBJECT_PATH):
            print(f"[WARN] Skip dataset={dataset}: missing sequences/objects directory.")
            continue
        obj_names = [name for name in os.listdir(OBJECT_PATH) if os.path.isdir(os.path.join(OBJECT_PATH, name))]

        def resolve_obj_mesh(obj_name: str) -> str | None:
            obj_dir = os.path.join(OBJECT_PATH, obj_name)
            # Common InterAct layout.
            p1 = os.path.join(obj_dir, f"{obj_name}.obj")
            if os.path.exists(p1):
                return p1
            # ARCTIC layout.
            p2 = os.path.join(obj_dir, "mesh.obj")
            if os.path.exists(p2):
                return p2
            # Fallback: first .obj in directory.
            cands = [x for x in os.listdir(obj_dir) if x.endswith('.obj')]
            if cands:
                return os.path.join(obj_dir, cands[0])
            return None

        for name in obj_names:
            print("Processing object:", name)
            mesh_path = resolve_obj_mesh(name)
            if mesh_path is None:
                print(f"[WARN] Missing mesh for object={name}; skip.")
                continue
            mesh_obj = trimesh.load(mesh_path, force='mesh')
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
            obj_dic[name] = obj_verts.mean(axis=0)
            mesh_obj.vertices = (obj_verts - obj_verts.mean(axis=0, keepdims=True))
            os.makedirs(os.path.join(OBJECT_PATH, name), exist_ok=True)
            # Export back to the same path we loaded to preserve dataset-specific naming.
            mesh_obj.export(mesh_path)
            
        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            print("Processing sequence:", name)
            if not os.path.exists(os.path.join(MOTION_PATH, name, 'object.npz')):
                continue
            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
            if obj_name not in obj_dic:
                print(f"[WARN] object.npz references unknown object={obj_name}; skip sequence={name}")
                continue

            rotation = Rotation.from_rotvec(obj_angles)

            
            new_obj_trans = obj_trans + rotation.apply(obj_dic[obj_name])
            
            
            obj = {
                'angles': obj_angles,
                'trans': new_obj_trans,
                'name': obj_name,
            }
            
            np.savez(os.path.join(MOTION_PATH, name, 'object.npz'), **obj)
            


        
