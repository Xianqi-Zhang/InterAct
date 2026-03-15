import trimesh
import os
import argparse
import hashlib
import numpy as np

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

def stable_seed(dataset, obj_name):
    key = f"{dataset}/{obj_name}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample object points from canonical meshes.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="behave,intercap,grab,omomo,arctic",
        help="Comma-separated dataset names to process.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory.")
    parser.add_argument("--id-root", type=str, default="./assets/sample_objids", help="Root sample-id directory.")
    parser.add_argument(
        "--fallback-sample-count",
        type=int,
        default=340,
        help="If sample ids are missing, sample this many mesh vertices deterministically.",
    )
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    data_root = args.data_root
    id_root = args.id_root

    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        object_path = os.path.join(dataset_path, "objects")
        id_path = os.path.join(id_root, dataset)
        if not os.path.isdir(object_path):
            print(f"[WARN] Skip dataset={dataset}: missing objects directory.")
            continue
        has_id_dir = os.path.isdir(id_path)
        if not has_id_dir:
            print(f"[WARN] Missing sample-id directory for dataset={dataset}: {id_path}. Will use fallback sampling.")
        object_names = [x for x in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, x))]
        for obj_name in object_names:
            mesh_path = resolve_obj_mesh(object_path, obj_name)
            if mesh_path is None:
                print(f"[WARN] Missing mesh for dataset={dataset} object={obj_name}; skip.")
                continue
            mesh_obj = trimesh.load(mesh_path, force="mesh")
            ids_path = os.path.join(id_path, f"{obj_name}.npy")
            if has_id_dir and os.path.exists(ids_path):
                sample_ids = np.load(ids_path)
                obj_points = mesh_obj.vertices[sample_ids]
            else:
                verts = mesh_obj.vertices
                if verts.shape[0] == 0:
                    print(f"[WARN] Empty mesh vertices for dataset={dataset} object={obj_name}; skip.")
                    continue
                rng = np.random.default_rng(stable_seed(dataset, obj_name))
                count = min(args.fallback_sample_count, verts.shape[0])
                sampled_idx = rng.choice(verts.shape[0], size=count, replace=False)
                obj_points = verts[sampled_idx]
                os.makedirs(id_path, exist_ok=True)
                np.save(ids_path, sampled_idx)
                print(
                    f"[WARN] Fallback sampling used for dataset={dataset} object={obj_name}: "
                    f"{obj_points.shape[0]} points, ids saved to {ids_path}."
                )
            np.save(os.path.join(object_path, obj_name, "sample_points.npy"), obj_points)
   
        
