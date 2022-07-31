import numpy as np
import os
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def readPly(filename):
    with open(filename, 'rb') as fid:
        plydata = PlyData.read(fid)
    vertex = plydata['vertex'].data
    data = [vertex[name].astype(np.float32) for name in vertex.dtype.names]
    vertex_with_props = np.array(data)
    return vertex_with_props.T


def savePly(point_cloud, filename):
    if os.path.isfile(filename):
        warnings.warn(f'File Exists!')
        return
    ncols = point_cloud.shape[1]
    py_types = (float, float, float, float, float, float,
                np.uint8, np.uint8, np.uint8, np.uint8)[:ncols]
    npy_types = [('x', 'f4'),   ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'),  ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'uint8'),  ('greed', 'uint8'), ('blue', 'uint8'), ('label', 'uint8')][:ncols]
    # format into NumPy structured array
    vertices = []
    for row_idx in range(point_cloud.shape[0]):
        point = point_cloud[row_idx]
        vertices.append(tuple(dtype(val)
                        for dtype, val in zip(py_types, point)))
    structured_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(structured_array, 'vertex')
    # write ply
    PlyData([el]).write(filename)
    print('Save:', filename)


def synthiaNpz2Ply(sourceDir, plyDir):
    filelist = os.listdir(sourceDir)
    for file in tqdm(filelist):
        if file[-len('.npz'):] != '.npz':
            continue
        saveFile = os.path.join(plyDir, file[:-len('.npz')]+'.ply')
        data = np.load(os.path.join(sourceDir, file))
        xyz = data['pc'].astype(np.float32).reshape(-1, 3)
        rgb = (data['rgb']*256).astype(np.uint8).reshape(-1, 3)
        lb = data['semantic'].astype(np.uint8).reshape(-1, 1)
        ptn = xyz.shape[0]
        emptyNormal = np.zeros((ptn, 3), dtype=np.float32)
        savePly(np.hstack((xyz, emptyNormal, rgb, lb)), saveFile)


def synthiaPly2Npz(sourceDir, prDir, saveDir):
    filelist = os.listdir(sourceDir)
    for file in tqdm(filelist):
        if file[-len('.npz'):] != '.npz':
            continue
        save_pth = os.path.join(saveDir, file)
        if os.path.exists(save_pth):
            continue
        srcData = np.load(os.path.join(sourceDir, file))
        xyz = srcData['pc']
        rgb = srcData['rgb']
        lb = srcData['semantic']
        center = srcData['center']
        ptn = xyz.shape[0]
        prData = readPly(os.path.join(prDir, file[:-len('.npz')]+'.ply'))
        pr = prData[:, 7]
        normal = prData[:, 3:6]
        para = prData[:, 8:12]
        assert prData.shape[0] == ptn
        np.savez(os.path.join(saveDir, file), center=center, pc=xyz, rgb=rgb,
                 semantic=lb,  normal=normal, primitive=pr, parameter=para)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PrimitiveFitting PreProcessing Config")
    parser.add_argument("--target", choices=("ply", "npz"),
                        default="ply",
                        help="target file format")
    parser.add_argument("--sourceDir", type=str,
                        default='./Data/SynSource',
                        help="input path of synthia4D source file(official .npz)")
    parser.add_argument("--plyDir", type=str,
                        default='./Data/Synthia4D',
                        help="output path of converted ply file(for primitive fitting)")
    parser.add_argument("--prDir", type=str,
                        default='./Data/output',
                        help="input path of fitted ply file to be converted")
    parser.add_argument("--npzDir", type=str,
                        default='./Data',
                        help="output path of npz file for training")
    args = parser.parse_args()

    if args.target == "ply":
        os.makedirs(args.plyDir, exist_ok=True)
        print("=> Converting from Npz source to Ply file")
        synthiaNpz2Ply(args.sourceDir, args.plyDir)
    elif args.target == "npz":
        os.makedirs(args.plyDir, exist_ok=True)
        print("=> Converting from Ply file to Npz file")
        synthiaPly2Npz(args.sourceDir, args.prDir, args.npzDir)
