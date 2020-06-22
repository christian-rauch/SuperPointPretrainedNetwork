#!/usr/bin/env python3
from sklearn.decomposition import PCA, IncrementalPCA
import argparse
import os
import glob
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('export_path', type=str, help="batch export input images and feature maps")
    parser.add_argument('--ncomponents', type=int, default=3, help="number of PCA components")
    args = parser.parse_args()

    file_paths = glob.glob(os.path.join(args.export_path, "*.npz"))

    features = list()

    for file_path in file_paths:
        npz_file = np.load(file_path)
        I = npz_file['I']
        F = npz_file['F']
        npz_file.close()

        features.extend(F.reshape(-1, F.shape[-1]))

    features = np.array(features)

    ipca = IncrementalPCA(n_components=args.ncomponents)
    ipca.fit(features)
    np.savetxt(os.path.join(args.export_path, "components.csv"), ipca.components_)
