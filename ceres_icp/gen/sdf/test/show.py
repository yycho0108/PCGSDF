#!/usr/bin/env python3

import numpy as np
import open3d as o3d

sc = np.loadtxt('/tmp/scene-cloud.csv')
# oc = np.loadtxt('/tmp/object-cloud.csv')
# oc = np.loadtxt('/tmp/scene-compiled-cloud.csv')
print(sc.min(), sc.max())

scp = o3d.geometry.PointCloud()
scp.points = o3d.utility.Vector3dVector(sc.astype(np.float64))

# ocp = o3d.geometry.PointCloud()
# ocp.points = o3d.utility.Vector3dVector(oc.astype(np.float64))

o3d.visualization.draw_geometries([scp])
# o3d.visualization.draw_geometries([ocp])

# o3d.visualization.draw_geometries([scp, ocp])
# o3d.visualization.draw_geometries([ocp])
