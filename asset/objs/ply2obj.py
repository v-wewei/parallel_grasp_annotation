import os

import trimesh


for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".ply"):
            filename = os.path.join(root, file)
            print(filename)
            mesh = trimesh.load(filename, process=False, force='mesh')
            mesh.visual = trimesh.visual.ColorVisuals()
            e = mesh.export(file.split(".")[0]+".obj", file_type='obj')
