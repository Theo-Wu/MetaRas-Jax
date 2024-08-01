import numpy as np
import jax.numpy as jnp
import os
from skimage.io import imread
from util_parallel import normalize

# def load_obj(filename_obj, merge=False, normalization=False, load_texture=False, texture_res=4, texture_type='surface'):
#     """
#     Load Wavefront .obj file.
#     This function only supports vertices (v x x x) and faces (f x x x).
#     """
#     assert texture_type in ['surface', 'vertex']

#     with open(filename_obj) as f:
#         lines = f.readlines()

#     # load vertices
#     vertices = []
    
#     for line in lines:
#         if len(line.split()) == 0:
#             continue
#         if line.split()[0] == 'v':
#             vertices.append([float(v) for v in line.split()[1:4]])
#     # vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()
#     vertices = np.vstack(vertices).astype(np.float32)

#     # load faces
#     faces = []
#     normal = []

#     for line in lines:
#         if len(line.split()) == 0:
#             continue
#         if line.split()[0] == 'f':
#             vs = line.split()[1:]
#             nv = len(vs)
#             v0 = int(vs[0].split('/')[0])
#             if merge:
#                 normal.append(int(vs[0].split('/')[-1]))
#             for i in range(nv - 2):
#                 v1 = int(vs[i + 1].split('/')[0])
#                 v2 = int(vs[i + 2].split('/')[0])
#                 faces.append([v0, v1, v2])

#     if merge:
#         quad_faces = []
#         for i in range(1,np.max(normal)+1):
#             indices = [k for k in range(len(normal)) if normal[k] == i]
#             assert len(indices) == 2, "Only two triangles are merged into one quadrangle, but len(indices) is {}".format(len(indices))
#             faces[indices[0]].append(faces[indices[1]][0])
#             quad_faces.append(faces[indices[0]])
#         faces = np.vstack(quad_faces).astype(np.int32) - 1
#     else:
#         # faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1
#         faces = np.vstack(faces).astype(np.int32) - 1

#     # load vertex normal
#     vertex_normals = []
#     for line in lines:
#         if len(line.split()) == 0:
#             continue
#         if line.split()[0] == 'vn':
#             vertex_normals.append([float(v) for v in line.split()[1:4]])
#     vertex_normals = np.vstack(vertex_normals).astype(np.float32)

#     # load textures
#     if load_texture and texture_type == 'surface':
#         textures = None
#         for line in lines:
#             if line.startswith('mtllib'):
#                 filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
#                 textures = load_textures(filename_obj, filename_mtl, texture_res)
#         if textures is None:
#             raise Exception('Failed to load textures.')
#     elif load_texture and texture_type == 'vertex':
#         textures = []
#         for line in lines:
#             if len(line.split()) == 0:
#                 continue
#             if line.split()[0] == 'v':
#                 textures.append([float(v) for v in line.split()[4:7]])
#         # textures = torch.from_numpy(np.vstack(textures).astype(np.float32)).cuda()
#         textures = np.stack([textures]).astype(np.float32)

#     # normalize into a unit cube centered zero
#     if normalization:
#         vertices -= vertices.min(0)[None, :]
#         vertices /= np.abs(vertices).max()
#         vertices *= 2
#         vertices -= vertices.max(0)[None, :] / 2

#     if load_texture:
#         return vertices, faces, vertex_normals, textures
#     else:
#         return vertices, faces, vertex_normals

def load_obj(filename_obj, merge=False, normalization=False, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    assert texture_type in ['surface', 'vertex']

    with open(filename_obj) as f:
        lines = f.readlines()

    vertices = []
    faces = []
    vertex_normals = []

    for line in lines:
        if len(line.split()) == 0:
            continue
            
        # load vertices
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
                
        # load normal
        if line.split()[0] == 'vn':
            vertex_normals.append([float(v) for v in line.split()[1:4]])
     
    vertices = np.vstack(vertices).astype(np.float32)
    vertex_normals = np.vstack(vertex_normals).astype(np.float32)
    
    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :] #[0,max-min]
        vertices /= np.abs(vertices).max() #[0,1]
        vertices *= 2 
        vertices -= vertices.max(0)[None, :] / 2
        
        # vertex_normals -= vertex_normals.min(0)[None, :]
        # vertex_normals /= np.abs(vertex_normals).max()
        # vertex_normals *= 2
        # vertex_normals -= vertex_normals.max(0)[None, :] / 2

    # average the normal
    normals_aver = np.zeros_like(vertices)
    normal_index = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])-1
            n0 = int(vs[0].split('/')[-1])-1
            if merge:
                normal_index.append(n0)
            normals_aver[v0] += vertex_normals[n0]
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])-1 
                v2 = int(vs[i + 2].split('/')[0])-1
                faces.append([v0, v1, v2])
                n1 = int(vs[i+1].split('/')[-1])-1
                n2 = int(vs[i+2].split('/')[-1])-1
                normals_aver[v1] += vertex_normals[n1]
                normals_aver[v2] += vertex_normals[n2]

    if merge:
        quad_faces = []
        for i in range(1,np.max(normal_index)+1):
            indices = [k for k in range(len(normal_index)) if normal_index[k] == i]
            assert len(indices) == 2, "Only two triangles are merged into one quadrangle, but len(indices) is {}".format(len(indices))
            faces[indices[0]].append(faces[indices[1]][0])
            quad_faces.append(faces[indices[0]])
        faces = np.vstack(quad_faces).astype(np.int32)
    else:
        faces = np.vstack(faces).astype(np.int32)
    normals_aver = normalize(normals_aver,-1)
    return vertices, faces, vertex_normals, normals_aver

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames

def load_textures(filename_obj, filename_mtl, texture_res):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = jnp.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = jnp.vstack(faces).astype(jnp.int32) - 1
    faces = vertices[faces]
    # faces = torch.from_numpy(faces).cuda()
    faces[1 < faces] = faces[1 < faces] % 1

    colors, texture_filenames = load_mtl(filename_mtl)

    # textures = torch.ones(faces.shape[0], texture_res**2, 3, dtype=torch.float32)
    # textures = textures.cuda()
    textures = jnp.ones(faces.shape[0], texture_res**2, 3, dtype=jnp.float32)

    #
    for material_name, color in list(colors.items()):
        # color = torch.from_numpy(color).cuda()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :] = color[None, :]

    for material_name, filename_texture in list(texture_filenames.items()):
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, -1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # pytorch does not support negative slicing for the moment
        image = image[::-1, :, :]
        # image = torch.from_numpy(image.copy()).cuda()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        # is_update = torch.from_numpy(is_update).cuda()
        textures = load_textures_cuda.load_textures(image, faces, textures, is_update)
    return textures
