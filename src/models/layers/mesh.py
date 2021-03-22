import random
from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh
import chamferdist


def triangle_area(tri):
    x_min = min([point[0] for point in tri])
    x_max = max([point[0] for point in tri])
    y_min = min([point[1] for point in tri])
    y_max = max([point[1] for point in tri])
    area_rectangle = (y_max - y_min) * (x_max - x_min)
    t1 = 0.5 * abs((tri[0][0] - tri[1][0]) * (tri[0][1] - tri[1][1]))
    t2 = 0.5 * abs((tri[0][0] - tri[2][0]) * (tri[0][1] - tri[2][1]))
    t3 = 0.5 * abs((tri[1][0] - tri[2][0]) * (tri[1][1] - tri[2][1]))
    return area_rectangle - t1 - t2 - t3


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        self.path = file

        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        fill_mesh(self, file, opt)

        self.vs = torch.tensor(self.vs)
        self.faces = None

        self.export_folder = export_folder
        self.history_data = None
        if False and hold_history:
            self.init_history()

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
            'groups': [],
            'gemm_edges': [self.gemm_edges.copy()],
            'occurrences': [],
            'old2current': np.arange(self.edges_count, dtype=np.int32),
            'current2old': np.arange(self.edges_count, dtype=np.int32),
            'edges_mask': [torch.ones(self.edges_count, dtype=torch.bool)],
            'edges_count': [self.edges_count],
        }
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source],
                                                 self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)

    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas

    @staticmethod
    def subdivide(in_path, out_path, opts):
        mesh = Mesh(in_path, opts)
        vs = list(mesh.vs[mesh.v_mask])
        mesh.calculate_faces()
        new_indices = list(range(len(vs)))
        faces = mesh.faces

        new_faces = []
        new_edges = []
        for face in faces:
            face_coords = face

            face_center = (vs[face_coords[0]] + vs[face_coords[1]] + vs[face_coords[2]]) / 3  # TODO MAYBE BUG
            face_center_index = len(vs)
            vs.append(face_center)
            new_indices.append(len(vs))
            new_faces.extend([
                (face_coords[0], face_coords[1], face_center_index),
                (face_coords[0], face_coords[2], face_center_index),
                (face_coords[1], face_coords[2], face_center_index)
            ])

            new_edges.extend([
                [face_coords[0], face_center_index],
                [face_coords[1], face_center_index],
                [face_coords[2], face_center_index]
            ])

        mesh.faces = new_faces
        mesh.edges = list(mesh.edges) + new_edges
        mesh.vs = vs

        with open(out_path, 'w+') as f:
            for _, v in enumerate(vs):
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face_id in range(len(new_faces) - 1):
                f.write(
                    "f %d %d %d\n" % (new_faces[face_id][0] + 1, new_faces[face_id][1] + 1, new_faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in mesh.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

        return mesh

    def calculate_faces(mesh):
        faces = []
        vs = list(mesh.vs[mesh.v_mask])
        gemm = np.array(mesh.gemm_edges)
        new_indices = np.zeros(mesh.v_mask.shape[0], dtype=np.int32)
        new_indices[mesh.v_mask] = np.arange(0, np.ma.where(mesh.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = mesh.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(mesh.__cycle_to_face(cycle, new_indices))
        mesh.vs = [vs[i] for i in new_indices]
        mesh.faces = faces

    def sample(self, total_samples):
        if self.faces is None:
            self.calculate_faces()
        # faces_sizes = [triangle_area(face) for face in self.faces]
        faces_sizes = [1. for face in self.faces]
        total_sizes = sum(faces_sizes)
        faces_sizes = [x / total_sizes for x in faces_sizes]
        samples = []
        for i, face in enumerate(self.faces):
            fvs = [self.vs[face[i]] for i in range(3)]

            # sample random weights as constants
            n_samples = int(faces_sizes[i] * total_samples) + 1
            rands = torch.rand(n_samples, 3).to(fvs[0].device)
            rands = rands / torch.sum(rands, dim=-1).unsqueeze(-1)
            face_samples = None
            for j in range(3):
                a = rands[:, j]
                x = fvs[j]
                T = a.unsqueeze(1).expand(n_samples, 3) * x.unsqueeze(0).expand(n_samples, 3)
                if face_samples is None:
                    face_samples = T
                else:
                    face_samples += T
            samples.extend(T)

        b = torch.stack(samples)
        return b[torch.randperm(b.shape[0])[:total_samples]]

    @staticmethod
    def chamfer_distance(meshes1, meshes2, resolution=3000):
        chamfer_dist = chamferdist.ChamferDistance()

        # differential sampling
        s1 = torch.stack([m.sample(resolution) for m in meshes1])
        s2 = torch.stack([m.sample(resolution) for m in meshes2])

        return chamfer_dist.forward(s1, s2)

    def subidivide_3(self, n):
        vs = list(self.vs[self.v_mask])
        self.calculate_faces()
        new_indices = list(range(len(vs)))
        faces = np.random.choice(np.arange(len(self.faces)), size=n, replace=False)
        faces = [self.faces[i] for i in faces]
        new_faces = []
        new_edges = []
        for face in faces:
            face_coords = face

            face_center = (vs[face_coords[0]] + vs[face_coords[1]] + vs[face_coords[2]]) / 3  # TODO MAYBE BUG
            face_center_index = len(vs)
            vs.append(face_center)
            new_indices.append(len(vs))
            new_faces.extend([
                (face_coords[0], face_coords[1], face_center_index),
                (face_coords[0], face_coords[2], face_center_index),
                (face_coords[1], face_coords[2], face_center_index)
            ])

            new_edges.extend([
                [face_coords[0], face_center_index],
                [face_coords[1], face_center_index],
                [face_coords[2], face_center_index]
            ])

        for f in faces:
            self.faces.remove(f)
        self.faces.extend(new_faces)
        self.edges = list(self.edges) + new_edges
        self.vs = vs

    def subdivide_2(self, n):
        vs = list(self.vs)
        new_indices = list(range(len(vs)))
        faces = np.random.choice(np.arange(len(self.faces)), size=n, replace=False)
        faces = [self.faces[i] for i in faces]
        new_faces = []
        new_edges = []
        to_remove_edges = []
        for face in faces:
            face_coords = face

            new_index_val = (vs[face_coords[1]] + vs[face_coords[2]]) / 2  # TODO MAYBE BUG
            new_index = len(vs)
            vs.append(new_index_val)
            new_indices.append(len(vs))
            new_faces.extend([
                (face_coords[0], face_coords[1], new_index),
                (face_coords[0], face_coords[2], new_index)
            ])

            new_edges.extend([
                [face_coords[0], new_index],
                [face_coords[1], new_index],
                [face_coords[2], new_index]
            ])

            to_remove_edges.append([face_coords[1], face_coords[2]])

        for f in faces:
            self.faces.remove(f)

        to_remove_edges_idx = []
        for i, e in enumerate(self.edges):
            for e2 in to_remove_edges:
                if set(e) == set(e2):
                    to_remove_edges_idx.append(i)
                    break
        to_remove_edges_idx = sorted(to_remove_edges_idx)
        for i, j in enumerate(to_remove_edges_idx):
            del self.edges[j-i]
        self.faces.extend(new_faces)
        self.edges = list(self.edges) + new_edges
        self.vs = vs

    def export_2(self, out_path):
        vs, edges, faces = self.vs, self.edges, self.faces
        new_indices = list(range(len(vs)))
        with open(out_path, 'w+') as f:
            for _, v in enumerate(vs):
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face_id in range(len(faces) - 1):
                f.write(
                    "f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))
            print(len(edges))

    def to(self, device):
        if isinstance(self.vs, list):
            self.vs = torch.stack(self.vs)
        self.vs = self.vs.to(device)
        return self