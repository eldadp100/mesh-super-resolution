import os


def read_mesh(fn):
    # return V, E, F
    V, E, F = [None], set(), set()
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                v = [float(x) for x in line.split()[1:]]
                V.append(v)
            if line[:2] == 'e ':
                e = tuple(sorted([int(x) for x in line.split()[1:]]))
                E.add(e)
            if line[:2] == 'f ':
                f = tuple(sorted([int(x) for x in line.split()[1:]]))
                F.add(f)
    return V, E, F


def subdivide(V, E, F, to_E=1600):
    _F_lst = list(F)
    for f in _F_lst:
        F.remove(f)
        v1, v2, v3 = V[f[0]], V[f[1]], V[f[2]]
        c = [(v1[i] + v2[i] + v3[i]) / 3 for i in range(3)]
        V.append(c)
        F.add(tuple(sorted([len(V) - 1, f[0], f[1]])))
        F.add(tuple(sorted([len(V) - 1, f[0], f[2]])))
        F.add(tuple(sorted([len(V) - 1, f[1], f[2]])))
        E.add(tuple(sorted([len(V) - 1, f[0]])))
        E.add(tuple(sorted([len(V) - 1, f[1]])))
        E.add(tuple(sorted([len(V) - 1, f[2]])))

    while len(E) + 3 <= to_E:
        f = F.pop()
        v1, v2, v3 = V[f[0]], V[f[1]], V[f[2]]
        c = [(v1[i] + v2[i] + v3[i]) / 3 for i in range(3)]
        V.append(c)
        F.add(tuple(sorted([len(V) - 1, f[0], f[1]])))
        F.add(tuple(sorted([len(V) - 1, f[0], f[2]])))
        F.add(tuple(sorted([len(V) - 1, f[1], f[2]])))
        E.add(tuple(sorted([len(V) - 1, f[0]])))
        E.add(tuple(sorted([len(V) - 1, f[1]])))
        E.add(tuple(sorted([len(V) - 1, f[2]])))

    if (to_E - len(E)) == 1:
        V.append((-0.010101, -0.010102, -0.010103))
        V.append((-0.010104, -0.010105, -0.010106))
        E.add((len(V) - 2, len(V) - 1))

    if (to_E - len(E)) == 2:
        V.append((-0.010101, -0.010102, -0.010103))
        V.append((-0.010104, -0.010105, -0.010106))
        V.append((-0.010107, -0.010108, -0.010109))
        E.add((len(V) - 2, len(V) - 1))
        E.add((len(V) - 3, len(V) - 1))
    print(len(E))
    return V, E, F


def export_mesh(V, E, F, fn):
    if os.path.exists(fn):
        os.remove(fn)
    with open(fn, 'w+') as FILE:
        for v in V[1:]:
            FILE.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for e in E:
            FILE.write(f"e {e[0]} {e[1]}\n")
        for f in F:
            FILE.write(f"f {f[0]} {f[1]} {f[2]}\n")


def do_subdivide(in_path, out_path, to_E):
    V, E, F = read_mesh(in_path)
    V, E, F = subdivide(V, E, F, to_E)
    export_mesh(V, E, F, out_path)
