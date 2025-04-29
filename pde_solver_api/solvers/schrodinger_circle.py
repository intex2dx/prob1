import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import meshzoo


def local_stiffness_matrix(p):
    area = 0.5 * np.linalg.det(np.array([
        [1, p[0, 0], p[0, 1]],
        [1, p[1, 0], p[1, 1]],
        [1, p[2, 0], p[2, 1]]
    ]))
    B = np.array([
        [p[1,1] - p[2,1], p[2,1] - p[0,1], p[0,1] - p[1,1]],
        [p[2,0] - p[1,0], p[0,0] - p[2,0], p[1,0] - p[0,0]]
    ]) / (2 * area)
    return area * B.T @ B


def local_mass_matrix(p):
    area = 0.5 * np.linalg.det(np.array([
        [1, p[0, 0], p[0, 1]],
        [1, p[1, 0], p[1, 1]],
        [1, p[2, 0], p[2, 1]]
    ]))
    return (area / 12.0) * (np.ones((3, 3)) + 3 * np.eye(3))


def solve_schrodinger_circle(radius=1.0, num_points=50, num_eigenvalues=5):
    points, cells = meshzoo.disk(n_radial=num_points, n_angular=6*num_points)
    n_points = points.shape[0]
    K = scipy.sparse.lil_matrix((n_points, n_points))
    M = scipy.sparse.lil_matrix((n_points, n_points))

    for tri in cells:
        p = points[tri]
        k_local = local_stiffness_matrix(p)
        m_local = local_mass_matrix(p)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += k_local[i, j]
                M[tri[i], tri[j]] += m_local[i, j]

    boundary_nodes = np.where(np.isclose(np.linalg.norm(points, axis=1), radius, atol=0.01))[0]
    for node in boundary_nodes:
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        M[node, :] = 0
        M[:, node] = 0
        M[node, node] = 1e-10

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(K, k=num_eigenvalues, M=M, sigma=0.0, which='LM')
    idx = np.argsort(eigvals)
    return eigvals[idx], eigvecs[:, idx], points, cells
