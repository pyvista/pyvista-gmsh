import pvgmsh
import pyvista
import numpy as np


def test_generate_mesh():
    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0], [0.3, 0, 0]])
    faces = np.hstack([[4, 0, 1, 2, 3]])
    surf = pyvista.PolyData(vertices, faces)
    # pvgmsh does not support PhysicalGroup; group configuration can be easily done with PyVista.
    mesh = pvgmsh.generate_mesh(surf)
    assert mesh.number_of_points > surf.number_of_points
    assert mesh.number_of_cells > surf.number_of_cells
