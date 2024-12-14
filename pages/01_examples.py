import pyvista as pv  # noqa: INP001, D100
from stpyvista import stpyvista

## Initialize a plotter object
plotter = pv.Plotter(window_size=[400, 400])

## Create a mesh
mesh = pv.Sphere(radius=1.0, center=(0, 0, 0))

## Associate a scalar field to the mesh
x, y, z = mesh.cell_centers().points.T
mesh["My scalar"] = z

## Add mesh to the plotter
plotter.add_mesh(
    mesh,
    scalars="My scalar",
    cmap="prism",
    show_edges=True,
    edge_color="#001100",
    ambient=0.2,
)

## Some final touches
plotter.background_color = "white"
plotter.view_isometric()

## Pass a plotter to stpyvista
stpyvista(plotter)
