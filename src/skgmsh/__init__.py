"""scikit-gmsh package for 3D mesh generation."""

from __future__ import annotations

import datetime  # noqa: F401
import io
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

import gmsh
from pygmsh.helpers import extract_to_meshio
import pyvista as pv
import scooby
import shapely

if TYPE_CHECKING:
    from collections.abc import Sequence

INITIAL_MESH_ONLY_2D = 3
FRONTAL_DELAUNAY_2D = 6
DELAUNAY_3D = 1
INITIAL_MESH_ONLY_3D = 3

SILENT = 0
SIMPLE = 0

TRUE = 1
FALSE = 0

# major, minor, patch
version_info = 0, 3, "dev0"

# Nice string for the version
__version__ = ".".join(map(str, version_info))


class Report(scooby.Report):  # type: ignore[misc]
    """
    Generate an environment package and hardware report.

    Parameters
    ----------
    ncol : int, default: 3
        Number of package-columns in html table; only has effect if
        ``mode='HTML'`` or ``mode='html'``.

    text_width : int, default: 80
        The text width for non-HTML display modes.

    """

    def __init__(self: Report, ncol: int = 3, text_width: int = 80) -> None:  # numpydoc ignore=PR01
        """Generate a :class:`scooby.Report` instance."""
        # mandatory packages
        core: list[str] = [
            "matplotlib",
            "numpy",
            "pooch",
            "pyvista",
            "scooby",
            "vtk",
            "gmsh",
            "meshio",
            "pygmsh",
            "pyvista",
        ]

        # optional packages
        optional: list[str] = [
            "imageio",
            "pyvistaqt",
            "PyQt5",
            "IPython",
            "colorcet",
            "cmocean",
            "ipywidgets",
            "scipy",
            "tqdm",
            "jupyterlab",
            "pytest_pyvista",
            "trame",
            "trame_client",
            "trame_server",
            "trame_vtk",
            "trame_vuetify",
            "jupyter_server_proxy",
            "nest_asyncio",
        ]

        extra_meta = [
            ("GPU Details", "None"),
        ]

        super().__init__(
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            extra_meta=extra_meta,
        )


def delaunay_3d(
    edge_source: pv.PolyData,
    target_sizes: float | Sequence[float] | None = None,
) -> pv.UnstructuredGrid | None:
    """
    Delaunay 3D mesh algorithm.

    Parameters
    ----------
    edge_source : pyvista.PolyData
        Specify the source object used to specify constrained
        edges and loops. If set, and lines/polygons are defined, a
        constrained triangulation is created. The lines/polygons
        are assumed to reference points in the input point set
        (i.e. point ids are identical in the input and
        source).

    target_sizes : float | Sequence[float]
        Target mesh size close to the points.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh from the 3D delaunay generation.

    Notes
    -----
    .. versionadded:: 0.2.0

    """
    points = edge_source.points
    faces = edge_source.irregular_faces

    gmsh.initialize()
    if target_sizes is None:
        gmsh.option.set_number("Mesh.Algorithm", INITIAL_MESH_ONLY_3D)
        gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    else:
        gmsh.option.set_number("Mesh.Algorithm3D", DELAUNAY_3D)
    gmsh.option.set_number("General.Verbosity", SILENT)
    gmsh.option.set_number("Mesh.AlgorithmSwitchOnFailure", FALSE)
    gmsh.option.set_number("Mesh.RecombinationAlgorithm", SIMPLE)
    gmsh.option.set_number("Mesh.RecombineNodeRepositioning", FALSE)

    if target_sizes is None:
        target_sizes = 0.0

    if isinstance(target_sizes, float):
        target_sizes = [target_sizes] * edge_source.number_of_points

    for i, (point, target_size) in enumerate(zip(points, target_sizes)):
        id_ = i + 1
        gmsh.model.geo.add_point(point[0], point[1], point[2], target_size, id_)

    surface_loop = []
    for i, face in enumerate(faces):
        curve_tags = []
        for j, _ in enumerate(face):
            start_tag = face[j - 1] + 1
            end_tag = face[j] + 1
            curve_tag = gmsh.model.geo.add_line(start_tag, end_tag)
            curve_tags.append(curve_tag)
        gmsh.model.geo.add_curve_loop(curve_tags, i + 1)
        gmsh.model.geo.add_plane_surface([i + 1], i + 1)
        surface_loop.append(i + 1)

    gmsh.model.geo.remove_all_duplicates()
    gmsh.model.geo.synchronize()

    gmsh.model.geo.add_surface_loop(surface_loop, 1)
    gmsh.model.geo.add_volume([1], 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    mesh = pv.wrap(extract_to_meshio())
    gmsh.clear()
    gmsh.finalize()

    ind = []
    for i, cell in enumerate(mesh.cell):
        if cell.type != pv.CellType.TETRA:
            ind.append(i)
    mesh = mesh.remove_cells(ind)
    mesh.clear_data()

    return mesh


def frontal_delaunay_2d(  # noqa: C901, PLR0912
    edge_source: pv.PolyData | shapely.geometry.Polygon,
    target_sizes: float | Sequence[float] | None = None,
    recombine: bool = False,  # noqa: FBT001, FBT002
) -> pv.UnstructuredGrid | None:
    """
    Frontal-Delaunay 2D mesh algorithm.

    Parameters
    ----------
    edge_source : pyvista.PolyData | shapely.geometry.Polygon
        Specify the source object used to specify constrained
        edges and loops. If set, and lines/polygons are defined, a
        constrained triangulation is created. The lines/polygons
        are assumed to reference points in the input point set
        (i.e. point ids are identical in the input and
        source).

    target_sizes : float
        Target mesh size close to the points.
        Default max size of edge_source in each direction.

    recombine : bool
        Recombine the generated mesh into quadrangles.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh from the 2D delaunay generation.

    Notes
    -----
    .. versionadded:: 0.2.0

    """
    gmsh.initialize()
    if target_sizes is None:
        gmsh.option.set_number("Mesh.Algorithm", INITIAL_MESH_ONLY_2D)
        gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    else:
        gmsh.option.set_number("Mesh.Algorithm", FRONTAL_DELAUNAY_2D)
    gmsh.option.set_number("General.Verbosity", SILENT)

    if target_sizes is None:
        target_sizes = 0.0

    if isinstance(edge_source, shapely.geometry.Polygon):
        wire_tags = []
        for linearring in [edge_source.exterior, *list(edge_source.interiors)]:
            coords = linearring.coords[:-1].copy()
            tags = []
            for coord in coords:
                x = coord[0]
                y = coord[1]
                z = coord[2]
                tags.append(gmsh.model.geo.add_point(x, y, z, target_sizes))
            curve_tags = []
            for i, _ in enumerate(tags):
                start_tag = tags[i - 1]
                end_tag = tags[i]
                curve_tags.append(gmsh.model.geo.add_line(start_tag, end_tag))
            wire_tags.append(gmsh.model.geo.add_curve_loop(curve_tags))
        gmsh.model.geo.add_plane_surface(wire_tags)
        gmsh.model.geo.synchronize()
    else:
        points = edge_source.points
        lines = edge_source.lines

        if isinstance(target_sizes, float):
            target_sizes = [target_sizes] * edge_source.number_of_points

        embedded_points = []
        for target_size, point in zip(target_sizes, points):
            embedded_points.append(gmsh.model.geo.add_point(point[0], point[1], point[2], target_size))

        for i in range(lines[0] - 1):
            id_ = i + 1
            gmsh.model.geo.add_line(lines[i + 1] + 1, lines[i + 2] + 1, id_)

        gmsh.model.geo.add_curve_loop(range(1, lines[0]), 1)
        gmsh.model.geo.add_plane_surface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, embedded_points, 2, 1)

    if recombine:
        gmsh.model.mesh.set_recombine(2, 1)

    gmsh.model.mesh.generate(2)
    mesh = pv.from_meshio(extract_to_meshio())
    gmsh.clear()
    gmsh.finalize()

    ind = []
    for index, cell in enumerate(mesh.cell):
        if cell.type in [pv.CellType.VERTEX, pv.CellType.LINE]:
            ind.append(index)

    return mesh.remove_cells(ind)


class Delaunay2D:
    """
    Delaunay 2D mesh algorithm.

    Parameters
    ----------
    edge_source : pyvista.PolyData | shapely.Polygon
        Specify the source object used to specify constrained
        edges and loops. If set, and lines/polygons are defined, a
        constrained triangulation is created. The lines/polygons
        are assumed to reference points in the input point set
        (i.e. point ids are identical in the input and
        source).

    shell : sequence
        A sequence of (x, y [,z]) numeric coordinate pairs or triples, or
        an array-like with shape (N, 2) or (N, 3).
        Also can be a sequence of Point objects.

    holes : sequence
        A sequence of objects which satisfy the same requirements as the
        shell parameters above

    cell_size : float
       Meshing constraint at point.

    Notes
    -----
    .. versionadded:: 0.2.0

    """

    def __init__(
        self: Delaunay2D,
        *,
        edge_source: pv.PolyData | shapely.Polygon | None = None,
        shell: Sequence[tuple[int]] | None = None,
        holes: Sequence[tuple[int]] | None = None,
        cell_size: float | None = None,
    ) -> None:
        """Initialize the Delaunay2D class."""
        if edge_source is not None:
            self._edge_source = edge_source
        else:
            self._edge_source = shapely.Polygon(shell, holes)
        self._cell_size = cell_size

    @property
    def edge_source(self: Delaunay2D) -> pv.PolyData | shapely.geometry.Polygon:
        """Get the edge source."""
        return self._edge_source

    @property
    def mesh(self: Delaunay2D) -> pv.PolyData:
        """Get the mesh."""
        mesh = frontal_delaunay_2d(self._edge_source, target_sizes=self._cell_size)
        return pv.PolyData(mesh.points, mesh.cells)

    @property
    def cell_size(self: Delaunay2D) -> float | None:
        """Get the cell_size of the mesh."""
        return self._cell_size

    @cell_size.setter
    def cell_size(self: Delaunay2D, size: int) -> None:
        """Set the cell_size of the mesh."""
        self._cell_size = size


class Delaunay3D:
    """
    Delaunay 3D mesh algorithm.

    Parameters
    ----------
    edge_source : pyvista.PolyData
        Specify the source object used to specify constrained
        edges and loops. If set, and lines/polygons are defined, a
        constrained triangulation is created. The lines/polygons
        are assumed to reference points in the input point set
        (i.e. point ids are identical in the input and
        source).

    Notes
    -----
    .. versionadded:: 0.2.0

    """

    def __init__(
        self: Delaunay3D,
        edge_source: pv.PolyData,
        cell_size: float | None = None,
    ) -> None:
        """Initialize the Delaunay3D class."""
        self._edge_source = edge_source
        self._cell_size = cell_size

    @property
    def edge_source(self: Delaunay3D) -> pv.PolyData:
        """Get the edge source."""
        return self._edge_source

    @property
    def mesh(self: Delaunay3D) -> pv.UnstructuredGrid:
        """Get the mesh."""
        self._mesh = delaunay_3d(self.edge_source, target_sizes=self.cell_size)
        return self._mesh

    @property
    def cell_size(self: Delaunay3D) -> float | None:
        """Get the cell_size of the mesh."""
        return self._cell_size

    @cell_size.setter
    def cell_size(self: Delaunay3D, size: int) -> None:
        """Set the cell_size of the mesh."""
        self._cell_size = size


class Geometry2D:  # noqa: D101
    def __init__(self):  # noqa: ANN204, D107
        # List to store geometry elements
        self.elements = []
        self.virtual_file = io.StringIO()  # Using StringIO as a virtual file

    def add_point(self, x, y, z=0.0, lc=1.0):  # noqa: ANN001, ANN201
        """Add a point to the geometry"""  # noqa: D400, D415
        self.elements.append(f"Point({len(self.elements) + 1}) = {{{x}, {y}, {z}, {lc}}};")

    def add_line(self, point1_id, point2_id):  # noqa: ANN001, ANN201
        """Add a line connecting two points"""  # noqa: D400, D415
        self.elements.append(f"Line({len(self.elements) + 1}) = {{{point1_id}, {point2_id}}};")

    def add_circle(self, point1_id, point2_id, point3_id):  # noqa: ANN001, ANN201
        """Add a circular arc through three points"""  # noqa: D400, D415
        self.elements.append(f"Circle({len(self.elements) + 1}) = {{{point1_id}, {point2_id}, {point3_id}}};")

    def add_surface(self, line_ids):  # noqa: ANN001, ANN201
        """Add a surface using a set of lines"""  # noqa: D400, D415
        line_str = ", ".join(map(str, line_ids))
        self.elements.append(f"Line Loop({len(self.elements) + 1}) = {{{line_str}}};")
        self.elements.append(f"Plane Surface({len(self.elements) + 1}) = {{{len(self.elements)}}};")

    def add_physical_group(self, dimension, entity_ids, name=None):  # noqa: ANN001, ANN201
        """Add a physical group for meshing purposes"""  # noqa: D400, D415
        entity_str = ", ".join(map(str, entity_ids))
        if name:
            self.elements.append(f"Physical Group({dimension}) = {{{entity_str}}};")
        else:
            self.elements.append(f"Physical Group({dimension}) = {{{entity_str}}};")

    def to_virtual_file(self):  # noqa: ANN201
        """Save geometry to a virtual file"""  # noqa: D400, D415
        self.virtual_file = io.StringIO()  # Create a new virtual file
        self.virtual_file.write("// Gmsh Geometry File\n")
        for element in self.elements:
            self.virtual_file.write(f"{element}\n")
        self.virtual_file.seek(0)  # Reset the file pointer to the beginning

    def from_virtual_file(self, virtual_file):  # noqa: ANN001, ANN201
        """Load geometry from a virtual file"""  # noqa: D400, D415
        self.elements.clear()
        virtual_file.seek(0)  # Move pointer to the start of the file
        for line in virtual_file:
            line = line.strip()  # noqa: PLW2901
            if line and not line.startswith("//"):
                self.elements.append(line)

    def get_virtual_file_content(self):  # noqa: ANN201
        """Get the content of the virtual file"""  # noqa: D400, D415
        self.virtual_file.seek(0)  # Move the pointer to the start
        return self.virtual_file.read()

    def execute_gmsh(self, gmsh_path="gmsh", output_file="mesh.msh", dimension=3):  # noqa: ANN001, ANN201
        """
        Execute Gmsh to generate a mesh from the virtual geometry file.
        gmsh_path: Path to the Gmsh executable
        output_file: Name of the output mesh file
        dimension: The dimension of the mesh to generate (2D or 3D)
        """  # noqa: D205, D400, D415
        # Save the virtual file content to a temporary .geo file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".geo") as temp_geo:
            temp_geo.write(self.get_virtual_file_content().encode("utf-8"))
            temp_geo.flush()
            temp_geo_name = temp_geo.name  # Get the temporary file name

        # Construct the Gmsh command
        command = [gmsh_path, temp_geo_name, "-o", output_file, f"-{dimension}", "-format", "msh"]

        # Execute the Gmsh command
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
            print("Gmsh execution successful.")  # noqa: T201
            print(result.stdout)  # noqa: T201
        except subprocess.CalledProcessError as e:
            print(f"Error executing Gmsh: {e.stderr}")  # noqa: T201
        finally:
            # Clean up the temporary file
            os.remove(temp_geo_name)  # noqa: PTH107

    def __str__(self) -> str:
        """Return the geometry as a string in Gmsh format."""
        return "\n".join(self.elements)


# Example usage
# Create geometry, save to virtual file, and execute Gmsh
manager = Geometry2D()
manager.add_point(0, 0, 0, 1.0)  # Point 1: Origin
manager.add_point(1, 0, 0, 1.0)  # Point 2: X-axis
manager.add_point(1, 1, 0, 1.0)  # Point 3: Perpendicular point
manager.add_line(1, 2)  # Line 1: Connect points 1 and 2
manager.add_line(2, 3)  # Line 2: Connect points 2 and 3
manager.add_line(3, 1)  # Line 3: Connect points 3 and 1
manager.add_surface([1, 2, 3])  # Surface: Enclose lines 1, 2, and 3

# Save geometry to virtual file
manager.to_virtual_file()

# Execute Gmsh to generate the mesh from the virtual file
manager.execute_gmsh(gmsh_path="/usr/local/bin/gmsh", output_file="output.msh", dimension=3)
