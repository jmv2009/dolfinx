from mpi4py import MPI

import numpy as np
from scipy.sparse.linalg import spsolve

import basix
import dolfinx.cpp as _cpp
import ffcx
import ufl
from dolfinx.cpp.mesh import Mesh_float64, create_geometry, create_topology
from dolfinx.fem import DofMap, coordinate_element
from dolfinx.io.utils import cell_perm_vtk
from dolfinx.la import matrix_csr
from dolfinx.mesh import CellType

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("Serial only")


nx = 16
ny = 16
nz = 16
n_cells = nx * ny * nz
cells: list = [[], []]
orig_idx: list = [[], []]
idx = 0
for i in range(n_cells):
    iz = i // (nx * ny)
    j = i % (nx * ny)
    iy = j // nx
    ix = j % nx

    v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix
    v1 = v0 + 1
    v2 = v0 + (nx + 1)
    v3 = v1 + (nx + 1)
    v4 = v0 + (nx + 1) * (ny + 1)
    v5 = v1 + (nx + 1) * (ny + 1)
    v6 = v2 + (nx + 1) * (ny + 1)
    v7 = v3 + (nx + 1) * (ny + 1)
    if ix < nx / 2:
        cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
        orig_idx[0] += [idx]
        idx += 1
    else:
        cells[1] += [v0, v1, v2, v4, v5, v6]
        orig_idx[1] += [idx]
        idx += 1
        cells[1] += [v1, v2, v3, v5, v6, v7]
        orig_idx[1] += [idx]
        idx += 1

n_points = (nx + 1) * (ny + 1) * (nz + 1)
sqxy = (nx + 1) * (ny + 1)
geom = []
for v in range(n_points):
    iz = v // sqxy
    p = v % sqxy
    iy = p // (nx + 1)
    ix = p % (nx + 1)
    geom += [[ix / nx, iy / ny, iz / nz]]
geomx = np.array(geom, dtype=np.float64)

ghost_owners: list = [[], []]
boundary_vertices: list = []

topology = create_topology(
    MPI.COMM_SELF,
    [CellType.hexahedron, CellType.prism],
    cells,
    orig_idx,
    ghost_owners,
    boundary_vertices,
)

hexahedron = coordinate_element(CellType.hexahedron, 1)
prism = coordinate_element(CellType.prism, 1)
nodes = np.arange(geomx.shape[0], dtype=np.int64)
xdofs = np.array(cells[0] + cells[1], dtype=np.int64)

geom = create_geometry(
    topology, [hexahedron._cpp_object, prism._cpp_object], nodes, xdofs, geomx.flatten(), 3
)

mesh = Mesh_float64(MPI.COMM_WORLD, topology, geom)

# Order 1 dofmaps
elements = [
    basix.create_element(basix.ElementFamily.P, basix.CellType.hexahedron, 1),
    basix.create_element(basix.ElementFamily.P, basix.CellType.prism, 1),
]

cpp_elements = [_cpp.fem.FiniteElement_float64(e._e, 1, True) for e in elements]

dofmaps = _cpp.fem.create_dofmaps(mesh.comm, mesh.topology, cpp_elements)
q = [DofMap(dofmaps[0]), DofMap(dofmaps[1])]

# Both dofmaps have the same IndexMap, but different cell_dofs
# Create SparsityPattern
sp = _cpp.la.SparsityPattern(MPI.COMM_WORLD, [q[0].index_map, q[0].index_map], [1, 1])
for ct in range(2):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    print(f"For cell type {ct}, create sparsity with {num_cells_type} cells.")
    for j in range(num_cells_type):
        cell_dofs_j = q[ct].cell_dofs(j)
        sp.insert(cell_dofs_j, cell_dofs_j)
sp.finalize()

a = []
for cell_name in ["hexahedron", "prism"]:
    print(f"Compiling form for {cell_name}")
    element = basix.ufl.element("Lagrange", cell_name, 1)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(3,)))
    space = ufl.FunctionSpace(domain, element)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    k = 12.0
    a += [(ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * u * v) * ufl.dx]
w, module, _ = ffcx.codegeneration.jit.compile_forms(a, options={"scalar_type": np.float64})
kernels = [getattr(w[i].form_integrals[0], "tabulate_tensor_float64") for i in range(2)]
ffi = module.ffi

# Assembler
A = matrix_csr(sp)
print(f"Assembling into matrix of size {len(A.data)} non-zeros")

# For each cell type
for ct in range(2):
    num_cells_type = mesh.topology.index_maps(3)[ct].size_local
    geom_dm = mesh.geometry.dofmaps(ct)
    kernel = kernels[ct]
    for j in range(num_cells_type):
        cell_dofs_j = q[ct].cell_dofs(j)
        A_local = np.zeros((len(cell_dofs_j) ** 2), dtype=np.float64)
        cell_geom = mesh.geometry.x[geom_dm[j]]
        kernel(
            ffi.cast("double *", A_local.ctypes.data),
            ffi.NULL,
            ffi.NULL,
            ffi.cast("double *", cell_geom.ctypes.data),
            ffi.NULL,
            ffi.NULL,
        )
        A.add(A_local, cell_dofs_j, cell_dofs_j, 1)


A_scipy = A.to_scipy()
b_scipy = np.ones(A_scipy.shape[1])

x = spsolve(A_scipy, b_scipy)
print(f"Solution vector norm {np.linalg.norm(x)}")

# I/O
# Save to XDMF
xdmf = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="mesh" GridType="Collection" CollectionType="spatial">

"""

vtk_topology = []
perm = [cell_perm_vtk(CellType.hexahedron, 8), cell_perm_vtk(CellType.prism, 6)]

geom_dm = mesh.geometry.dofmaps(0)
for c in geom_dm:
    vtk_topology += list(c[perm[0]])

xdmf += f"""
      <Grid Name="hex" GridType="Uniform">
        <Topology TopologyType="Hexahedron">
          <DataItem Dimensions="{geom_dm.shape[0]} 8" Precision="4" NumberType="Int" Format="XML">
          {" ".join(str(val) for val in vtk_topology)}
          </DataItem>
        </Topology>
        <Geometry GeometryType="XYZ" NumberType="float" Rank="2" Precision="8">
          <DataItem Dimensions="{mesh.geometry.x.shape[0]} 3" Format="XML">
            {" ".join(str(val) for val in mesh.geometry.x.flatten())}
          </DataItem>
        </Geometry>
        <Attribute Name="u" Center="Node" NumberType="float" Precision="8">
          <DataItem Dimensions="{len(x)}" Format="XML">
            {" ".join(str(val) for val in x)}
          </DataItem>
       </Attribute>
      </Grid>"""

vtk_topology = []
geom_dm = mesh.geometry.dofmaps(1)
for c in geom_dm:
    vtk_topology += list(c[perm[1]])

xdmf += f"""
      <Grid Name="prism" GridType="Uniform">
        <Topology TopologyType="Wedge">
          <DataItem Dimensions="{geom_dm.shape[0]} 6" Precision="4" NumberType="Int" Format="XML">
          {" ".join(str(val) for val in vtk_topology)}
          </DataItem>
        </Topology>
        <Geometry GeometryType="XYZ" NumberType="float" Rank="2" Precision="8">
          <DataItem Dimensions="{mesh.geometry.x.shape[0]} 3" Format="XML">
            {" ".join(str(val) for val in mesh.geometry.x.flatten())}
          </DataItem>
        </Geometry>
        <Attribute Name="u" Center="Node" NumberType="float" Precision="8">
          <DataItem Dimensions="{len(x)}" Format="XML">
            {" ".join(str(val) for val in x)}
          </DataItem>
        </Attribute>
      </Grid>"""

xdmf += """
    </Grid>
  </Domain>
</Xdmf>
"""

fd = open("mixed-mesh.xdmf", "w")
fd.write(xdmf)
fd.close()
