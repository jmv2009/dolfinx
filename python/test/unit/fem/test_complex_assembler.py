# Copyright (C) 2018 Igor A. Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly in complex mode"""

import numpy as np
import pytest

import ufl
from basix.ufl import element
import dolfinx
from dolfinx import fem
from dolfinx.fem import Function, FunctionSpace, form
from dolfinx.la import InsertMode
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner

from mpi4py import MPI

if dolfinx.has_petsc:
    from petsc4py import PETSc
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_assembly(dtype):
    """Test assembly of complex matrices and vectors"""

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    P2 = element("Lagrange", mesh.basix_cell(), 2)
    V = FunctionSpace(mesh, P2)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    g = -2 + 3.0j
    j = 1.0j

    a_real = form(inner(u, v) * dx, dtype=dtype)
    L1 = form(inner(g, v) * dx, dtype=dtype)

    b = fem.assemble_vector(L1)
    b.scatter_reverse(InsertMode.add)
    bnorm = abs(b.array.sum())
    b_norm_ref = abs(-2 + 3.0j)
    print(bnorm, b_norm_ref)
    assert bnorm == pytest.approx(b_norm_ref)

    A = fem.assemble_matrix(a_real)
    A.finalize()
    A0_norm = A.squared_norm()

    x = ufl.SpatialCoordinate(mesh)

    a_imag = form(j * inner(u, v) * dx, dtype=dtype)
    f = 1j * ufl.sin(2 * np.pi * x[0])
    L0 = form(inner(f, v) * dx, dtype=dtype)
    A = fem.assemble_matrix(a_imag)
    A.finalize()
    A1_norm = A.squared_norm()
    assert A0_norm == pytest.approx(A1_norm)

    b = fem.assemble_vector(L0)
    b.scatter_reverse(InsertMode.add)
    b1_norm = b.norm()

    a_complex = form((1 + j) * inner(u, v) * dx, dtype=dtype)
    f = ufl.sin(2 * np.pi * x[0])
    L2 = form(inner(f, v) * dx, dtype=dtype)
    A = fem.assemble_matrix(a_complex)
    A.finalize()
    A2_norm = A.squared_norm()
    assert A1_norm == pytest.approx(A2_norm / 2.0)
    b = fem.assemble_vector(L2)
    b.scatter_reverse(InsertMode.add)
    b2_norm = b.norm()
    assert b2_norm == pytest.approx(b1_norm)


@pytest.mark.skipif(dolfinx.has_petsc is False, reason="Need PETSc for solver")
@pytest.mark.skipif(not np.issubdtype(PETSc.ScalarType, np.complexfloating),
                    reason="Only works in complex mode.")
def test_complex_assembly_solve():
    """Solve a positive definite helmholtz problem and verify solution
    with the method of manufactured solutions"""

    degree = 3
    mesh = create_unit_square(MPI.COMM_WORLD, 20, 20)
    P = element("Lagrange", mesh.basix_cell(), degree)
    V = FunctionSpace(mesh, P)

    x = ufl.SpatialCoordinate(mesh)

    # Define source term
    A = 1.0 + 2.0 * (2.0 * np.pi)**2
    f = (1. + 1j) * A * ufl.cos(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])

    # Variational problem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    C = 1.0 + 1.0j
    a = form(C * inner(grad(u), grad(v)) * dx + C * inner(u, v) * dx)
    L = form(inner(f, v) * dx)

    # Assemble
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOptionsPrefix("test_lu_")
    opts = PETSc.Options("test_lu_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    solver.setFromOptions()
    x = A.createVecRight()
    solver.setOperators(A)
    solver.solve(b, x)

    # Reference Solution
    def ref_eval(x):
        return np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    u_ref = Function(V)
    u_ref.interpolate(ref_eval)

    diff = (x - u_ref.vector).norm(PETSc.NormType.N2)
    assert diff == pytest.approx(0.0, abs=1e-1)
