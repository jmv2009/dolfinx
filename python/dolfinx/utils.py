# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Utility functions for calling PETSc C functions from Numba functions."""


from __future__ import annotations

import ctypes as _ctypes
import os
import pathlib
import typing

import numpy as _np
import numpy.typing as _npt

__all__ = ["cffi_utils", "numba_utils", "ctypes_utils"]


def get_petsc_lib() -> pathlib.Path:
    """Find the full path of the PETSc shared library.

    Returns:
        Full path to the PETSc shared library.

    Raises:
        RuntimeError: If PETSc library cannot be found for if more than
            one library is found.
    """
    import petsc4py as _petsc4py

    petsc_dir = _petsc4py.get_config()['PETSC_DIR']
    petsc_arch = _petsc4py.lib.getPathArchPETSc()[1]
    candidate_paths = [os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
                       os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib")]
    exists_paths = []
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            exists_paths.append(candidate_path)

    if len(exists_paths) == 0:
        raise RuntimeError("Could not find a PETSc shared library.")
    elif len(exists_paths) > 1:
        raise RuntimeError("More than one PETSc shared library found.")

    return pathlib.Path(exists_paths[0])


def load_petsc_lib(loader: typing.Callable[[str], typing.Any]) -> typing.Any:
    """Load PETSc shared library using loader callable, e.g. ctypes.CDLL.

    Args:
        loader: A callable that accepts a library path and returns a wrapped library.

    Returns:
        A wrapped library of the type returned by the callable.
    """
    lib_path = get_petsc_lib()
    try:
        try:
            petsc_lib = loader(lib_path)
        except TypeError:
            petsc_lib = loader(str(lib_path))
    except OSError as e:
        print(f"Failed to load shared library found at {lib_path}.")
        raise e

    return petsc_lib


class numba_utils:
    """Utility attributes for working with Numba and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions from within Numba functions.

    Note:
        `Numba <https://numba.pydata.org/>`_ must be available
        to use these utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes, data.ctypes, mode)
    """
    try:
        import petsc4py.PETSc as _PETSc

        import llvmlite as _llvmlite
        import numba as _numba
        _llvmlite.binding.load_library_permanently(str(get_petsc_lib()))

        _int = _numba.from_dtype(_PETSc.IntType)
        _scalar = _numba.from_dtype(_PETSc.ScalarType)
        _real = _numba.from_dtype(_PETSc.RealType)
        _int_ptr = _numba.core.types.CPointer(_int)
        _scalar_ptr = _numba.core.types.CPointer(_scalar)
        _MatSetValues_sig = _numba.core.typing.signature(_numba.core.types.intc,
                                                         _numba.core.types.uintp, _int, _int_ptr,
                                                         _int, _int_ptr, _scalar_ptr, _numba.core.types.intc)
        MatSetValuesLocal = _numba.core.types.ExternalFunction("MatSetValuesLocal", _MatSetValues_sig)
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _numba.core.types.ExternalFunction("MatSetValuesBlockedLocal", _MatSetValues_sig)
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except ImportError:
        pass


class ctypes_utils:
    """Utility attributes for working with ctypes and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions, typically from within Numba functions.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes, data.ctypes, mode)
    """
    import petsc4py.PETSc as _PETSc
    _lib_ctypes = load_petsc_lib(_ctypes.cdll.LoadLibrary)

    # Note: ctypes does not have complex types, hence we use void* for
    # scalar data
    _int = _np.ctypeslib.as_ctypes_type(_PETSc.IntType)

    MatSetValuesLocal = _lib_ctypes.MatSetValuesLocal
    """See PETSc `MatSetValuesLocal
    <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
    documentation."""
    MatSetValuesLocal.argtypes = [_ctypes.c_void_p, _int, _ctypes.POINTER(_int), _int,
                                  _ctypes.POINTER(_int), _ctypes.c_void_p, _ctypes.c_int]

    MatSetValuesBlockedLocal = _lib_ctypes.MatSetValuesBlockedLocal
    """See PETSc `MatSetValuesBlockedLocal
    <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
    documentation."""
    MatSetValuesBlockedLocal.argtypes = [_ctypes.c_void_p, _int, _ctypes.POINTER(_int), _int,
                                         _ctypes.POINTER(_int), _ctypes.c_void_p, _ctypes.c_int]


class cffi_utils:
    """Utility attributes for working with CFFI (ABI mode) and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions, typically from within Numba functions.

    Note:
        `CFFI <https://cffi.readthedocs.io/>`_ and  `Numba
        <https://numba.pydata.org/>`_ must be available to use these
        utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, ffi.from_buffer(rows), n, ffi.from_buffer(cols),
                                ffi.from_buffer(rows(data), mode)
    """
    try:
        from petsc4py import PETSc as _PETSc

        import cffi as _cffi
        import numba as _numba
        import numba.core.typing.cffi_utils as _cffi_support

        # Register complex types
        _ffi = _cffi.FFI()
        _cffi_support.register_type(_ffi.typeof('float _Complex'), _numba.types.complex64)
        _cffi_support.register_type(_ffi.typeof('double _Complex'), _numba.types.complex128)

        _lib_cffi = load_petsc_lib(_ffi.dlopen)

        def _petsc_c_types() -> str:
            from petsc4py import PETSc as _PETSc
            assert _PETSc.IntType == _np.int32 or _PETSc.IntType == _np.int64
            if _PETSc.IntType == _np.int32:
                c_int_t = "int32_t"
            elif _PETSc.IntType == _np.int64:
                c_int_t = "int64_t"

            scalar_t = _PETSc.ScalarType
            assert scalar_t == _np.float32 or scalar_t == _np.float64 \
                or scalar_t == _np.complex64 or scalar_t == _np.complex128
            if scalar_t == _np.float32:
                c_scalar_t = "float"
            elif scalar_t == _np.float64:
                c_scalar_t = "double"
            elif scalar_t == _np.complex64:
                c_scalar_t = "float _Complex"
            elif scalar_t == _np.complex128:
                c_scalar_t = "double _Complex"
            return c_int_t, c_scalar_t

        _c_int_t, _c_scalar_t = _petsc_c_types()
        _ffi.cdef(f"""
                int MatSetValuesLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                int MatSetValuesBlockedLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                                    """)

        MatSetValuesLocal = _lib_cffi.MatSetValuesLocal
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _lib_cffi.MatSetValuesBlockedLocal
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except ImportError:
        pass


def numba_ufx_kernel_signature(dtype: _npt.DTypeLike, xdtype: _npt.DTypeLike):
    """Return a Numba C signature for the UFCx ``tabulate_tensor`` interface.

    Args:
        dtype: The scalar type for the finite element data.
        xdtype: The geometry float type.

    Returns:
        Numba

    Raises:
        ImportError: If ``numba`` cannot be imoprted.
    """
    try:
        import numba
        import numba.types as types
        return types.void(types.CPointer(numba.from_dtype(dtype)),
                          types.CPointer(numba.from_dtype(dtype)),
                          types.CPointer(numba.from_dtype(dtype)),
                          types.CPointer(numba.from_dtype(xdtype)),
                          types.CPointer(types.int32),
                          types.CPointer(types.int32))
    except ImportError as e:
        raise e
