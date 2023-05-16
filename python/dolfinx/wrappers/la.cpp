// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>

namespace py = pybind11;

namespace
{
// InsertMode types
enum class PyInsertMode
{
  add,
  insert
};

// Declare objects that have multiple scalar types
template <typename T>
void declare_objects(py::module& m, const std::string& type)
{
  // dolfinx::la::Vector
  std::string pyclass_vector_name = std::string("Vector_") + type;
  py::class_<dolfinx::la::Vector<T>, std::shared_ptr<dolfinx::la::Vector<T>>>(
      m, pyclass_vector_name.c_str())
      .def(py::init([](std::shared_ptr<const dolfinx::common::IndexMap> map,
                       int bs) { return dolfinx::la::Vector<T>(map, bs); }),
           py::arg("map"), py::arg("bs"))
      .def(py::init([](const dolfinx::la::Vector<T>& vec)
                    { return dolfinx::la::Vector<T>(vec); }),
           py::arg("vec"))
      .def_property_readonly("dtype", [](const dolfinx::la::Vector<T>& self)
                             { return py::dtype::of<T>(); })
      .def("set", &dolfinx::la::Vector<T>::set, py::arg("v"))
      .def("squared_norm", [](const dolfinx::la::Vector<T>& self)
           { return dolfinx::la::squared_norm(self); })
      .def(
          "norm",
          [](dolfinx::la::Vector<T>& self, dolfinx::la::Norm type)
          { return dolfinx::la::norm(self, type); },
          py::arg("type") = dolfinx::la::Norm::l2)
      .def_property_readonly("index_map", &dolfinx::la::Vector<T>::index_map)
      .def_property_readonly("bs", &dolfinx::la::Vector<T>::bs)
      .def_property_readonly("array",
                             [](dolfinx::la::Vector<T>& self)
                             {
                               std::span<T> array = self.mutable_array();
                               return py::array_t<T>(array.size(), array.data(),
                                                     py::cast(self));
                             })
      .def("scatter_forward", &dolfinx::la::Vector<T>::scatter_fwd)
      .def(
          "scatter_reverse",
          [](dolfinx::la::Vector<T>& self, PyInsertMode mode)
          {
            switch (mode)
            {
            case PyInsertMode::add: // Add
              self.scatter_rev(std::plus<T>());
              break;
            case PyInsertMode::insert: // Insert
              self.scatter_rev([](T /*a*/, T b) { return b; });
              break;
            default:
              throw std::runtime_error("InsertMode not recognized.");
              break;
            }
          },
          py::arg("mode"));

  // dolfinx::la::MatrixCSR
  std::string pyclass_matrix_name = std::string("MatrixCSR_") + type;
  py::class_<dolfinx::la::MatrixCSR<T>,
             std::shared_ptr<dolfinx::la::MatrixCSR<T>>>(
      m, pyclass_matrix_name.c_str())
      .def(py::init([](const dolfinx::la::SparsityPattern& p)
                    { return dolfinx::la::MatrixCSR<T>(p); }),
           py::arg("p"))
      .def_property_readonly("dtype", [](const dolfinx::la::MatrixCSR<T>& self)
                             { return py::dtype::of<T>(); })
      .def("squared_norm", &dolfinx::la::MatrixCSR<T>::squared_norm)
      .def("mat_add_values", &dolfinx::la::MatrixCSR<T>::mat_add_values)
      .def("set",
           static_cast<void (dolfinx::la::MatrixCSR<T>::*)(T)>(
               &dolfinx::la::MatrixCSR<T>::set),
           py::arg("x"))
      .def("finalize", &dolfinx::la::MatrixCSR<T>::finalize)
      .def_property_readonly(
          "shape",
          [](const dolfinx::la::MatrixCSR<T>& self)
          {
            std::size_t nrows = self.num_all_rows();
            auto map_col = self.index_map(1);
            std::size_t ncols = map_col->size_local() + map_col->num_ghosts();
            return std::array<std::size_t, 2>({nrows, ncols});
          })
      .def("to_dense",
           [](const dolfinx::la::MatrixCSR<T>& self)
           {
             std::size_t nrows = self.num_all_rows();
             auto map_col = self.index_map(1);
             std::size_t ncols = map_col->size_local() + map_col->num_ghosts();
             return dolfinx_wrappers::as_pyarray(self.to_dense(),
                                                 std::array{nrows, ncols});
           })
      .def_property_readonly("data",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               std::span<T> array = self.values();
                               return py::array_t<T>(array.size(), array.data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("indices",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               auto& array = self.cols();
                               return py::array_t(array.size(), array.data(),
                                                  py::cast(self));
                             })
      .def_property_readonly("indptr",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               auto& array = self.row_ptr();
                               return py::array_t(array.size(), array.data(),
                                                  py::cast(self));
                             })
      .def("finalize_begin", &dolfinx::la::MatrixCSR<T>::finalize_begin)
      .def("finalize_end", &dolfinx::la::MatrixCSR<T>::finalize_end);
}

} // namespace

namespace dolfinx_wrappers
{
void la(py::module& m)
{
  py::enum_<PyInsertMode>(m, "InsertMode")
      .value("add", PyInsertMode::add)
      .value("insert", PyInsertMode::insert);

  py::enum_<dolfinx::la::Norm>(m, "Norm")
      .value("l1", dolfinx::la::Norm::l1)
      .value("l2", dolfinx::la::Norm::l2)
      .value("linf", dolfinx::la::Norm::linf)
      .value("frobenius", dolfinx::la::Norm::frobenius);

  // dolfinx::la::SparsityPattern
  py::class_<dolfinx::la::SparsityPattern,
             std::shared_ptr<dolfinx::la::SparsityPattern>>(m,
                                                            "SparsityPattern")
      .def(
          py::init(
              [](const MPICommWrapper comm,
                 const std::array<
                     std::shared_ptr<const dolfinx::common::IndexMap>, 2>& maps,
                 const std::array<int, 2>& bs)
              { return dolfinx::la::SparsityPattern(comm.get(), maps, bs); }),
          py::arg("comm"), py::arg("maps"), py::arg("bs"))
      .def(
          py::init(
              [](const MPICommWrapper comm,
                 const std::vector<
                     std::vector<const dolfinx::la::SparsityPattern*>>
                     patterns,
                 const std::array<
                     std::vector<std::pair<std::reference_wrapper<
                                               const dolfinx::common::IndexMap>,
                                           int>>,
                     2>& maps,
                 const  std::array<std::vector<int>, 2>& bs) {
                return dolfinx::la::SparsityPattern(comm.get(), patterns, maps,
                                                    bs);
              }),
          py::arg("comm"), py::arg("patterns"), py::arg("maps"), py::arg("bs"))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map,
           py::arg("dim"))
      .def("block_size", &dolfinx::la::SparsityPattern::block_size,
           py::arg("dim"))
      .def("column_index_map", &dolfinx::la::SparsityPattern::column_index_map)
      .def("nnz_diag", &dolfinx::la::SparsityPattern::nnz_diag)
      .def("nnz_off_diag", &dolfinx::la::SparsityPattern::nnz_off_diag)
      .def("finalize", &dolfinx::la::SparsityPattern::finalize)
      .def_property_readonly("num_nonzeros",
                             &dolfinx::la::SparsityPattern::num_nonzeros)
      .def(
          "insert",
          [](dolfinx::la::SparsityPattern& self,
             const py::array_t<std::int32_t, py::array::c_style>& rows,
             const py::array_t<std::int32_t, py::array::c_style>& cols)
          {
            self.insert(std::span(rows.data(), rows.size()),
                        std::span(cols.data(), cols.size()));
          },
          py::arg("rows"),  py::arg("cols"))
      .def(
          "insert_diagonal",
          [](dolfinx::la::SparsityPattern& self,
             const py::array_t<std::int32_t, py::array::c_style>& rows)
          { self.insert_diagonal(std::span(rows.data(), rows.size())); },
          py::arg("rows"))
      .def_property_readonly(
          "graph", [](dolfinx::la::SparsityPattern& self)
          {
            auto [edges, ptr] = self.graph();
            return std::pair(py::array_t(
                          edges.size(), edges.data(), py::cast(self)),
                             py::array_t(ptr.size(), ptr.data(),
                                                       py::cast(self)));
          });

  // Declare objects that are templated over type
  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");
}
} // namespace dolfinx_wrappers
