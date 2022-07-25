// Copyright (C) 2006-2020 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Topology.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <string>
#include <utility>

namespace dolfinx::fem
{
class CoordinateElement;
}

namespace dolfinx::graph
{
template <typename T>
class AdjacencyList;
}

namespace dolfinx::mesh
{

/// A Mesh consists of a set of connected and numbered mesh topological
/// entities, and geometry data
class Mesh
{
public:
  /// Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <typename Topology, typename Geometry>
  Mesh(MPI_Comm comm, Topology&& topology, Geometry&& geometry)
      : _topology(std::forward<Topology>(topology)),
        _geometry(std::forward<Geometry>(geometry)), _comm(comm)
  {
    // Do nothing
  }

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh) = default;

  /// Move constructor
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh) = default;

  /// Destructor
  ~Mesh() = default;

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  // TODO: Is there any use for this? In many situations one has to get the
  // topology of a const Mesh, which is done by Mesh::topology_mutable. Note
  // that the python interface (calls Mesh::topology()) may still rely on it.
  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology();

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const;

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  Topology& topology_mutable() const;

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry& geometry();

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry& geometry() const;

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm comm() const;

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology:
  // TODO: This is mutable because of the current memory management within
  // mesh::Topology. It allows to obtain a non-const Topology from a
  // const mesh (via Mesh::topology_mutable()).
  //
  mutable Topology _topology;

  // Mesh geometry
  Geometry _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};

/// @brief Create a mesh using the default partitioner.
///
/// This function takes mesh input data that is distributed across
/// processes and creates a mesh::Mesh, with the mesh cell distribution
/// determined by the default cell partitioner. The default partitioner
/// is based a graph partitioning.
///
/// @param[in] comm The MPI communicator to build the mesh on
/// @param[in] cells The cells on the this MPI rank. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices). For lowest order cells this will be just the cell
/// vertices. For higher-order cells, other cells 'nodes' will be
/// included.
/// @param[in] element The coordinate element that describes the
/// geometric mapping for cells
/// @param[in] x The coordinates of mesh nodes
/// @param[in] xshape The shape of `x`
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A distributed Mesh.
Mesh create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                 const fem::CoordinateElement& element,
                 std::span<const double> x, std::array<std::size_t, 2> xshape,
                 GhostMode ghost_mode);

/// Create a mesh using a provided mesh partitioning function
Mesh create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                 const fem::CoordinateElement& element,
                 std::span<const double> x, std::array<std::size_t, 2> xshape,
                 GhostMode ghost_mode,
                 const CellPartitionFunction& cell_partitioner);

/// Create a new mesh consisting of a subset of entities in a mesh.
/// @param[in] mesh The mesh
/// @param[in] dim Entity dimension
/// @param[in] entities List of entity indicies in `mesh` to include in
/// the new mesh
/// @return The new mesh, and maps from the new mesh entities, vertices,
/// and geometry to the input mesh entities, vertices, and geometry.
std::tuple<Mesh, std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<std::int32_t>>
create_submesh(const Mesh& mesh, int dim,
               const std::span<const std::int32_t>& entities);

/// Helper function to get the (cell, local facet index) pairs
/// for a given facet. There is one pair for exterior facets,
/// and two pairs for interior facets.
/// @param[in] f The facet index
/// @param[in] cells The cell(s) connected to the facet
/// @param[in] c_to_f The cell to facet connectivity for the mesh
/// @return List of (cell, local facet index) pairs corresponding
/// to the facet.
template <std::int32_t num_cells>
static std::array<std::array<std::int32_t, 2>, num_cells>
get_cell_local_facet_pairs(
    std::int32_t f, const std::span<const std::int32_t>& cells,
    const dolfinx::graph::AdjacencyList<std::int32_t>& c_to_f)
{
  // Loop over cells sharing facet
  assert(cells.size() == num_cells);
  std::array<std::array<std::int32_t, 2>, num_cells> cell_local_facet_pairs;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get local index of facet with respect to the cell
    std::int32_t cell = cells[c];
    auto cell_facets = c_to_f.links(cell);
    auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
    assert(facet_it != cell_facets.end());
    int local_f = std::distance(cell_facets.begin(), facet_it);
    cell_local_facet_pairs[c] = {cell, local_f};
  }

  return cell_local_facet_pairs;
}
} // namespace dolfinx::mesh
