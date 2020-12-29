#pragma once

#include "nanoflann.hpp"
#include "types.hpp"

#include <memory>

namespace cho {

template <class Distance = nanoflann::metric_L2>
struct KDTreeCloud3fAdaptor {
    using MatrixType = cho::Cloud3f;
    using IndexType = int;

    using self_t = KDTreeCloud3fAdaptor<Distance>;
    using num_t = float;
    using metric_t = typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, 3, IndexType>;

    /// Constructor: takes a const ref to the matrix object with the data points
    explicit KDTreeCloud3fAdaptor(const std::reference_wrapper<const MatrixType>& mat,
        const int leaf_max_size = 16)
        : data{ mat }
        , index{ new index_t(3, *this /* adaptor */,
              nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size)) }
    {
        index->buildIndex();
    }

public:
    /** Deleted copy constructor */
    // KDTreeCloud3fAdaptor(const self_t&) = delete;
    ~KDTreeCloud3fAdaptor() = default;

    /** Query for the \a num_closest closest points to a given point (entered as
   * query_point[0:dim-1]). Note that this is a short-cut method for
   * index->findNeighbors(). The user can also call index->... methods as
   * desired. \note nChecks_IGNORED is ignored but kept for compatibility with
   * the original FLANN interface.
   */
    inline void query(const num_t* query_point, const size_t num_closest,
        IndexType* out_indices, num_t* out_distances_sq,
        const int /* nChecks_IGNORED */ = 10) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
   * @{ */

    const self_t& derived() const { return *this; }
    self_t& derived() { return *this; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return data.get().cols(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const IndexType idx, const size_t dim) const
    {
        return data.get().coeff(IndexType(dim), idx);
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in
    //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
    //   the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

public:
    std::reference_wrapper<const MatrixType> data;
    std::shared_ptr<index_t> index{ nullptr };
};

using KDTree = KDTreeCloud3fAdaptor<>;
}
