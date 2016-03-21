"""
# Strength of connection metrics for AMG

This module implements various strength of connection metrics. The strength of
connection matrix measures how closely related two vertices are. Large
values indicate that vertices are close together. Low values indicate they
are far appart. Zero values indicate that vertices are not connected.

This modules borrows heavily from pyamg. See [https://github.com/pyamg/pyamg/blob/master/pyamg/strength.py](https://github.com/pyamg/pyamg/blob/master/pyamg/strength.py).
"""
module Strength

using SparseRelaxation

"""
    algebraic_distance{T}( A :: AbstractSparseMatrix{T} # Matrix to compute distance on
                         , k :: Int # number of smoothing iterations
                         , R :: Int # number of test vectors
                         , p # norm to measure in, use `Inf` for inf norm
                         , theta :: T = 0.1 # drop tolerance for small distances
                         , smoother :: Function = weighted_jacobi! # smoother to use
                         )

Computes a strength of connection matrix using algebraic distance. This metric
takes `R` test vectors and smooths them `k` times using `smoother`. The
resulting strength of connection between `i` and `j` is the `norm` of the
difference between row `i` and `j` of the test vectors. The sparsity pattern of
`A` is mantained. `smoother` must be inplace. `theta` determines the tolerance
for dropping small distances. `theta` must be in (0, 1).
"""
function algebraic_distance{T}( A :: AbstractSparseMatrix{T}
                              , k :: Int
                              , R :: Int
                              , p :: Float64
                              , theta :: T = 0.1
                              , smoother :: Function = weighted_jacobi!
                              )
    m,n = size(A)
    tvs = rand(n, R) - 0.5 # generate test vectors
    b = zeros(n) # zero rhs

    # smooth test vectors
    for _ in 1:k
        for i in R
            smoother(A, tvs[:,i], b)
        end
    end

    xs,ys = findn(A)
    dists = if p == Inf
                maximum(tvs[xs] - tvs[ys], 2)
            else
                (sum(abs(tvs[xs] - tvs[ys]).^p, 2)./R).^(1/p)
            end

    M = sparse(xs, ys, dists) # removes zero entries were i == j

    M.nzval = 1 ./ M.nzval # invert values; large tv differences indicate large distance

    filter_distances!(M, theta) # remove small values
    M = M + speye(M)            # add identity back
    normalize_cols!(M)          # scale columns

    M
end

"""
    filter_distances!{T}(A :: AbstractSparseMatrix{T}, theta :: T)

Removes entry `e` from strength of connection if `e < theta * max(colvalues)`.
"""
function filter_distances!{T}(A :: AbstractSparseMatrix{T}, theta :: T)
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    @inbounds for col in 1:n
        maxval :: T = maximum(vals[nzrange(A, col)])

        for j in nzrange(A, col)
            row = rows[j]
            val = vals[j]
            if val < theta * maxval
                vals[j] = 0
            end
        end
    end

    SparseMatrix.dropzeros!(A)
end

"""
    normalize_cols!{T}(A :: AbstractSparseMatrix{T})

Scale each column in `A` by its largest value.
"""
function normalize_cols!{T}(A :: AbstractSparseMatrix{T})
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    @inbounds for col in 1:n
        r = nzrange(A, col)
        maxval :: T = maximum(vals[r])
        vals[r] ./ maxval
    end
end

end
