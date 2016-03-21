"""

This module borrows heavily from pyamg. See [https://github.com/pyamg/pyamg/blob/master/pyamg/amg_core/smoothed_aggregation.h](https://github.com/pyamg/pyamg/blob/master/pyamg/amg_core/smoothed_aggregation.h).
"""
module Aggregation

"""
    standard_aggregation(A :: SparseMatrixCSC)

Returns a vector `v` where each entry in the vector gives the aggregate that
that vertex belongs to. Ex: v[1] = 2 means that vertex 1 belongs to aggregate
2.
"""
function standard_aggregation(A :: SparseMatrixCSC)
    m, n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)

    # hold aggregates. -1 is unaggregated
    aggregates = fill(-1, n)

    # find initial aggregates
    current_aggregate = 1
    for i in 1:n
        if aggregates[i] == -1 # not yet aggregated
            neighbors = rows[nzrange(A, i)]
            if all(x -> aggregates[x] == -1, neighbors) # no neighbors are aggregated
                aggregates[i] = current_aggregate
                aggregates[neighbors] = current_aggregate
                current_aggregate += 1
            end
        end
    end

    # Pull neighboring nodes into aggregates
    for i in 1:n
        if aggregates[i] == -1
            # find max weight edge
            max_weight = 0
            ind = -1
            for j in nzrange(A, i)
                if rows[j] != i && aggregates[rows[j]] != -1
                    w = vals[j]
                    if w > max_weight
                        max_weight = w
                        ind = rows[j]
                    end
                end
            end

            if ind == -1
                error("isolated node")
            else
                aggregates[i] = aggregates[ind]
            end
        end
    end

    aggregates
end

end
