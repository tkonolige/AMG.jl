"""
Levels in a hierarchy must have the following interface:
    1. restrict(level, x, b)
    2. prehook(level, x, b)
    3. posthook(level, x, b)
    4. prolongate(level, x, b)

TODO: I'm not sure about this interface.
"""
module Hierarchy

using SparseRelaxation

export prolongator_from_aggregates, AMGLevelm, DirectSolve, create_amg_level,
       create_amg_hierarchy, vcycle

"""
    prolongator_from_aggregates(Agg :: Vector)

Creates a prolongation operator `P` from `Agg`.
"""
prolongator_from_aggregates(Agg :: Vector) = sparse(1:length(Agg), Agg, ones(Agg))

"""
    immutable AMGLevel
        A :: SparseMatrixCSC # Matrix
        R :: SparseMatrixCSC # Restriction
        P :: SparseMatrixCSC # Prolongation
    end
"""
immutable AMGLevel
    A :: SparseMatrixCSC
    R :: SparseMatrixCSC
    P :: SparseMatrixCSC
end

immutable DirectSolve
    A :: SparseMatrixCSC
end

function create_amg_level(A :: SparseMatrixCSC, aggregation, strength)
    P = prolongator_from_aggregates(aggregation(strength(A)))
    R = transpose(P)
    AMGLevel(A, R, P)
end

# TODO: use coarse nnz or size?
function create_amg_hierarchy(A :: SparseMatrixCSC, aggregation, strength, coarse_size)
    if size(A, 1) <= coarse_size
        [DirectSolve(A)]
    else
        lvl = create_amg_level(A, aggregation, strength)
        [lvl; create_amg_hierarchy(lvl.R*A*lvl.P, aggregation, strength, coarse_size)]
    end
end

vcycle(lvl :: DirectSolve, x, b, levels) = lvl.A \ b

function vcycle(lvl :: AMGLevel, x, b, levels)
    gauss_seidel!(lvl.A, x, b)
    #= xr = lvl.R * x =#
    br = lvl.R * b
    x += lvl.P * vcycle(levels[1], zeros(br), br, levels[2:end])
    gauss_seidel!(lvl.A, x, b)
    x
end

vcycle(levels, x, b) = vcycle(levels[1], x, b, levels[2:end])

end
