# SyncSort: Cross-Architecture Synchronised Sorting of Multiple Arrays

While key-value pairs are standard in sorting, oftentimes these keys and values are kept in separate arrays (see Explanation below); we can sort multiple secondary vectors following the values of a primary key vector using a new sorting interface:

```julia
syncsort!(v, rest...)

# Example interface usage
num_particles = 100_000
ids = rand(Int64, num_particles)
xcoords = rand(num_particles)
ycoords = rand(num_particles)

# Reordering applied to `ids` will be done in sync with `xcoords` and `ycoords`
syncsort!(ids, xcoords, ycoords)
```


## Explanation

Key-value pair sorting assumes an _Array of Structures_ (AoS) format - e.g., a `Vector{Particle}` where `struct Particle` contains some fields `coord::Float64` and `id::Int64`. However, in many applications, it is more performant to store these fields as a _Structure of Arrays_ (SoA), i.e. separate `coords::Vector{Float64` and `ids::Vector{Int64}`.

If we had an Array of Structures, using standard sorting interfaces - as is the case in virtually all programming languages - we'd do something like:

```julia
# One primary key and two secondary values
struct Particle
    id::Int64
    xcoord::Float64
    ycoord::Float64
end

# Array of structures
particles = [Particle(rand(Int64), rand(Float64)) for i in 1:10_000]

# Sort by the field `id`
sort!(particles; by=p->p.id)
```

In high-performance applications, we'd use a Structure of Arrays, where we have separate arrays for our keys and values; the typical solution would involve creating a mask of indices sorting the primary array, in Julia named `sortperm`:

```julia
# Keys and values held in separate vectors
ids = rand(Int64, 10_000)
xcoords = rand(Float64, 10_000)
ycoords = rand(Float64, 10_000)

# Get indices permutation that sorts the primary array
sorted_indices = sortperm(ids)

# Use indices mask to permute arrays
ids = ids[sorted_indices]
xcoords = xcoords[sorted_indices]
ycoords = ycoords[sorted_indices]
```

An easier and much more performant solution - in terms of time, space, and especially program (re-)design - would be `syncsort`:

```julia
using SyncSort

# Keys and values held in separate vectors
ids = rand(Int64, 10_000)
xcoords = rand(Float64, 10_000)
ycoords = rand(Float64, 10_000)

syncsort!(ids, xcoords, ycoords)
```


## CPU Sorting Credits

The CPU sorting routines in `src/cpu_sort.jl` are copied verbatim from the Julia standard library, with some changes to propagate the secondary arrays and follow the swaps done by the standard sorting routines. All credits go to the Julia contributors, with special thanks to @LilithHafner for writing one of the most performant sorting algorithms of [any programming language](https://github.com/LilithHafner/InterLanguageSortingComparisons) - thank you! The Julia license text is included within the file. 


## License

This package is shared under the same license as the Julia standard library, MIT.
