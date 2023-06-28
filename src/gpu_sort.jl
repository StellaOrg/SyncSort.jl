# File   : SyncSort.jl
# License: MIT
# Author : Dominik Werner <d.wer2@gmx.de>
# Date   : 26.01.2023
# Paper  : https://drive.google.com/file/d/0B7uLFueU4vLfcjJfZFh3TlIxMFE/view?resourcekey=0-8Ovsx4PtAJn78xLboBEb_g



module SyncSort


using Base.Order
using Base: copymutable, midpoint, require_one_based_indexing, uinttype,
    sub_with_overflow, add_with_overflow, OneTo, BitSigned, BitIntegerType,
    IteratorSize, HasShape, IsInfinite, tail, midpoint

import Base:
    sort,
    sort!,
    issorted,
    sortperm,
    to_indices,
    midpoint

using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll



export merge_sort_gpu!


"""
Sort an array in parallel using a merge sort algorithm.

The array is sorted in place.

# Arguments
- `v`: array to be sorted
- `elements_per_thread`: number of elements to be sorted by each thread
- `v2`: temporary array to be used for swapping
- `global_tile_ranks`: temporary array to be used for storing the ranks of the splitters
"""
function merge_sort_gpu!(
    v::AbstractVector,
    ::Val{elements_per_thread},
    v2::AbstractVector=similar(v),
    #global_tile_ranks::AbstractVector=CuArray(fill((-1, -1), (div(length(v), elements_per_thread, RoundUp)))),
    global_tile_ranks::AbstractVector=similar(v,typeof((-1, -1)), (div(length(v), elements_per_thread, RoundUp))),
    ) where {elements_per_thread}

    # check on which backend we are running and set the device
    dev = get_backend(v)

    # Keep a pointer to the original array as we'll be swapping
    vpointer = v
    vlength = length(v)

    # Calculate the number of threads based on the number of elements and the length of the input array
    num_threads = div(vlength, elements_per_thread, RoundUp)

    # sort each block consisting of 'elements_per_thread' elements using insertion_sort
    initial_sort_kernel!(dev, 64)(v, elements_per_thread, ndrange=num_threads)
    KernelAbstractions.synchronize(dev)


    # Step 3: Merging
    subblock_size = elements_per_thread
    merge_block_size = elements_per_thread * 2
    threads_per_merge_block = 2
    # Every loop iteration, double the merge block size
    generate_ranks_kernel_function! = generate_ranks_kernel!(dev, 64)
    merge_two_blocks_kernel_function! = merge_two_blocks_kernel!(dev, 64)

    while merge_block_size < vlength * 2
        # The number of blocks we need to merge
        num_blocks = div(vlength, merge_block_size, RoundUp)
        num_subblocks = div(2 * vlength, merge_block_size, RoundUp)

        # Calculate the ranks for each thread
        generate_ranks_kernel_function!(
            v,
            global_tile_ranks,
            merge_block_size,
            threads_per_merge_block,
            num_threads,
            subblock_size,
            num_subblocks,
            num_blocks,
            ndrange=num_threads)
        KernelAbstractions.synchronize(dev)

        # Merge phase.
        merge_two_blocks_kernel_function!(
            v, v2,
            global_tile_ranks,
            merge_block_size,
            threads_per_merge_block,
            num_threads,
            subblock_size,
            num_subblocks,
            num_blocks,
            ndrange=num_threads)
        KernelAbstractions.synchronize(dev)

        # Swap arrays
        v, v2 = v2, v

        # The length of an array subsection that is sorted is now bigger
        subblock_size = merge_block_size
        merge_block_size *= 2
        threads_per_merge_block *= 2
    end

    # If we ended up with the second array as the result, swap it back
    if v2 === vpointer
        vpointer .= v
    end

    nothing
end 

@kernel function initial_sort_kernel!(v, elements_per_thread)
    # xtract global index and the start and end of this block
    index = @index(Global, Linear)
    start_array_index = (index - 1) * elements_per_thread + 1
    end_array_index = min(start_array_index + elements_per_thread - 1, size(v, 1))

    # now simply sort a view of this array
    sub_v = @view v[start_array_index:end_array_index]
    insertion_sort!(sub_v)
    #sort!(sub_v)
    nothing
end


function insertion_sort!(
    v::AbstractVector,
) 
    i = firstindex(v) + 1
    @inbounds while i <= lastindex(v)
        x = v[i]
        j = i - 1
        while j >= firstindex(v) && v[j] > x
            v[j+1] = v[j]
            j -= 1
        end
        v[j+1] = x
        i += 1
    end
    nothing
end


@kernel function generate_ranks_kernel!(v,
    ranks,
    merge_block_size,
    threads_per_merge_block,
    num_threads,
    subblock_size,
    num_subblocks,
    num_blocks,
)

    index = @index(Global, Linear)
    if threads_per_merge_block > num_threads
        threads_per_merge_block = num_threads
    end

    # First we calculate where the merge block starts and which rank
    # inside the merge block the current thread is
    merge_block_index = div(index - 1, threads_per_merge_block, RoundDown) + 1
    merge_block_start_index = (merge_block_index - 1) * merge_block_size + 1
    merge_block_end_index = merge_block_start_index + merge_block_size - 1
    if merge_block_end_index > size(v, 1)
        # we are at the last merge block
        merge_block_end_index = size(v, 1)
    end
    merge_block = @view v[merge_block_start_index:merge_block_end_index]

    # calculate the current rank id of the thread within the merge block
    rank_id = mod(index - 1, threads_per_merge_block) + 1

    if merge_block_index == num_blocks && num_subblocks % 2 == 1
        # do nothing because we are copying this chunk later
    else

        if rank_id == threads_per_merge_block || index == num_threads
            #last ranks are always the last elements in subblocks 
            ranks[index] = (subblock_size, size(merge_block, 1))

        else

            # Now we find the index for the current rank, search fr it in the
            # right rank with binary search and store it in "ranks"
            index_splitter_left_subblock = div(
                rank_id * subblock_size,
                threads_per_merge_block,
                RoundDown,
            )
            splitter_left_subblock = merge_block[index_splitter_left_subblock]
            if splitter_left_subblock < merge_block[subblock_size+1]
                # if the smallest element of the right subblock, then we dont need to search
                ranks[index] = (index_splitter_left_subblock, subblock_size)
            else
                right_subblock = @view merge_block[subblock_size+1:end]
                index_splitter_right_subblock = subblock_size + binary_search(
                    right_subblock,
                    splitter_left_subblock,
                )
                ranks[index] = (index_splitter_left_subblock, index_splitter_right_subblock)
            end
        end
    end

end

# Binary search to find the rank of an element within a block
# array must be ordered (obviously)
function binary_search(array::AbstractVector, target)
    low = 1
    high = size(array, 1)
    @inbounds while low < high
        index = low + div(high - low + 1, 2, RoundDown)
        if target < array[index]
            high = index - 1
        else
            low = index
        end
    end
    low
end



@kernel function merge_two_blocks_kernel!(
    v, v2, tile_ranks,
    merge_block_size,
    threads_per_merge_block,
    num_threads,
    subblock_size,
    num_subblocks,
    num_blocks,
)
    index = @index(Global, Linear)
    # First we calculate where the merge block starts and which rank
    # inside the merge block the current thread is
    merge_block_index = div(index - 1, threads_per_merge_block, RoundDown) + 1
    merge_block_start_index = (merge_block_index - 1) * merge_block_size + 1
    merge_block_end_index = merge_block_start_index + merge_block_size - 1


    if merge_block_end_index > size(v, 1)
        # we are at the last merge block
        merge_block_end_index = size(v, 1)
        #threads_per_merge_block = div(merge_block_end_index - merge_block_start_index + 1, elements_per_thread, RoundUp)
    end
    if threads_per_merge_block > num_threads
        threads_per_merge_block = num_threads
    end


    # calculate the current rank id of the thread within the merge block
    rank_id = mod(index - 1, threads_per_merge_block) + 1

    merge_block = @view v[merge_block_start_index:merge_block_end_index]

    if merge_block_index == num_blocks && num_subblocks % 2 == 1
        # copy the last subblock to the output array
        if rank_id == 1
            for i in merge_block_start_index:merge_block_end_index
                v2[i] = v[i]
            end
        end
    else
        # Now we find the index for the current rank, search
        # for the element in the other block and insert it
        left_end, right_end = tile_ranks[index]
        # print all data with @print

        if rank_id == 1
            left_start = firstindex(v)
            right_start = firstindex(v) + subblock_size
            
            merge_array_insert_start_index = firstindex(v)
        else
            left_start = tile_ranks[index-1][1] + 1
            right_start = tile_ranks[index-1][2] + 1
            merge_array_insert_start_index = (
                tile_ranks[index-1][1] +
                tile_ranks[index-1][2] - subblock_size +
                1
            )
        end




        tile_a = @view merge_block[left_start:left_end]
        tile_b = @view merge_block[right_start:right_end]
        #println(" right start: $right_start, $right_end, $merge_block_index")
        size_a = size(tile_a, 1)
        size_b = size(tile_b, 1)
        merge_tile_start_index = merge_block_start_index + merge_array_insert_start_index - 1
        merge_tile_end_index = merge_block_start_index + merge_array_insert_start_index + size_a + size_b - 2
        if merge_tile_end_index > size(v2, 1)
            merge_tile_end_index = size(v2, 1)
        end
        merge_tile = @view v2[merge_tile_start_index:merge_tile_end_index]
        merge_subblock!(merge_tile, tile_a, tile_b)
        #println("Index: $index\na: $tile_a\nb: $tile_b\nMerged: $merge_tile")
    end
end


# Merge two blocks A and B, running on a single thread
# Optimization notes:
#   type of those blocks is the same, something like ::view{::AbstractVector{T}}
#   with a constant size, it should be always the same
# do i wnat this to be in place or should it return?
function merge_subblock!(merge_tile, tile_a, tile_b)
    @assert size(merge_tile,1) == size(tile_a,1) + size(tile_b,1)

    tile_a_size = size(tile_a, 1)
    tile_b_size = size(tile_b, 1)

    # This could be redundant
    if tile_a_size == 0
        #println("\n\nFUCK YOU DOMINIK\n\n")
        @inbounds for i in axes(merge_tile, 1)
            merge_tile[i] = tile_b[i]
        end
        return
    end

    if tile_b_size == 0
        @inbounds for i in axes(merge_tile, 1)
            merge_tile[i] = tile_a[i]
        end
        return
    end

    a_index = firstindex(tile_a)
    b_index = firstindex(tile_b)

    # Merge all elements. some might be left over
    @inbounds for merge_index in axes(merge_tile, 1)
        if tile_a[a_index] <= tile_b[b_index]
            merge_tile[merge_index] = tile_a[a_index]
            a_index += 1

            if a_index > tile_a_size
                # Merge the rest of the b-block
                # TODO: make this looping more elegant
                j = 0
                for i in merge_index+1:lastindex(merge_tile)
                    merge_tile[i] = tile_b[b_index+j]
                    j += 1
                end
                break
            end
        else
            merge_tile[merge_index] = tile_b[b_index]
            b_index += 1

            if b_index > tile_b_size
                # Merge the rest of the a-block
                j = 0
                for i in merge_index+1:lastindex(merge_tile)
                    merge_tile[i] = tile_a[a_index+j]
                    j += 1
                end
                break
            end
        end
    end
end

lt=isless

"""
Quicksort taken from Julia/base
"""
midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
midpoint(lo::Integer, hi::Integer) = midpoint(promote(lo, hi)...)

function quicksort!(v::AbstractVector, lo::Integer, hi::Integer)
    @inbounds while lo < hi
        hi-lo <= 20 && return sort!(v, lo, hi, InsertionSort, Base.Order.Forward)
        j = partition!(v, lo, hi)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && quicksort!(v, lo, j-1)
            lo = j+1
        else
            j+1 < hi && quicksort!(v, j+1, hi)
            hi = j-1
        end
    end
    return v
end

function partition!(v::AbstractVector, lo::Integer, hi::Integer)
    pivot = selectpivot!(v, lo, hi)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while isless( v[i], pivot); i += 1; end;
        while isless( pivot, v[j]); j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
    end
    v[j], v[lo] = pivot, v[j]
    return j
end

@inline function selectpivot!(v::AbstractVector, lo::Integer, hi::Integer)
    @inbounds begin
        mi = midpoint(lo, hi)

        # sort v[mi] <= v[lo] <= v[hi] such that the pivot is immediately in place
        if isless( v[lo], v[mi])
            v[mi], v[lo] = v[lo], v[mi]
        end

        if isless( v[hi], v[lo])
            if isless( v[hi], v[mi])
                v[hi], v[lo], v[mi] = v[lo], v[mi], v[hi]
            else
                v[hi], v[lo] = v[lo], v[hi]
            end
        end

        # return the pivot
        return v[lo]
    end
end




########## Scratch Quick Sort from Julia/base
Algorithm = Base.Sort.Algorithm
SMALL_ALGORITHM = InsertionSort
macro getkw(syms...)
    getters = (getproperty(Base.Sort, Symbol(:_, sym)) for sym in syms)
    Expr(:block, (:($(esc(:((kw, $sym) = $getter(v, kw))))) for (sym, getter) in zip(syms, getters))...)
end


"""
    make_scratch(scratch::Union{Nothing, Vector}, T::Type, len::Integer)

Returns `(s, t)` where `t` is an `AbstractVector` of type `T` with length at least `len`
that is backed by the `Vector` `s`. If `scratch !== nothing`, then `s === scratch`.

This function will allocate a new vector if `scratch === nothing`, `resize!` `scratch` if it
is too short, and `reinterpret` `scratch` if its eltype is not `T`.
"""
function make_scratch(scratch::Nothing, T::Type, len::Integer)
    s = Vector{T}(undef, len)
    s, s
end
function make_scratch(scratch::Vector{T}, ::Type{T}, len::Integer) where T
    len > length(scratch) && resize!(scratch, len)
    scratch, scratch
end
function make_scratch(scratch::Vector, T::Type, len::Integer)
    len_bytes = len * sizeof(T)
    len_scratch = div(len_bytes, sizeof(eltype(scratch)))
    len_scratch > length(scratch) && resize!(scratch, len_scratch)
    scratch, reinterpret(T, scratch)
end


struct ScratchQuickSort{L<:Union{Integer,Missing}, H<:Union{Integer,Missing}, T<:Algorithm} <: Algorithm
    lo::L
    hi::H
    next::T
end
ScratchQuickSort(next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(missing, missing, next)
ScratchQuickSort(lo::Union{Integer, Missing}, hi::Union{Integer, Missing}) = ScratchQuickSort(lo, hi, SMALL_ALGORITHM)
ScratchQuickSort(lo::Union{Integer, Missing}, next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(lo, lo, next)
ScratchQuickSort(r::OrdinalRange, next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(first(r), last(r), next)

function scratchpartition!(t::AbstractVector, lo::Integer, hi::Integer, offset::Integer,
    v::AbstractVector, rev::Bool, pivot_dest::AbstractVector, pivot_index_offset::Integer)
    # Ideally we would use `pivot_index = rand(lo:hi)`, but that requires Random.jl
    # and would mutate the global RNG in sorting.
    pivot_index = typeof(hi-lo)(hash(lo) % (hi-lo+1)) + lo
    @inbounds begin
        pivot = v[pivot_index]
        while lo < pivot_index
            x = v[lo]
            fx = rev ? !isless( x, pivot) : isless( pivot, x)
            t[(fx ? hi : lo) - offset] = x
            offset += fx
            lo += 1
        end
        while lo < hi
            x = v[lo+1]
            fx = rev ? isless( pivot, x) : !isless( x, pivot)
            t[(fx ? hi : lo) - offset] = x
            offset += fx
            lo += 1
        end
        pivot_index = lo-offset + pivot_index_offset
        pivot_dest[pivot_index] = pivot
    end

    # t_pivot_index = lo-offset (i.e. without pivot_index_offset)
    # t[t_pivot_index] is whatever it was before unless t is the pivot_dest
    # t[<t_pivot_index] <* pivot, stable
    # t[>t_pivot_index] >* pivot, reverse stable

    pivot_index
end



function scratchquicksort!(v::AbstractVector, a::ScratchQuickSort=ScratchQuickSort(), 
    t=nothing, offset=nothing, swap=false, rev=false, lo = firstindex(v), hi = size(v,1), scrath = nothing)

    if t === nothing
        scratch, t = make_scratch(scratch, eltype(v), hi-lo+1)
        offset = 1-lo
        kw = (;kw..., scratch)
    end

    while lo < hi && hi - lo > 20
        j = if swap
            scratchpartition!(v, lo+offset, hi+offset, offset, t, rev, v, 0)
        else
            scratchpartition!(t, lo, hi, -offset, v, rev, v, -offset)
        end
        swap = !swap

        # For ScratchQuickSort(), a.lo === a.hi === missing, so the first two branches get skipped
        if !ismissing(a.lo) && j <= a.lo # Skip sorting the lower part
            swap && copyto!(v, lo, t, lo+offset, j-lo)
            rev && reverse!(v, lo, j-1)
            lo = j+1
            rev = !rev
        elseif !ismissing(a.hi) && a.hi <= j # Skip sorting the upper part
            swap && copyto!(v, j+1, t, j+1+offset, hi-j)
            rev || reverse!(v, j+1, hi)
            hi = j-1
        elseif j-lo < hi-j
            # Sort the lower part recursively because it is smaller. Recursing on the
            # smaller part guarantees O(log(n)) stack space even on pathological inputs.
            scratchquicksort!(v, a, t, offset, swap, rev,lo = lo, hi=j-1, scratch = scratch)
            lo = j+1
            rev = !rev
        else # Sort the higher part recursively
            scratchquicksort!(v, a, t, offset, swap, rev=!rev, lo = j+1, hi = hi, scratch = scratch)
            hi = j-1
        end
    end
    hi < lo && return scratch
    swap && copyto!(v, lo, t, lo+offset, hi-lo+1)
    rev && reverse!(v, lo, hi)
    #_sort!(v, a.next, o, (;kw..., lo, hi))
    v_view = @view v[lo:hi]
    insertion_sort!(v_view)
end
################ RADIX SORT
top_set_bit(x::Integer) = ceil(Integer, log2(x + oneunit(x)))
"""
    uint_unmap(T::Type, u::Unsigned, order::Ordering)

Reconstruct the unique value `x::T` that uint_maps to `u`. Satisfies
`x === uint_unmap(T, uint_map(x::T, order), order)` for all `x <: T`.

See also: [`uint_map`](@ref) [`UIntMappable`](@ref)
"""
function uint_unmap end


### Primitive Types

# Integers
uint_map(x::Unsigned, ::ForwardOrdering) = x
uint_unmap(::Type{T}, u::T, ::ForwardOrdering) where T <: Unsigned = u

uint_map(x::Signed, ::ForwardOrdering) =
    unsigned(xor(x, typemin(x)))
uint_unmap(::Type{T}, u::Unsigned, ::ForwardOrdering) where T <: Signed =
    xor(signed(u), typemin(T))

UIntMappable(T::BitIntegerType, ::ForwardOrdering) = unsigned(T)

# Floats are not UIntMappable under regular orderings because they fail on NaN edge cases.
# uint mappings for floats are defined in Float, where the Left and Right orderings
# guarantee that there are no NaN values

# Chars
uint_map(x::Char, ::ForwardOrdering) = reinterpret(UInt32, x)
uint_unmap(::Type{Char}, u::UInt32, ::ForwardOrdering) = reinterpret(Char, u)
UIntMappable(::Type{Char}, ::ForwardOrdering) = UInt32

### Reverse orderings
uint_map(x, rev::ReverseOrdering) = ~uint_map(x, rev.fwd)
uint_unmap(T::Type, u::Unsigned, rev::ReverseOrdering) = uint_unmap(T, ~u, rev.fwd)
UIntMappable(T::Type, order::ReverseOrdering) = UIntMappable(T, order.fwd)


### Vectors

# Convert v to unsigned integers in place, maintaining sort order.
function uint_map!(v::AbstractVector, lo::Integer, hi::Integer, order::Ordering)
    u = reinterpret(UIntMappable(eltype(v), order), v)
    @inbounds for i in lo:hi
        u[i] = uint_map(v[i], order)
    end
    u
end

function uint_unmap!(v::AbstractVector, u::AbstractVector{U}, lo::Integer, hi::Integer,
                     order::Ordering, offset::U=zero(U),
                     index_offset::Integer=0) where U <: Unsigned
    @inbounds for i in lo:hi
        v[i] = uint_unmap(eltype(v), u[i+index_offset]+offset, order)
    end
    v
end

"""
    send_to_end!(f::Function, v::AbstractVector; [lo, hi])

Send every element of `v` for which `f` returns `true` to the end of the vector and return
the index of the last element for which `f` returns `false`.

`send_to_end!(f, v, lo, hi)` is equivalent to `send_to_end!(f, view(v, lo:hi))+lo-1`

Preserves the order of the elements that are not sent to the end.
"""
function send_to_end!(f::F, v::AbstractVector; lo=firstindex(v), hi=lastindex(v)) where F <: Function
    i = lo
    @inbounds while i <= hi && !f(v[i])
        i += 1
    end
    j = i + 1
    @inbounds while j <= hi
        if !f(v[j])
            v[i], v[j] = v[j], v[i]
            i += 1
        end
        j += 1
    end
    i - 1
end
"""
    send_to_end!(f::Function, v::AbstractVector, o::DirectOrdering[, end_stable]; lo, hi)

Return `(a, b)` where `v[a:b]` are the elements that are not sent to the end.

If `o isa ReverseOrdering` then the "end" of `v` is `v[lo]`.

If `end_stable` is set, the elements that are sent to the end are stable instead of the
elements that are not
"""
@inline send_to_end!(f::F, v::AbstractVector, ::ForwardOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (lo, hi-send_to_end!(!f, view(v, hi:-1:lo))) : (lo, send_to_end!(f, v; lo, hi))
@inline send_to_end!(f::F, v::AbstractVector, ::ReverseOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (send_to_end!(!f, v; lo, hi)+1, hi) : (hi-send_to_end!(f, view(v, hi:-1:lo))+1, hi)




lt = isless
after_zero(::ForwardOrdering, x) = !signbit(x)
after_zero(::ReverseOrdering, x) = signbit(x)
is_concrete_IEEEFloat(T::Type) = T <: Base.IEEEFloat && isconcretetype(T)
function radixsort!(v::AbstractVector, o::DirectOrdering=Forward)
    scratch = nothing
    lo = firstindex(v)
    hi = size(v,1)

    if is_concrete_IEEEFloat(eltype(v)) && o isa DirectOrdering
        lo, hi = send_to_end!(isnan, v, o, true; lo, hi)
        iv = reinterpret(uinttype(eltype(v)), v)
        j = send_to_end!(x -> after_zero(o, x), v; lo, hi)
        scratch = _sort!(iv, a.next, Reverse, (;kw..., lo, hi=j))
    elseif !(eltype(v) == Int || eltype(v) == UInt)
        throw(error("The type of your array is not suitable for radixsort"))
    end
    mn = mx = v[lo]
    @inbounds for i in (lo+1):hi
        vi = v[i]
        lt(vi, mn) && (mn = vi)
        lt( mx, vi) && (mx = vi)
    end

    lt( mn, mx) || return scratch # all same

    umn = uint_map(mn, o)
    urange = uint_map(mx, o)-umn
    bits = unsigned(top_set_bit(urange))

    # At this point, we are committed to radix sort.
    u = uint_map!(v, lo, hi, o)

    # we subtract umn to avoid radixing over unnecessary bits. For example,
    # Int32[3, -1, 2] uint_maps to UInt32[0x80000003, 0x7fffffff, 0x80000002]
    # which uses all 32 bits, but once we subtract umn = 0x7fffffff, we are left with
    # UInt32[0x00000004, 0x00000000, 0x00000003] which uses only 3 bits, and
    # Float32[2.012, 400.0, 12.345] uint_maps to UInt32[0x3fff3b63, 0x3c37ffff, 0x414570a4]
    # which is reduced to UInt32[0x03c73b64, 0x00000000, 0x050d70a5] using only 26 bits.
    # the overhead for this subtraction is small enough that it is worthwhile in many cases.

    # this is faster than u[lo:hi] .-= umn as of v1.9.0-DEV.100
    @inbounds for i in lo:hi
        u[i] -= umn
    end

    scratch, t = make_scratch(scratch, eltype(v), hi-lo+1)
    tu = reinterpret(eltype(u), t)
    if radix_sort!(u, lo, hi, bits, tu, 1-lo)
        uint_unmap!(v, u, lo, hi, o, umn)
    else
        uint_unmap!(v, tu, lo, hi, o, umn, 1-lo)
    end
    scratch
end


function radix_chunk_size_heuristic(lo::Integer, hi::Integer, bits::Unsigned)
    # chunk_size is the number of bits to radix over at once.
    # We need to allocate an array of size 2^chunk size, and on the other hand the higher
    # the chunk size the fewer passes we need. Theoretically, chunk size should be based on
    # the Lambert W function applied to length. Empirically, we use this heuristic:
    guess = min(10, log(maybe_unsigned(hi-lo))*3/4+3)
    # TODO the maximum chunk size should be based on architecture cache size.

    # We need iterations * chunk size ≥ bits, and these cld's
    # make an effort to get iterations * chunk size ≈ bits
    UInt8(cld(bits, cld(bits, guess)))
end


# The return value indicates whether v is sorted (true) or t is sorted (false)
# This is one of the many reasons radix_sort! is not exported.
function radix_sort!(v::AbstractVector{U}, lo::Integer, hi::Integer, bits::Unsigned,
    t::AbstractVector{U}, offset::Integer,
    chunk_size=radix_chunk_size_heuristic(lo, hi, bits)) where U <: Unsigned
    # bits is unsigned for performance reasons.
    counts = Vector{Int}(undef, 1 << chunk_size + 1) # TODO use scratch for this

    shift = 0
    while true
        @noinline radix_sort_pass!(t, lo, hi, offset, counts, v, shift, chunk_size)
        # the latest data resides in t
        shift += chunk_size
        shift < bits || return false
        @noinline radix_sort_pass!(v, lo+offset, hi+offset, -offset, counts, t, shift, chunk_size)
        # the latest data resides in v
        shift += chunk_size
        shift < bits || return true
    end
end
function radix_sort_pass!(t, lo, hi, offset, counts, v, shift, chunk_size)
    mask = UInt(1) << chunk_size - 1  # mask is defined in pass so that the compiler
    @inbounds begin                   #  ↳ knows it's shape
        # counts[2:mask+2] will store the number of elements that fall into each bucket.
        # if chunk_size = 8, counts[2] is bucket 0x00 and counts[257] is bucket 0xff.
        counts .= 0
        for k in lo:hi
            x = v[k]                  # lookup the element
            i = (x >> shift)&mask + 2 # compute its bucket's index for this pass
            counts[i] += 1            # increment that bucket's count
        end

        counts[1] = lo + offset       # set target index for the first bucket
        cumsum!(counts, counts)       # set target indices for subsequent buckets
        # counts[1:mask+1] now stores indices where the first member of each bucket
        # belongs, not the number of elements in each bucket. We will put the first element
        # of bucket 0x00 in t[counts[1]], the next element of bucket 0x00 in t[counts[1]+1],
        # and the last element of bucket 0x00 in t[counts[2]-1].

        for k in lo:hi
            x = v[k]                  # lookup the element
            i = (x >> shift)&mask + 1 # compute its bucket's index for this pass
            j = counts[i]             # lookup the target index
            t[j] = x                  # put the element where it belongs
            counts[i] = j + 1         # increment the target index for the next
        end                           #  ↳ element in this bucket
    end
end


end # module MergeSortGPU
