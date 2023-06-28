# This file was copied from https://github.com/JuliaLang/julia/blob/v1.9.1/base/sort.jl with some
# modifications to propagate secondary arrays to be sorted and follow swaps. The original license
# is included below. Thank you!

# MIT License

# Copyright (c) 2009-2023: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors: https://github.com/JuliaLang/julia/contributors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# end of terms and conditions

# Please see [THIRDPARTY.md](./THIRDPARTY.md) for license information for other software used in this project.


module CPUSort

# Relevant exports
export syncsort!
export RadixSort, ScratchQuickSort, InsertionSort


# Internal imports
using Base.Order

using Base: copymutable, midpoint, require_one_based_indexing, uinttype,
    sub_with_overflow, add_with_overflow, OneTo, BitSigned, BitIntegerType,
    IteratorSize, HasShape, IsInfinite, tail
top_set_bit(x::Integer) = ceil(Integer, log2(x + oneunit(x)))


## Alternative keyword management

macro getkw(syms...)
    getters = (getproperty(CPUSort, Symbol(:_, sym)) for sym in syms)
    Expr(:block, (:($(esc(:((kw, $sym) = $getter(v, o, kw))))) for (sym, getter) in zip(syms, getters))...)
end

for (sym, exp, type) in [
        (:lo, :(firstindex(v)), Integer),
        (:hi, :(lastindex(v)),  Integer),
        (:mn, :(throw(ArgumentError("mn is needed but has not been computed"))), :(eltype(v))),
        (:mx, :(throw(ArgumentError("mx is needed but has not been computed"))), :(eltype(v))),
        (:scratch, nothing, :(Union{Nothing, Vector})), # could have different eltype
        ]
    usym = Symbol(:_, sym)
    @eval function $usym(v, o, kw)
        # using missing instead of nothing because scratch could === nothing.
        res = get(kw, $(Expr(:quote, sym)), missing)
        res !== missing && return kw, res::$type
        $sym = $exp
        (;kw..., $sym), $sym::$type
    end
end

## Scratch space management

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


## sorting algorithm components ##

"""
    _sort!(v::AbstractVector, a::Algorithm, o::Ordering, kw; t, offset)

An internal function that sorts `v` using the algorithm `a` under the ordering `o`,
subject to specifications provided in `kw` (such as `lo` and `hi` in which case it only
sorts `view(v, lo:hi)`)

Returns a scratch space if provided or constructed during the sort, or `nothing` if
no scratch space is present.

!!! note
    `_sort!` modifies but does not return `v`.

A returned scratch space will be a `Vector{T}` where `T` is usually the eltype of `v`. There
are some exceptions, for example if `eltype(v) == Union{Missing, T}` then the scratch space
may be be a `Vector{T}` due to `MissingOptimization` changing the eltype of `v` to `T`.

`t` is an appropriate scratch space for the algorithm at hand, to be accessed as
`t[i + offset]`. `t` is used for an algorithm to pass a scratch space back to itself in
internal or recursive calls.
"""
function _sort! end

abstract type Algorithm end


"""
    MissingOptimization(next) <: Algorithm

Filter out missing values.

Missing values are placed after other values according to `DirectOrdering`s. This pass puts
them there and passes on a view into the original vector that excludes the missing values.
This pass is triggered for both `sort([1, missing, 3])` and `sortperm([1, missing, 3])`.
"""
struct MissingOptimization{T <: Algorithm} <: Algorithm
    next::T
end

struct WithoutMissingVector{T, U} <: AbstractVector{T}
    data::U
    function WithoutMissingVector(data; unsafe=false)
        if !unsafe && any(ismissing, data)
            throw(ArgumentError("data must not contain missing values"))
        end
        new{nonmissingtype(eltype(data)), typeof(data)}(data)
    end
end
Base.@propagate_inbounds function Base.getindex(v::WithoutMissingVector, i)
    out = v.data[i]
    @assert !(out isa Missing)
    out::eltype(v)
end
Base.@propagate_inbounds function Base.setindex!(v::WithoutMissingVector, x, i)
    v.data[i] = x
    v
end
Base.size(v::WithoutMissingVector) = size(v.data)
Base.axes(v::WithoutMissingVector) = axes(v.data)

"""
    send_to_end!(f::Function, v::AbstractVector; [lo, hi])

Send every element of `v` for which `f` returns `true` to the end of the vector and return
the index of the last element for which `f` returns `false`.

`send_to_end!(f, v, lo, hi)` is equivalent to `send_to_end!(f, view(v, lo:hi))+lo-1`

Preserves the order of the elements that are not sent to the end.
"""
function send_to_end!(f::F, v::AbstractVector, rest; lo=firstindex(v), hi=lastindex(v)) where F <: Function
    i = lo
    @inbounds while i <= hi && !f(v[i])
        i += 1
    end
    j = i + 1
    @inbounds while j <= hi
        if !f(v[j])
            swap_rest!(v, rest, i, j)
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
@inline send_to_end!(f::F, v::AbstractVector, rest, ::ForwardOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (lo, hi-send_to_end!(!f, view(v, hi:-1:lo), view_rest(rest, hi:-1:lo))) : (lo, send_to_end!(f, v, rest; lo, hi))
@inline send_to_end!(f::F, v::AbstractVector, rest, ::ReverseOrdering, end_stable=false; lo, hi) where F <: Function =
    end_stable ? (send_to_end!(!f, v, rest; lo, hi)+1, hi) : (hi-send_to_end!(f, view(v, hi:-1:lo), view_rest(rest, hi:-1:lo))+1, hi)


"""
Swap two elements at indices i and j for the key vector v and secondary vectors in rest.

NEW ADDITION
"""
@inline function swap_rest!(v, rest, i, j)
    v[i], v[j] = v[j], v[i]
    for r in rest
        r[i], r[j] = r[j], r[i]
    end
end


"""
Construct views into each of the vectors of within a tuple.

NEW ADDITION
"""
@inline function view_rest(rest::Tuple, inds...)
    @inbounds tuple((view(r, inds...) for r in rest)...)
end


"""
Reverse each of the vectors within a tuple.

NEW ADDITION
"""
@inline function reverse_rest!(rest::Tuple, lo, hi)
    for r in rest
        reverse!(r, lo, hi)
    end
end


"""
Create copies of the vectors within a tuple.

NEW ADDITION
"""
@inline function similar_rest(rest::Tuple)
    tuple((similar(r) for r in rest)...)
end


@inline function _sort!(v::AbstractVector, rest, a::MissingOptimization, o::Ordering, kw)
    @getkw lo hi
    if o isa DirectOrdering && eltype(v) >: Missing && nonmissingtype(eltype(v)) != eltype(v)
        lo, hi = send_to_end!(ismissing, v, rest, o; lo, hi)
        _sort!(WithoutMissingVector(v, unsafe=true), rest, a.next, o, (;kw..., lo, hi))
    elseif o isa Perm && o.order isa DirectOrdering && eltype(v) <: Integer &&
                eltype(o.data) >: Missing && nonmissingtype(eltype(o.data)) != eltype(o.data) &&
                all(i === j for (i,j) in zip(v, eachindex(o.data)))

        throw(error("SyncSort not defined for Missing types with Perm ordering yet."))

        # TODO make this branch known at compile time
        # This uses a custom function because we need to ensure stability of both sides and
        # we can assume v is equal to eachindex(o.data) which allows a copying partition
        # without allocations.
        lo_i, hi_i = lo, hi
        for i in eachindex(o.data) # equal to copy(v)
            x = o.data[i]
            if ismissing(x) == (o.order == Reverse) # should x go at the beginning/end?
                v[lo_i] = i
                lo_i += 1
            else
                v[hi_i] = i
                hi_i -= 1
            end
        end
        reverse!(v, lo_i, hi)
        if o.order == Reverse
            lo = lo_i
        else
            hi = hi_i
        end

        _sort!(v, a.next, Perm(o.order, WithoutMissingVector(o.data, unsafe=true)), (;kw..., lo, hi), rest)
    else
        _sort!(v, rest, a.next, o, kw)
    end
end


"""
    IEEEFloatOptimization(next) <: Algorithm

Move NaN values to the end, partition by sign, and reinterpret the rest as unsigned integers.

IEEE floating point numbers (`Float64`, `Float32`, and `Float16`) compare the same as
unsigned integers with the bits with a few exceptions. This pass

This pass is triggered for both `sort([1.0, NaN, 3.0])` and `sortperm([1.0, NaN, 3.0])`.
"""
struct IEEEFloatOptimization{T <: Algorithm} <: Algorithm
    next::T
end

after_zero(::ForwardOrdering, x) = !signbit(x)
after_zero(::ReverseOrdering, x) = signbit(x)
is_concrete_IEEEFloat(T::Type) = T <: Base.IEEEFloat && isconcretetype(T)
@inline function _sort!(v::AbstractVector, rest, a::IEEEFloatOptimization, o::Ordering, kw)
    @getkw lo hi

    if is_concrete_IEEEFloat(eltype(v)) && o isa DirectOrdering
        lo, hi = send_to_end!(isnan, v, rest, o, true; lo, hi)
        iv = reinterpret(uinttype(eltype(v)), v)
        j = send_to_end!(x -> after_zero(o, x), v, rest; lo, hi)
        scratch = _sort!(iv, rest, a.next, Reverse, (;kw..., lo, hi=j))
        if scratch === nothing # Union split
            _sort!(iv, rest, a.next, Forward, (;kw..., lo=j+1, hi, scratch))
        else
            _sort!(iv, rest, a.next, Forward, (;kw..., lo=j+1, hi, scratch))
        end
    elseif eltype(v) <: Integer && o isa Perm && o.order isa DirectOrdering && is_concrete_IEEEFloat(eltype(o.data))
        lo, hi = send_to_end!(i -> isnan(@inbounds o.data[i]), v, rest, o.order, true; lo, hi)
        ip = reinterpret(uinttype(eltype(o.data)), o.data)
        j = send_to_end!(i -> after_zero(o.order, @inbounds o.data[i]), v, rest; lo, hi)
        scratch = _sort!(v, rest, a.next, Perm(Reverse, ip), (;kw..., lo, hi=j))
        if scratch === nothing # Union split
            _sort!(v, rest, a.next, Perm(Forward, ip), (;kw..., lo=j+1, hi, scratch))
        else
            _sort!(v, rest, a.next, Perm(Forward, ip), (;kw..., lo=j+1, hi, scratch))
        end
    else
        _sort!(v, rest, a.next, o, kw)
    end
end


"""
    BoolOptimization(next) <: Algorithm

Sort `AbstractVector{Bool}`s using a specialized version of counting sort.

Accesses each element at most twice (one read and one write), and performs at most two
comparisons.
"""
struct BoolOptimization{T <: Algorithm} <: Algorithm
    next::T
end
_sort!(v::AbstractVector, rest, a::BoolOptimization, o::Ordering, kw) = _sort!(v, rest, a.next, o, kw)
@inline function _sort!(v::AbstractVector{Bool}, rest, ::BoolOptimization, o::Ordering, kw)
    if size(rest,1) != 0
        throw(error("SyncSort is not defined for Vector{Bool} yet."))
    end
    first = lt(o, false, true) ? false : lt(o, true, false) ? true : return v
    @getkw lo hi scratch
    count = 0
    @inbounds for i in lo:hi
        if v[i] == first
            count += 1
        end
    end
    @inbounds v[lo:lo+count-1] .= first
    @inbounds v[lo+count:hi] .= !first
    scratch
end


"""
    IsUIntMappable(yes, no) <: Algorithm

Determines if the elements of a vector can be mapped to unsigned integers while preserving
their order under the specified ordering.

If they can be, dispatch to the `yes` algorithm and record the unsigned integer type that
the elements may be mapped to. Otherwise dispatch to the `no` algorithm.
"""
struct IsUIntMappable{T <: Algorithm, U <: Algorithm} <: Algorithm
    yes::T
    no::U
end
@inline function _sort!(v::AbstractVector, rest, a::IsUIntMappable, o::Ordering, kw)
    #println("Is uintmappable?")
    if UIntMappable(eltype(v), o) !== nothing
        _sort!(v, rest, a.yes, o, kw)
    else
        _sort!(v, rest, a.no, o, kw)
    end
end


"""
    Small{N}(small=SMALL_ALGORITHM, big) <: Algorithm

Sort inputs with `length(lo:hi) <= N` using the `small` algorithm. Otherwise use the `big`
algorithm.
"""
struct Small{N, T <: Algorithm, U <: Algorithm} <: Algorithm
    small::T
    big::U
end
Small{N}(small, big) where N = Small{N, typeof(small), typeof(big)}(small, big)
Small{N}(big) where N = Small{N}(SMALL_ALGORITHM, big)
@inline function _sort!(v::AbstractVector, rest, a::Small{N}, o::Ordering, kw) where N
    @getkw lo hi
    if (hi-lo) < N
        _sort!(v, rest, a.small, o, kw)
    else
        _sort!(v, rest,  a.big, o, kw)
    end
end


struct InsertionSortAlg <: Algorithm end

"""
    InsertionSort

Use the insertion sort algorithm.

Insertion sort traverses the collection one element at a time, inserting
each element into its correct, sorted position in the output vector.

Characteristics:
* *stable*: preserves the ordering of elements which compare equal
(e.g. "a" and "A" in a sort of letters which ignores case).
* *in-place* in memory.
* *quadratic performance* in the number of elements to be sorted:
it is well-suited to small collections but should not be used for large ones.
"""
const InsertionSort = InsertionSortAlg()
const SMALL_ALGORITHM = InsertionSortAlg()

@inline function _sort!(v::AbstractVector, rest, ::InsertionSortAlg, o::Ordering, kw)
    @getkw lo hi scratch
    lo_plus_1 = (lo + 1)::Integer

    @inbounds for i = lo_plus_1:hi
        j = i
        x = v[i]
        x_rest = tuple((r[i] for r in rest)...)
        while j > lo
            y = v[j-1]
            if !(lt(o, x, y)::Bool)
                break
            end
            v[j] = y
            
            for r in rest
                r[j] = r[j - 1]
            end
            j -= 1
        end
        v[j] = x
        for irest in eachindex(rest)
            rest[irest][j] = x_rest[irest]
        end
    end
    scratch
end


"""
    CheckSorted(next) <: Algorithm

Check if the input is already sorted and for large inputs, also check if it is
reverse-sorted. The reverse-sorted check is unstable.
"""
struct CheckSorted{T <: Algorithm} <: Algorithm
    next::T
end
@inline function _sort!(v::AbstractVector, rest, a::CheckSorted, o::Ordering, kw)
    @getkw lo hi scratch

    # For most arrays, a presorted check is cheap (overhead < 5%) and for most large
    # arrays it is essentially free (<1%).
    _issorted(v, lo, hi, o) && return scratch

    # For most large arrays, a reverse-sorted check is essentially free (overhead < 1%)
    if hi-lo >= 500 && _issorted(v, lo, hi, ReverseOrdering(o))
        # If reversing is valid, do so. This violates stability.
        reverse!(v, lo, hi)
        reverse_rest!(rest, lo, hi)
        return scratch
    end

    _sort!(v, rest, a.next, o, kw)
end


"""
    ComputeExtrema(next) <: Algorithm

Compute the extrema of the input under the provided order.

If the minimum is no less than the maximum, then the input is already sorted. Otherwise,
dispatch to the `next` algorithm.
"""
struct ComputeExtrema{T <: Algorithm} <: Algorithm
    next::T
end
@inline function _sort!(v::AbstractVector, rest, a::ComputeExtrema, o::Ordering, kw)
    @getkw lo hi scratch
    mn = mx = v[lo]
    @inbounds for i in (lo+1):hi
        vi = v[i]
        lt(o, vi, mn) && (mn = vi)
        lt(o, mx, vi) && (mx = vi)
    end

    lt(o, mn, mx) || return scratch # all same

    _sort!(v, rest, a.next, o, (;kw..., mn, mx))
end


"""
    ConsiderCountingSort(counting=CountingSort(), next) <: Algorithm

If the input's range is small enough, use the `counting` algorithm. Otherwise, dispatch to
the `next` algorithm.

For most types, the threshold is if the range is shorter than half the length, but for types
larger than Int64, bitshifts are expensive and RadixSort is not viable, so the threshold is
much more generous.
"""
struct ConsiderCountingSort{T <: Algorithm, U <: Algorithm} <: Algorithm
    counting::T
    next::U
end
ConsiderCountingSort(next) = ConsiderCountingSort(CountingSort(), next)
@inline function _sort!(v::AbstractVector{<:Integer}, a::ConsiderCountingSort, o::DirectOrdering, kw)
    @getkw lo hi mn mx
    range = maybe_unsigned(o === Reverse ? mn-mx : mx-mn)

    if range < (sizeof(eltype(v)) > 8 ? 5(hi-lo)-100 : div(hi-lo, 2))
        _sort!(v, a.counting, o, kw)
    else
        _sort!(v, a.next, o, kw)
    end
end
_sort!(v::AbstractVector, a::ConsiderCountingSort, o::Ordering, kw) = _sort!(v, a.next, o, kw)


"""
    CountingSort <: Algorithm

Use the counting sort algorithm.

`CountingSort` is an algorithm for sorting integers that runs in Θ(length + range) time and
space. It counts the number of occurrences of each value in the input and then iterates
through those counts repopulating the input with the values in sorted order.
"""
struct CountingSort <: Algorithm end
maybe_reverse(o::ForwardOrdering, x) = x
maybe_reverse(o::ReverseOrdering, x) = reverse(x)
@inline function _sort!(v::AbstractVector{<:Integer}, ::CountingSort, o::DirectOrdering, kw)

    @getkw lo hi mn mx scratch
    range = maybe_unsigned(o === Reverse ? mn-mx : mx-mn)
    offs = 1 - (o === Reverse ? mx : mn)

    counts = fill(0, range+1) # TODO use scratch (but be aware of type stability)
    @inbounds for i = lo:hi
        counts[v[i] + offs] += 1
    end

    idx = lo
    @inbounds for i = maybe_reverse(o, 1:range+1)
        lastidx = idx + counts[i] - 1
        val = i-offs
        for j = idx:lastidx
            v[j] = val isa Unsigned && eltype(v) <: Signed ? signed(val) : val
        end
        idx = lastidx + 1
    end

    scratch
end


"""
    ConsiderRadixSort(radix=RadixSort(), next) <: Algorithm

If the number of bits in the input's range is small enough and the input supports efficient
bitshifts, use the `radix` algorithm. Otherwise, dispatch to the `next` algorithm.
"""
struct ConsiderRadixSort{T <: Algorithm, U <: Algorithm} <: Algorithm
    radix::T
    next::U
end
ConsiderRadixSort(next) = ConsiderRadixSort(RadixSort(), next)
@inline function _sort!(v::AbstractVector, rest, a::ConsiderRadixSort, o::DirectOrdering, kw)
    @getkw lo hi mn mx
    urange = uint_map(mx, o)-uint_map(mn, o)
    bits = unsigned(top_set_bit(urange))
    if sizeof(eltype(v)) <= 8 && bits+70 < 22log(hi-lo)
        _sort!(v, rest, a.radix, o, kw)
    else
        _sort!(v, rest, a.next, o, kw)
    end
end


"""
    RadixSort <: Algorithm

Use the radix sort algorithm.

`RadixSort` is a stable least significant bit first radix sort algorithm that runs in
`O(length * log(range))` time and linear space.

It first sorts the entire vector by the last `chunk_size` bits, then by the second
to last `chunk_size` bits, and so on. Stability means that it will not reorder two elements
that compare equal. This is essential so that the order introduced by earlier,
less significant passes is preserved by later passes.

Each pass divides the input into `2^chunk_size == mask+1` buckets. To do this, it
 * counts the number of entries that fall into each bucket
 * uses those counts to compute the indices to move elements of those buckets into
 * moves elements into the computed indices in the swap array
 * switches the swap and working array

`chunk_size` is larger for larger inputs and determined by an empirical heuristic.
"""
struct RadixSort <: Algorithm end
@inline function _sort!(v::AbstractVector, rest, a::RadixSort, o::DirectOrdering, kw)
    @getkw lo hi mn mx scratch
    #println("In sort:", a,o,"\n\n\n")
    #println("Radixsort: $(eltype(v)), $(typeof(v))")
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
    #println("Scratch: $(typeof(scratch))")
    rest_scratch = similar_rest(rest)
    scratch, t = make_scratch(scratch, eltype(v), hi-lo+1)
    tu = reinterpret(eltype(u), t)
    #println(size(tu))
    #println(typeof((u, lo, hi, bits, tu, 1-lo, rest)))
    if radix_sort!(u, lo, hi, bits, tu, 1-lo, rest, rest_scratch)
        uint_unmap!(v, u, (), (), lo, hi, o, umn)
    else
        uint_unmap!(v, tu, rest, rest_scratch, lo, hi, o, umn, 1-lo)
    end

    scratch
end


"""
    ScratchQuickSort(next::Algorithm=SMALL_ALGORITHM) <: Algorithm
    ScratchQuickSort(lo::Union{Integer, Missing}, hi::Union{Integer, Missing}=lo, next::Algorithm=SMALL_ALGORITHM) <: Algorithm

Use the `ScratchQuickSort` algorithm with the `next` algorithm as a base case.

`ScratchQuickSort` is like `QuickSort`, but utilizes scratch space to operate faster and allow
for the possibility of maintaining stability.

If `lo` and `hi` are provided, finds and sorts the elements in the range `lo:hi`, reordering
but not necessarily sorting other elements in the process. If `lo` or `hi` is `missing`, it
is treated as the first or last index of the input, respectively.

`lo` and `hi` may be specified together as an `AbstractUnitRange`.

Characteristics:
  * *stable*: preserves the ordering of elements which compare equal
    (e.g. "a" and "A" in a sort of letters which ignores case).
  * *not in-place* in memory.
  * *divide-and-conquer*: sort strategy similar to [`QuickSort`](@ref).
  * *linear runtime* if `length(lo:hi)` is constant
  * *quadratic worst case runtime* in pathological cases
  (vanishingly rare for non-malicious input)
"""
struct ScratchQuickSort{L<:Union{Integer,Missing}, H<:Union{Integer,Missing}, T<:Algorithm} <: Algorithm
    lo::L
    hi::H
    next::T
end
ScratchQuickSort(next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(missing, missing, next)
ScratchQuickSort(lo::Union{Integer, Missing}, hi::Union{Integer, Missing}) = ScratchQuickSort(lo, hi, SMALL_ALGORITHM)
ScratchQuickSort(lo::Union{Integer, Missing}, next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(lo, lo, next)
ScratchQuickSort(r::OrdinalRange, next::Algorithm=SMALL_ALGORITHM) = ScratchQuickSort(first(r), last(r), next)

# select a pivot, partition v[lo:hi] according
# to the pivot, and store the result in t[lo:hi].
#
# sets `pivot_dest[pivot_index+pivot_index_offset] = pivot` and returns that index.
function partition!(t::AbstractVector, lo::Integer, hi::Integer, offset::Integer, o::Ordering,
        v::AbstractVector, rev::Bool, pivot_dest::AbstractVector, pivot_index_offset::Integer,
        rest_t, rest_v, rest_pivot_dest)
    # Ideally we would use `pivot_index = rand(lo:hi)`, but that requires Random.jl
    # and would mutate the global RNG in sorting.
    pivot_index = typeof(hi-lo)(hash(lo) % (hi-lo+1)) + lo
    @inbounds begin
        pivot = v[pivot_index]
        while lo < pivot_index
            x = v[lo]
            fx = rev ? !lt(o, x, pivot) : lt(o, pivot, x)
            t[(fx ? hi : lo) - offset] = x
            for irest in eachindex(rest_t)
                rest_t[irest][(fx ? hi : lo) - offset] = rest_v[irest][lo]
            end

            offset += fx
            lo += 1
        end
        while lo < hi
            x = v[lo+1]
            fx = rev ? lt(o, pivot, x) : !lt(o, x, pivot)
            t[(fx ? hi : lo) - offset] = x
            for irest in eachindex(rest_t)
                rest_t[irest][(fx ? hi : lo) - offset] = rest_v[irest][lo + 1]
            end

            offset += fx
            lo += 1
        end

        new_pivot_index = lo-offset + pivot_index_offset
        for irest in eachindex(rest_pivot_dest)
            rest_pivot_dest[irest][new_pivot_index] = rest_v[irest][pivot_index]
        end

        pivot_index = new_pivot_index
        pivot_dest[pivot_index] = pivot
    end

    # t_pivot_index = lo-offset (i.e. without pivot_index_offset)
    # t[t_pivot_index] is whatever it was before unless t is the pivot_dest
    # t[<t_pivot_index] <* pivot, stable
    # t[>t_pivot_index] >* pivot, reverse stable

    pivot_index
end

@inline function _sort!(v::AbstractVector, rest, a::ScratchQuickSort, o::Ordering, kw;
                t=nothing, offset=nothing, swap=false, rev=false, rest_scratch = nothing)
    @getkw lo hi scratch

    if t === nothing
        scratch, t = make_scratch(scratch, eltype(v), hi-lo+1)
        offset = 1-lo
        kw = (;kw..., scratch)
    end

    if rest_scratch === nothing
        rest_scratch = similar_rest(rest)
    end

    while lo < hi && hi - lo > SMALL_THRESHOLD
        j = if swap
            partition!(v, lo+offset, hi+offset, offset, o, t, rev, v, 0, rest, rest_scratch, rest)
        else
            partition!(t, lo, hi, -offset, o, v, rev, v, -offset, rest_scratch, rest, rest)
        end
        swap = !swap

        # For ScratchQuickSort(), a.lo === a.hi === missing, so the first two branches get skipped
        if !ismissing(a.lo) && j <= a.lo # Skip sorting the lower part
            if swap
                copyto!(v, lo, t, lo+offset, j-lo)
                for irest in eachindex(rest)
                    copyto!(rest[irest], lo, rest_scratch[irest], lo+offset, j-lo)
                end
            end
            if rev
                reverse!(v, lo, j-1)
                reverse_rest!(rest, lo, j-1)
            end

            lo = j+1
            rev = !rev
        elseif !ismissing(a.hi) && a.hi <= j # Skip sorting the upper part

            if swap
                copyto!(v, j+1, t, j+1+offset, hi-j)
                for irest in eachindex(rest)
                    copyto!(rest[irest], j+1, rest_scratch[irest], j+1+offset, hi-j)
                end
            end
            if rev
                reverse!(v, j+1, hi)
                reverse_rest!(rest, j+1, hi)
            end

            hi = j-1
        elseif j-lo < hi-j
            # Sort the lower part recursively because it is smaller. Recursing on the
            # smaller part guarantees O(log(n)) stack space even on pathological inputs.
            _sort!(v, rest, a, o, (;kw..., lo, hi=j-1); t, offset, swap, rev, rest_scratch)
            lo = j+1
            rev = !rev
        else # Sort the higher part recursively
            _sort!(v, rest, a, o, (;kw..., lo=j+1, hi); t, offset, swap, rev=!rev, rest_scratch)
            hi = j-1
        end
    end
    hi < lo && return scratch

    if swap
        copyto!(v, lo, t, lo + offset, hi-lo+1)
        for irest in eachindex(rest)
            copyto!(rest[irest], lo, rest_scratch[irest], lo+offset, hi-lo+1)
        end
    end
    if rev
        reverse!(v, lo, hi)
        reverse_rest!(rest, lo, hi)
    end

    _sort!(v, rest, a.next, o, (;kw..., lo, hi))
end


"""
    StableCheckSorted(next) <: Algorithm

Check if an input is sorted and/or reverse-sorted.

The definition of reverse-sorted is that for every pair of adjacent elements, the latter is
less than the former. This is stricter than `issorted(v, Reverse(o))` to avoid swapping pairs
of elements that compare equal.
"""
struct StableCheckSorted{T<:Algorithm} <: Algorithm
    next::T
end
@inline function _sort!(v::AbstractVector, rest, a::StableCheckSorted, o::Ordering, kw)
    @getkw lo hi scratch
    if _issorted(v, lo, hi, o)
        return scratch
    elseif _issorted(v, lo, hi, Lt((x, y) -> !lt(o, x, y)))
        # Reverse only if necessary. Using issorted(..., Reverse(o)) would violate stability.
        reverse!(v, lo, hi)
        reverse_rest!(rest, lo, hi)
        return scratch
    end

    _sort!(v, rest, a.next, o, kw)
end


# The return value indicates whether v is sorted (true) or t is sorted (false)
# This is one of the many reasons radix_sort! is not exported.
function radix_sort!(v::AbstractVector{U}, lo::Integer, hi::Integer, bits::Unsigned,
                     t::AbstractVector{U}, offset::Integer,rest,rest_ts,
                     chunk_size=radix_chunk_size_heuristic(lo, hi, bits)) where U <: Unsigned
    # bits is unsigned for performance reasons.
    counts = Vector{Int}(undef, 1 << chunk_size + 1) # TODO use scratch for this

    #println("In Radixsort main func")
    shift = 0
    while true
        @noinline radix_sort_pass!(t, lo, hi, offset, counts, v, shift, chunk_size, rest_ts,rest)
        # the latest data resides in t
        shift += chunk_size
        shift < bits || return false
        @noinline radix_sort_pass!(v, lo+offset, hi+offset, -offset, counts, t, shift, chunk_size, rest, rest_ts)
        # the latest data resides in v
        shift += chunk_size
        shift < bits || return true
    end
end
function radix_sort_pass!(t, lo, hi, offset, counts, v, shift, chunk_size, rest_ts, rest)
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
            for irest in eachindex(rest)
                #println("Type of ID $j, $k")
                rest_ts[irest][j] = rest[irest][k]
                # t_ = rest_ts[irest]
                # v_ = rest[irest]
                # t_[j] = v_[k]
            end
            counts[i] = j + 1         # increment the target index for the next
        end                           #  ↳ element in this bucket
    end
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

maybe_unsigned(x::Integer) = x # this is necessary to avoid calling unsigned on BigInt
maybe_unsigned(x::BitSigned) = unsigned(x)
function _issorted(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    @boundscheck checkbounds(v, lo:hi)
    @inbounds for i in (lo+1):hi
        lt(o, v[i], v[i-1]) && return false
    end
    true
end


## default sorting policy ##

"""
    InitialOptimizations(next) <: Algorithm

Attempt to apply a suite of low-cost optimizations to the input vector before sorting. These
optimizations may be automatically applied by the `sort!` family of functions when
`alg=InsertionSort`, `alg=MergeSort`, or `alg=QuickSort` is passed as an argument.

`InitialOptimizations` is an implementation detail and subject to change or removal in
future versions of Julia.

If `next` is stable, then `InitialOptimizations(next)` is also stable.

The specific optimizations attempted by `InitialOptimizations` are
[`MissingOptimization`](@ref), [`BoolOptimization`](@ref), dispatch to
[`InsertionSort`](@ref) for inputs with `length <= 10`, and [`IEEEFloatOptimization`](@ref).
"""
InitialOptimizations(next) = MissingOptimization(
    BoolOptimization(
        Small{10}(
            IEEEFloatOptimization(
                next))))
"""
    DEFAULT_STABLE

The default sorting algorithm.

This algorithm is guaranteed to be stable (i.e. it will not reorder elements that compare
equal). It makes an effort to be fast for most inputs.

The algorithms used by `DEFAULT_STABLE` are an implementation detail. See extended help
for the current dispatch system.

# Extended Help

`DEFAULT_STABLE` is composed of two parts: the [`InitialOptimizations`](@ref) and a hybrid
of Radix, Insertion, Counting, Quick sorts.

We begin with MissingOptimization because it has no runtime cost when it is not
triggered and can enable other optimizations to be applied later. For example,
BoolOptimization cannot apply to an `AbstractVector{Union{Missing, Bool}}`, but after
[`MissingOptimization`](@ref) is applied, that input will be converted into am
`AbstractVector{Bool}`.

We next apply [`BoolOptimization`](@ref) because it also has no runtime cost when it is not
triggered and when it is triggered, it is an incredibly efficient algorithm (sorting `Bool`s
is quite easy).

Next, we dispatch to [`InsertionSort`](@ref) for inputs with `length <= 10`. This dispatch
occurs before the [`IEEEFloatOptimization`](@ref) pass because the
[`IEEEFloatOptimization`](@ref)s are not beneficial for very small inputs.

To conclude the [`InitialOptimizations`](@ref), we apply [`IEEEFloatOptimization`](@ref).

After these optimizations, we branch on whether radix sort and related algorithms can be
applied to the input vector and ordering. We conduct this branch by testing if
`UIntMappable(v, order) !== nothing`. That is, we see if we know of a reversible mapping
from `eltype(v)` to `UInt` that preserves the ordering `order`. We perform this check after
the initial optimizations because they can change the input vector's type and ordering to
make them `UIntMappable`.

If the input is not [`UIntMappable`](@ref), then we perform a presorted check and dispatch
to [`ScratchQuickSort`](@ref).

Otherwise, we dispatch to [`InsertionSort`](@ref) for inputs with `length <= 40` and then
perform a presorted check ([`CheckSorted`](@ref)).

We check for short inputs before performing the presorted check to avoid the overhead of the
check for small inputs. Because the alternate dispatch is to [`InsertionSort`](@ref) which
has efficient `O(n)` runtime on presorted inputs, the check is not necessary for small
inputs.

We check if the input is reverse-sorted for long vectors (more than 500 elements) because
the check is essentially free unless the input is almost entirely reverse sorted.

Note that once the input is determined to be [`UIntMappable`](@ref), we know the order forms
a [total order](wikipedia.org/wiki/Total_order) over the inputs and so it is impossible to
perform an unstable sort because no two elements can compare equal unless they _are_ equal,
in which case switching them is undetectable. We utilize this fact to perform a more
aggressive reverse sorted check that will reverse the vector `[3, 2, 2, 1]`.

After these potential fast-paths are tried and failed, we [`ComputeExtrema`](@ref) of the
input. This computation has a fairly fast `O(n)` runtime, but we still try to delay it until
it is necessary.

Next, we [`ConsiderCountingSort`](@ref). If the range the input is small compared to its
length, we apply [`CountingSort`](@ref).

Next, we [`ConsiderRadixSort`](@ref). This is similar to the dispatch to counting sort,
but we conside rthe number of _bits_ in the range, rather than the range itself.
Consequently, we apply [`RadixSort`](@ref) for any reasonably long inputs that reach this
stage.

Finally, if the input has length less than 80, we dispatch to [`InsertionSort`](@ref) and
otherwise we dispatch to [`ScratchQuickSort`](@ref).
"""
const DEFAULT_STABLE = InitialOptimizations(
    IsUIntMappable(
        Small{40}(
            CheckSorted(
                ComputeExtrema(                             # SyncSort: Removed ConsiderCountingSort
                    ConsiderRadixSort(
                        Small{80}(
                            ScratchQuickSort()))))),
        StableCheckSorted(
            ScratchQuickSort())))
"""
    DEFAULT_UNSTABLE

An efficient sorting algorithm.

The algorithms used by `DEFAULT_UNSTABLE` are an implementation detail. They are currently
the same as those used by [`DEFAULT_STABLE`](@ref), but this is subject to change in future.
"""
const DEFAULT_UNSTABLE = DEFAULT_STABLE
const SMALL_THRESHOLD  = 20

function Base.show(io::IO, alg::Algorithm)
    print_tree(io, alg, 0)
end
function print_tree(io::IO, alg::Algorithm, cols::Int)
    print(io, "    "^cols)
    show_type(io, alg)
    print(io, '(')
    for (i, name) in enumerate(fieldnames(typeof(alg)))
        arg = getproperty(alg, name)
        i > 1 && print(io, ',')
        if arg isa Algorithm
            #println(io)
            print_tree(io, arg, cols+1)
        else
            i > 1 && print(io, ' ')
            print(io, arg)
        end
    end
    print(io, ')')
end
show_type(io::IO, alg::Algorithm) = Base.show_type_name(io, typeof(alg).name)
show_type(io::IO, alg::Small{N}) where N = print(io, "Base.Sort.Small{$N}")

defalg(v::AbstractArray) = DEFAULT_STABLE
defalg(v::AbstractArray{<:Union{Number, Missing}}) = DEFAULT_UNSTABLE
defalg(v::AbstractArray{Missing}) = DEFAULT_UNSTABLE # for method disambiguation
defalg(v::AbstractArray{Union{}}) = DEFAULT_UNSTABLE # for method disambiguation

"""
    sort!(v; alg::Algorithm=defalg(v), lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)

Sort the vector `v` in place. A stable algorithm is used by default. You can select a
specific algorithm to use via the `alg` keyword (see [Sorting Algorithms](@ref) for
available algorithms). The `by` keyword lets you provide a function that will be applied to
each element before comparison; the `lt` keyword allows providing a custom "less than"
function (note that for every `x` and `y`, only one of `lt(x,y)` and `lt(y,x)` can return
`true`); use `rev=true` to reverse the sorting order. `rev=true` preserves forward stability:
Elements that compare equal are not reversed. These options are independent and can
be used together in all possible combinations: if both `by` and `lt` are specified, the `lt`
function is applied to the result of the `by` function; `rev=true` reverses whatever
ordering specified via the `by` and `lt` keywords.

# Examples
```jldoctest
julia> v = [3, 1, 2]; sort!(v); v
3-element Vector{Int64}:
 1
 2
 3

julia> v = [3, 1, 2]; sort!(v, rev = true); v
3-element Vector{Int64}:
 3
 2
 1

julia> v = [(1, "c"), (3, "a"), (2, "b")]; sort!(v, by = x -> x[1]); v
3-element Vector{Tuple{Int64, String}}:
 (1, "c")
 (2, "b")
 (3, "a")

julia> v = [(1, "c"), (3, "a"), (2, "b")]; sort!(v, by = x -> x[2]); v
3-element Vector{Tuple{Int64, String}}:
 (3, "a")
 (2, "b")
 (1, "c")
```
"""
function syncsort!(v::AbstractVector{T}, rest...;
                   alg::Algorithm=defalg(v),
                   lt=isless,
                   by=identity,
                   rev::Union{Bool,Nothing}=nothing,
                   order::Ordering=Forward,
                   scratch::Union{Vector{T}, Nothing}=nothing) where T
    for r in rest
        @assert r isa AbstractVector
        @assert size(v,1) == size(r,1)
    end
    _sort!(v,  rest, maybe_apply_initial_optimizations(alg), ord(lt,by,rev,order), (;scratch),)
    v
end

"""
    sort(v; alg::Algorithm=defalg(v), lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)

Variant of [`sort!`](@ref) that returns a sorted copy of `v` leaving `v` itself unmodified.

Uses `Base.copymutable` to support immutable collections and iterables.

!!! compat "Julia 1.10"
    `sort` of arbitrary iterables requires at least Julia 1.10.

# Examples
```jldoctest
julia> v = [3, 1, 2];

julia> sort(v)
3-element Vector{Int64}:
 1
 2
 3

julia> v
3-element Vector{Int64}:
 3
 1
 2
```
"""
function sort(v; kws...)
    size = IteratorSize(v)
    size == HasShape{0}() && throw(ArgumentError("$v cannot be sorted"))
    size == IsInfinite() && throw(ArgumentError("infinite iterator $v cannot be sorted"))
    sort!(copymutable(v); kws...)
end
sort(v::AbstractVector; kws...) = sort!(copymutable(v); kws...) # for method disambiguation
sort(::AbstractString; kws...) =
    throw(ArgumentError("sort(::AbstractString) is not supported"))
sort(::Tuple; kws...) =
    throw(ArgumentError("sort(::Tuple) is only supported for NTuples"))

function sort(x::NTuple{N}; lt::Function=isless, by::Function=identity,
              rev::Union{Bool,Nothing}=nothing, order::Ordering=Forward) where N
    o = ord(lt,by,rev,order)
    if N > 9
        v = sort!(copymutable(x), DEFAULT_STABLE, o)
        tuple((v[i] for i in 1:N)...)
    else
        _sort(x, o)
    end
end
_sort(x::Union{NTuple{0}, NTuple{1}}, o::Ordering) = x
function _sort(x::NTuple, o::Ordering)
    a, b = Base.IteratorsMD.split(x, Val(length(x)>>1))
    merge(_sort(a, o), _sort(b, o), o)
end
merge(x::NTuple, y::NTuple{0}, o::Ordering) = x
merge(x::NTuple{0}, y::NTuple, o::Ordering) = y
merge(x::NTuple{0}, y::NTuple{0}, o::Ordering) = x # Method ambiguity
merge(x::NTuple, y::NTuple, o::Ordering) =
    (lt(o, y[1], x[1]) ? (y[1], merge(x, tail(y), o)...) : (x[1], merge(tail(x), y, o)...))


## uint mapping to allow radix sorting primitives other than UInts ##

"""
    UIntMappable(T::Type, order::Ordering)

Return `typeof(uint_map(x::T, order))` if [`uint_map`](@ref) and
[`uint_unmap`](@ref) are implemented.

If either is not implemented, return `nothing`.
"""
UIntMappable(T::Type, order::Ordering) = nothing

"""
    uint_map(x, order::Ordering)::Unsigned

Map `x` to an un unsigned integer, maintaining sort order.

The map should be reversible with [`uint_unmap`](@ref), so `isless(order, a, b)` must be
a linear ordering for `a, b <: typeof(x)`. Satisfies
`isless(order, a, b) === (uint_map(a, order) < uint_map(b, order))`
and `x === uint_unmap(typeof(x), uint_map(x, order), order)`

See also: [`UIntMappable`](@ref) [`uint_unmap`](@ref)
"""
function uint_map end

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

function uint_unmap!(v::AbstractVector, u::AbstractVector{U}, rest_v, rest_u,
                     lo::Integer, hi::Integer,
                     order::Ordering, offset::U=zero(U),
                     index_offset::Integer=0) where U <: Unsigned
    @inbounds for i in lo:hi
        v[i] = uint_unmap(eltype(v), u[i+index_offset]+offset, order)
        for irest in eachindex(rest_v)
            rest_v[irest][i] = rest_u[irest][i + index_offset]
        end
    end
    v
end



### Unused constructs for backward compatibility ###

## Old algorithms ##

struct QuickSortAlg     <: Algorithm end
struct MergeSortAlg     <: Algorithm end

"""
    PartialQuickSort{T <: Union{Integer,OrdinalRange}}

Indicate that a sorting function should use the partial quick sort
algorithm. Partial quick sort returns the smallest `k` elements sorted from smallest
to largest, finding them and sorting them using [`QuickSort`](@ref).

Characteristics:
  * *not stable*: does not preserve the ordering of elements which
    compare equal (e.g. "a" and "A" in a sort of letters which
    ignores case).
  * *in-place* in memory.
  * *divide-and-conquer*: sort strategy similar to [`MergeSort`](@ref).

  Note that `PartialQuickSort(k)` does not necessarily sort the whole array. For example,

```jldoctest
julia> x = rand(100);

julia> k = 50:100;

julia> s1 = sort(x; alg=QuickSort);

julia> s2 = sort(x; alg=PartialQuickSort(k));

julia> map(issorted, (s1, s2))
(true, false)

julia> map(x->issorted(x[k]), (s1, s2))
(true, true)

julia> s1[k] == s2[k]
true
```
"""
struct PartialQuickSort{T <: Union{Integer,OrdinalRange}} <: Algorithm
    k::T
end

"""
    QuickSort

Indicate that a sorting function should use the quick sort
algorithm, which is *not* stable.

Characteristics:
  * *not stable*: does not preserve the ordering of elements which
    compare equal (e.g. "a" and "A" in a sort of letters which
    ignores case).
  * *in-place* in memory.
  * *divide-and-conquer*: sort strategy similar to [`MergeSort`](@ref).
  * *good performance* for large collections.
"""
const QuickSort     = QuickSortAlg()

"""
    MergeSort

Indicate that a sorting function should use the merge sort
algorithm. Merge sort divides the collection into
subcollections and repeatedly merges them, sorting each
subcollection at each step, until the entire
collection has been recombined in sorted form.

Characteristics:
  * *stable*: preserves the ordering of elements which compare
    equal (e.g. "a" and "A" in a sort of letters which ignores
    case).
  * *not in-place* in memory.
  * *divide-and-conquer* sort strategy.
  * *good performance* for large collections but typically not quite as
    fast as [`QuickSort`](@ref).
"""
const MergeSort     = MergeSortAlg()

maybe_apply_initial_optimizations(alg::Algorithm) = alg
maybe_apply_initial_optimizations(alg::QuickSortAlg) = InitialOptimizations(alg)
maybe_apply_initial_optimizations(alg::MergeSortAlg) = InitialOptimizations(alg)
maybe_apply_initial_optimizations(alg::InsertionSortAlg) = InitialOptimizations(alg)

# selectpivot!
#
# Given 3 locations in an array (lo, mi, and hi), sort v[lo], v[mi], v[hi] and
# choose the middle value as a pivot
#
# Upon return, the pivot is in v[lo], and v[hi] is guaranteed to be
# greater than the pivot

@inline function selectpivot!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    @inbounds begin
        mi = midpoint(lo, hi)

        # sort v[mi] <= v[lo] <= v[hi] such that the pivot is immediately in place
        if lt(o, v[lo], v[mi])
            v[mi], v[lo] = v[lo], v[mi]
        end

        if lt(o, v[hi], v[lo])
            if lt(o, v[hi], v[mi])
                v[hi], v[lo], v[mi] = v[lo], v[mi], v[hi]
            else
                v[hi], v[lo] = v[lo], v[hi]
            end
        end

        # return the pivot
        return v[lo]
    end
end

# partition!
#
# select a pivot, and partition v according to the pivot

function partition!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
    pivot = selectpivot!(v, lo, hi, o)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while lt(o, v[i], pivot); i += 1; end;
        while lt(o, pivot, v[j]); j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
    end
    v[j], v[lo] = pivot, v[j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end

function sort!(v::AbstractVector, lo::Integer, hi::Integer, a::QuickSortAlg, o::Ordering)
    @inbounds while lo < hi
        hi-lo <= SMALL_THRESHOLD && return sort!(v, lo, hi, SMALL_ALGORITHM, o)
        j = partition!(v, lo, hi, o)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && sort!(v, lo, j-1, a, o)
            lo = j+1
        else
            j+1 < hi && sort!(v, j+1, hi, a, o)
            hi = j-1
        end
    end
    return v
end

sort!(v::AbstractVector{T}, lo::Integer, hi::Integer, a::MergeSortAlg, o::Ordering, t0::Vector{T}) where T =
    invoke(sort!, Tuple{typeof.((v, lo, hi, a, o))..., AbstractVector{T}}, v, lo, hi, a, o, t0) # For disambiguation
function sort!(v::AbstractVector{T}, lo::Integer, hi::Integer, a::MergeSortAlg, o::Ordering,
        t0::Union{AbstractVector{T}, Nothing}=nothing) where T
    @inbounds if lo < hi
        hi-lo <= SMALL_THRESHOLD && return sort!(v, lo, hi, SMALL_ALGORITHM, o)

        m = midpoint(lo, hi)

        t = t0 === nothing ? similar(v, m-lo+1) : t0
        length(t) < m-lo+1 && resize!(t, m-lo+1)
        Base.require_one_based_indexing(t)

        sort!(v, lo,  m,  a, o, t)
        sort!(v, m+1, hi, a, o, t)

        i, j = 1, lo
        while j <= m
            t[i] = v[j]
            i += 1
            j += 1
        end

        i, k = 1, lo
        while k < j <= hi
            if lt(o, v[j], t[i])
                v[k] = v[j]
                j += 1
            else
                v[k] = t[i]
                i += 1
            end
            k += 1
        end
        while k < j
            v[k] = t[i]
            k += 1
            i += 1
        end
    end

    return v
end

function sort!(v::AbstractVector, lo::Integer, hi::Integer, a::PartialQuickSort,
               o::Ordering)
    @inbounds while lo < hi
        hi-lo <= SMALL_THRESHOLD && return sort!(v, lo, hi, SMALL_ALGORITHM, o)
        j = partition!(v, lo, hi, o)

        if j <= first(a.k)
            lo = j+1
        elseif j >= last(a.k)
            hi = j-1
        else
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            if j-lo < hi-j
                lo < (j-1) && sort!(v, lo, j-1, a, o)
                lo = j+1
            else
                hi > (j+1) && sort!(v, j+1, hi, a, o)
                hi = j-1
            end
        end
    end
    return v
end

end # module Sort
