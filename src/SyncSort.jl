# File   : SyncSort.jl
# License: MIT
# Author : Dominik Werner <d.wer2@gmx.de>
# Date   : 26.01.2023
# Paper  : https://drive.google.com/file/d/0B7uLFueU4vLfcjJfZFh3TlIxMFE/view?resourcekey=0-8Ovsx4PtAJn78xLboBEb_g



module SyncSort
using Metal

using KernelAbstractions
using LoopVectorization
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
end # module MergeSortGPU
