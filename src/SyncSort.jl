# File   : SyncSort.jl
# License: MIT
# Author : Dominik Werner <d.wer2@gmx.de>
# Date   : 26.01.2023
# Paper  : https://drive.google.com/file/d/0B7uLFueU4vLfcjJfZFh3TlIxMFE/view?resourcekey=0-8Ovsx4PtAJn78xLboBEb_g



module SyncSort

export syncsort!



include("cpu_sort.jl")
using .CPUSort

end # module MergeSortGPU
