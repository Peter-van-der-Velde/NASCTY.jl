import JLD2
import MLUtils
import OneHotArrays

"""
	splittraces(D_X::Matrix, D_y⃗::Vector, N::Integer, Q::Integer = 0)

Split the dataset into a training set and a validation set.

Arguments:
- `D_X::Matrix`: Matrix containing the input traces.
- `D_y⃗::Vector`: Vector containing the corresponding labels.
- `N::Integer`: Size of the training set.
- `Q::Integer = 0`: Size of the validation set. If not provided, the entire dataset beyond the training set is used for validation.

Returns:
- Tuple `(D_train, K_train, D_val, K_val)` containing the training set traces `D_train`, training set labels `K_train`, validation set traces `D_val`, and validation set labels `K_val`.
"""
function splittraces(D_X::Matrix, D_y⃗::Vector, N::Integer, Q::Integer = 0)
	lenD = (D_X |> size)[2]

    @assert N <= lenD "Training set size N ($N) cannot be larger than the dataset D $((D_X |> size)[2])."
	D_train = D_X[ : , 1:N]
	K_train = D_y⃗[1:N]

	@assert (N+Q) <= lenD "Validation set size N ($N) cannot be larger than the remaining datapoints ($((D_X |> size)[2] - N))."
	Q = Q == 0 ? lenD - N : Q

	D_val = D_X[ : , N+1:N+Q]
	K_val = D_y⃗[N+1:N+Q]

	(D_train, K_train, D_val, K_val)
end

"""
    reshapetraces(X::Matrix)

Reshapes a 2D matrix `X` into a 3D array where each column of `X` becomes a slice along the second dimension of the resulting array.

# Arguments
- `X::Matrix`: The input 2D matrix to reshape.

# Returns
A 3D array where each column of the input matrix `X` becomes a slice along the second dimension of the resulting array.

# Examples
```julia
julia> X = [1 2 3; 4 5 6]
2×3 Matrix{Int64}:
 1  2  3
 4  5  6

julia> reshaphtraces(X)
2×1×3 Array{Int64, 3}:
[:, :, 1] =
 1
 4
[:, :, 2] =
 2
 5

[:, :, 3] =
 3
 6
```
"""
reshapetraces(X) =
	X |>
	  size |>
	  △X -> reshape(X, (△X[1], 1, △X[2]))

extrema2(X) = mapreduce(x -> (x,x), ((min1, max1), (min2, max2)) -> (min(min1, min2), max(max1, max2)), X)

"""
	minmaxscale(x⃗::Vector, minv::Number=0, maxv::Number=1)

Scales a vector `x⃗` to the range [`minv`, `maxv`], using min-max normalisation.

The min-max normalisation algorithm is defined as:

``f(x⃗) = \frag{x⃗ - min(x⃗)}{max(x⃗) - min(x⃗)} * (maxv - minv)) + minv``


# Arguments
- `x⃗::Vector`: The input vector to be scaled.
- `minv::Number=0`: The minimum value of the desired range. Default is 0.
- `maxv::Number=1`: The maximum value of the desired range. Default is 1.

# Returns
A vector with values scaled to the specified range.

# Example
```julia
x = [1, 2, 3, 4, 5]
scaled_x = minmaxscale(x, -1, 1)
println(scaled_x)  # Output: [-1.0, -0.5, 0.0, 0.5, 1.0]
```
"""
# function minmaxscale44(x⃗, minv::Float32=0.0f0, maxv::Float32=1.0f0)::Vector{Float32}
# 	xsmin, xsmax = x⃗ |> extrema2 .|> Float32

#     (x⃗ .- xsmin) .* ((xsmax - xsmin) / (maxv - minv)) .+ minv
# end
function minmaxscale(x⃗::AbstractVector, minv::Float32=0.0f0, maxv::Float32=1.0f0)::Vector{Float32}
	xs_min, xs_max = x⃗ |> extrema2 .|> Float32

    range = xs_max - xs_min
    return @. Float32(minv + (x⃗ - xs_min) / range * (maxv - minv))
end



"""
	minmaxscale!(x⃗::Vector, minv::Number=0, maxv::Number=1)

Scales a vector `x⃗` to be between `minv` and `maxv` using min-max normalisation.

The min-max normalisation algorithm is defined as:

``f(x⃗) = \frag{x⃗ - min(x⃗)}{max(x⃗) - min(x⃗)} * (maxv - minv)) + minv``


# Arguments
- `x⃗::Vector`: The input vector to be scaled.
- `minv::Number=0`: The minimum value of the desired range. Default is 0.
- `maxv::Number=1`: The maximum value of the desired range. Default is 1.

# Returns
A vector with values scaled to the specified range.

# Example
```julia
x = [1, 2, 3, 4, 5]
scaled_x = minmaxscale!(x, -1, 1)
println(scaled_x)  # Output: [-1.0, -0.5, 0.0, 0.5, 1.0]
```
"""
function minmaxscale!(x⃗, minv::Float32=0.0f0, maxv::Float32=1.0f0)::Vector{Float32}
	xsmin, xsmax = x⃗ |> extrema2

    range = (xsmax - xsmin) / (maxv - minv)
    x⃗ .= xsmin .+ (x⃗ ) .* range .+ minv
end

"""
	minmaxscaleM(X::Matrix, min::Number=0.0, max::Number=1.0)

Applies min-max normalisation to each column of a matrix X and scales it
to be between min and max.

# Arguments
- `X`::Matrix: The input matrix where each column is treated as a separate vector to be scaled.
- `min`::Number=0.0: The minimum value of the desired range for the scaled columns. Default is 0.0.
- `max`::Number=1.0: The maximum value of the desired range for the scaled columns. Default is 1.0.

# Returns
A matrix with columns scaled to the specified range.
Example

```julia
X = [1 4; 2 5; 3 6]
scaled_X = minmaxscaleM(X, -1.0, 1.0)
println(scaled_X)  # Output: [-1.0 -1.0; 0.0 0.0; 1.0 1.0]
```
"""
function minmaxscale(X::Matrix, min::Float32=0.0f0, max::Float32=1.0f0)
    return mapslices(row -> minmaxscale(row, -1.0f0, 1.0f0), X, dims=2)
end

function minmaxscale2(X::Matrix, min::Float32=0.0f0, max::Float32=1.0f0)
    x_rows, _ = size(X)

    res = Float32.(X')

    @inbounds for i in 1:x_rows
        @views minmaxscale!(res[:, i], min, max)
    end

    return res'
end


"""
undersamplewithremainder(n_samples::Integer, xs, ys, shuffle::Bool=false)

Perform class-balanced undersampling while preserving leftover samples.

This function creates a class-balanced dataset by sampling an equal number of instances
from each class. It returns both the undersampled data and the remaining data that
wasn't selected, making it useful for creating balanced training/validation splits.

Based in part on the code found in MLUtils with added

src: https://github.com/JuliaML/MLUtils.jl/blob/main/src/resample.jl

# Arguments
- `n_samples::Integer`: Total number of samples in the output dataset. Must be a multiple
  of the number of classes.
- `xs`: Feature data. Should support `obsview` indexing.
- `ys`: Label data. Should support `MLUtils.group_indices`.
- `shuffle::Bool=false`: If `true`, shuffle the selected indices; otherwise keep them
  sorted.

# Returns
A tuple of four elements:
1. `xs_balanced`: Feature data with balanced class distribution (via `obsview`)
2. `ys_balanced`: Label data with balanced class distribution (via `obsview`)
3. `xs_remainder`: Feature data not selected in the balanced sample (via `obsview`)
4. `ys_remainder`: Label data not selected in the balanced sample (via `obsview`)
"""
function undersamplewithremainder(n_samples::Integer, xs, ys, shuffle::Bool=false)
	lm = MLUtils.group_indices(ys)
	mincount = minimum(length, values(lm))

	n_classes = lm |> length

	@assert n_samples % n_classes == 0 "Number of samples ($(n_samples)) is not a multiple of the amount of classes $(n_classes)"

	n_samples_per_class = Integer(n_samples / n_classes)

	@assert n_classes * mincount >= n_samples "Number of possible samples lower than input is able to deliver"

	inds = Int[]
	rem_inds = Int[]

	for (_, inds_for_lbl) in lm
		append!(inds, inds_for_lbl[1:n_samples_per_class])
		append!(rem_inds, inds_for_lbl[n_samples_per_class+1:length(inds_for_lbl)])
	end

	shuffle ? MLUtils.shuffle!(inds) : sort!(inds)

    xs_balanced = MLUtils.obsview(xs, inds)
    ys_balanced = MLUtils.obsview(ys, inds)
    xs_remainder = MLUtils.obsview(xs, rem_inds)
    ys_remainder = MLUtils.obsview(ys, rem_inds)

	return xs_balanced, ys_balanced, xs_remainder, ys_remainder
end

"""
    loadASCADsampledata(datapath)

Load ASCAD dataset from HDF5 file with preprocessing.

First it loads ASCAD traces/labels.
Then it scales them to [-1, 1], then it creates a balanced training/validation splits (35,584/3,840 samples).
Lastly, it reshapes them for neural network input.

# Arguments
- `datapath::String`: Path to JLD2 file containing ASCAD data with the expected
  structure (Profiling_traces/traces, Profiling_traces/labels, etc.)

# Returns
A tuple containing:
- `D_train`: Training traces, balanced and reshaped (Float32 Array)
- `K_train`: Training labels for balanced set
- `D_val`: Validation traces, balanced and reshaped (Float32 Array)
- `K_val`: Validation labels for balanced set
- `D_atk`: Attack traces, reshaped (Float32 Array)
- `K_atk`: Attack labels
"""
function loadASCADsampledata(datapath)
    TRAININGSIZE = 35584
    VALIDATIONSIZE = 3840

    f = JLD2.jldopen(datapath, "r")
    D = read(f, "Profiling_traces/traces")
    K = read(f, "Profiling_traces/labels")
    D_atk   = read(f, "Attack_traces/traces")
    K_atk   = read(f, "Attack_traces/labels")
    close(f)

    # scale the data from -1 to 1.
	# This will reduce the amount of required neural network parameters.
	D_mm = Float32.(minmaxscale(D, -1.0f0, 1.0f0))
	D_atk = Float32.(minmaxscale(D_atk, -1.0f0, 1.0f0))

	# balance the incoming data
	D_train, K_train, rX, rys = undersamplewithremainder(TRAININGSIZE, D_mm, K, true) |> collect
	D_val, K_val, _, _ = undersamplewithremainder(VALIDATIONSIZE, rX, rys, true) |> collect

	# Reshape traces for the neural networks.
	D_train = reshapetraces(D_train) |> collect
	D_val = reshapetraces(D_val) |> collect
	D_atk = reshapetraces(D_atk) |> collect

    c = 0:255
    K_train_ohb = OneHotArrays.onehotbatch(K_train, c)
    K_val_ohb = OneHotArrays.onehotbatch(K_val, c)
    K_atk_ohb = OneHotArrays.onehotbatch(K_atk, c)

	return (D_train, K_train_ohb, D_val, K_val_ohb, D_atk, K_atk_ohb)
end
