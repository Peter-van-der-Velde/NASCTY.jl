"""
    take(x⃗::AbstractArray, n::Number)

Returns the first `n` elements of the array `x⃗`.
If `n` is greater than the length of `x⃗`, the entire array is returned.

# Arguments
- `x⃗`: The input array.
- `n`: The number of elements to take.

# Returns
A new array containing the first `n` elements of `x⃗`.
"""
take(x⃗::AbstractArray, n::Number) = x⃗ |> length |>
	xl -> n <= xl ? x⃗[1:n] : x⃗[1:xl]

"""
	skip(x⃗::AbstractArray, n::Number)

Skips the first `n` elements of an array `x⃗`.

# Arguments
  - `x⃗`: The input array.
  - `n`: The number of elements to skip.

# Returns
    An array containing all elements of x⃗ except the first n elements.
	If n is greater than or equal to the length of x⃗, an empty array is returned.
"""
skip(x⃗::AbstractArray, n::Number) = x⃗ |> length |>
	xl -> n <= xl ? x⃗[n+1:end] : x⃗ |> empty
