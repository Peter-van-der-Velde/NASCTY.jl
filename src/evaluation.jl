"""
    G(p⃗, c)

Given a probability vector `p⃗` in which each element directly relates to a
classifier class in vector `c`, calculates the output key guessing vector `g⃗`
ordered from most likely to least.

# Arguments
- `p⃗::Vector`: Probability vector representing the likelihoods of each class.
- `c::Vector`: Vector  of classifier classes.

# Returns
- `g⃗::Vector`: Output key guessing vector ordered from most likely to least.

# Example
```julia
p⃗ = [0.2, 0.5, 0.3]
c = ['A', 'B', 'C']
G(p⃗, c)  # Output: ['B', 'C', 'A']

"""
G(p⃗, c⃗)::Vector =
	zip(p⃗, c⃗) |> collect |>
    ts -> sort(ts, by=first, rev=true) |>
          tgr -> map(t -> t[2], tgr)



"""
	accuracy(r⃗::Vector, threshold = 1)

Calculates the accuracy of a side-channel attack given a vector r⃗ of key ranks.

# Arguments
- `r⃗::Vector``: Vector of key ranks.
- `threshold::Integer`: Threshold for considering a key rank guess as successful. Defaults to 1.

# Returns
    Float64: Accuracy of the side-channel attack.

# Example
julia
```
r⃗ = [2, 1, 3]accuracy(r⃗)  # Output: 33.34
```
"""
accuracy(r⃗::Vector, threshold = 1) =
	filter(r -> r <=threshold, r⃗) |>
    sum |>
    n_correct -> n_correct / length(r⃗)
