#!/usr/bin/env -S julia --project=./ --threads auto

# This is a script for running NASCTY.jl on the fixed key dataset.
using CUDA
CUDA.set_runtime_version!(; local_toolkit=true)
# using CUDA # for  NVIDIA CUDA Support.
using cuDNN
# using AMDGPU # for AMD GPU ROCM Support.
# using Metal # for Apple Metal GPU Support. (Experimental)
# using oneAPI # for Intel oneAPI GPU Support. (Experimental)
# using OpenCL # for OpenCL support. (Experimental)

import JLD2
import Flux

import NASCTY

const DEFAULT_DATA_PATH = joinpath(@__DIR__, "data", "ASCAD_data", "ASCAD_databases", "ASCAD.h5")

function printhelp()
    println("""
    NASCTY ASCAD Dataset program
    =============================

    Usage: julia $(@__FILE__) [OPTIONS] [DATASET_PATH]

    Arguments:
      DATASET_PATH    Path to ASCAD dataset file (HDF5 format)
                      Default: $DEFAULT_DATA_PATH

    Options:
      -h, --help      Show this help message

    Example:
      julia $(@__FILE__) /path/to/ASCAD_data.h5
      julia $(@__FILE__) # Uses default location

    Note: The ASCAD dataset should be in HDF5 format as described in:
    https://github.com/ANSSI-FR/ASCAD
    """)
end

function parsearguments(args::Vector{String})
    datapath = DEFAULT_DATA_PATH

    if isempty(args)
        @info "No path provided, using default location: $datapath"
    else
        for (_, arg) in enumerate(args)
            if arg == "--help" || arg == "-h"
                printhelp()
                exit(0)
            elseif !startswith(arg, "-")
                datapath = abspath(expanduser(arg))
            end
        end
    end

    return datapath
end


@info "This is a script for running NASCTY.jl on the fixed key ASCAD dataset."
datapath = parsearguments(ARGS)

@info "Loading the ASCAD datapoints from $(datapath)"
D_train, K_train, D_val, K_val, D_atk, K_atk = NASCTY.loadASCADsampledata(datapath)

@info "Setting NASCTY hyperparameters"
hyperparameters = NASCTY.NASCTYGENOMEHYPERPARAMETERS()

# (D_train, K_train) |> gpu, Transfers all training data to the GPU at once before creating the DataLoader.
# Only for datasets that fit into GPU memory. For larger datasets omit the `gpu` call.
# https://fluxml.ai/Flux.jl/stable/guide/gpu/
const BATCHSIZE = hyperparameters.batch_size |> first
training_data = (D_train, K_train) |> Flux.gpu
train_loader = Flux.DataLoader(training_data, batchsize = BATCHSIZE, shuffle = true)

validation_data = (D_val, K_val) |> Flux.gpu

@info "Starting neural architecture search..."
generations = NASCTY.nascty(hyperparameters, train_loader, validation_data)

@info "Finished neural architecture search..."
bestgenomes = generations .|> generation -> argmin(g -> g.loss, generation)
bestgenome = argmin(g -> g.loss, bestgenomes)
