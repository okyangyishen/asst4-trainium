import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # TILING
    TILE_CIN = 128
    n_tiles_c_in = in_channels // TILE_CIN
    TILE_COUT = 128
    n_tiles_c_out = out_channels // TILE_COUT
    TILE_HW = 512
    if input_height == 32:
        TILE_H = 32
        TILE_W = TILE_HW // TILE_H
        n_tiles_h = input_height // TILE_H
        n_tiles_w = input_width // TILE_W
    else:
        TILE_H = 4
        TILE_W = TILE_HW // TILE_H
        n_tiles_h = math.ceil(input_height / TILE_H)
        n_tiles_w = math.ceil(input_width / TILE_W)

    X_re = X.reshape((batch_size, n_tiles_c_in, TILE_CIN, input_height, input_width))
    # X_re = X.reshape((batch_size, TILE_CIN, n_tiles_c_in, input_height, input_width)) # switched 1th and 2nd dimension for loading X_rowchunk at one time
    bias_re = bias.reshape((n_tiles_c_out, TILE_COUT))
    bias_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(TILE_COUT), 1), dtype=bias.dtype, buffer=nl.sbuf)
    for c_out_tile in nl.affine_range(n_tiles_c_out):
        bias_sbuf[c_out_tile] = nl.load(bias_re[c_out_tile])

    W = W.reshape((n_tiles_c_out, TILE_COUT, n_tiles_c_in, TILE_CIN, filter_height, filter_width))
    
    weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(TILE_COUT), n_tiles_c_in, TILE_CIN, filter_height, filter_width), dtype = W.dtype, buffer = nl.sbuf)
    weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(TILE_COUT), TILE_CIN), dtype=W.dtype, buffer=nl.sbuf)
    w_sbuf = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(TILE_CIN), TILE_COUT), dtype=W.dtype, buffer=nl.sbuf)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        weight_sbuf[c_out_tile] = nl.load(W[c_out_tile])

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype = W.dtype)
                    w_sbuf[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])
    

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for c_out_tile in nl.affine_range(n_tiles_c_out):
            # Convolution on each row-chunk of the image (handle large image loading issue)
            for h_tile in nl.affine_range(n_tiles_h):                
                X_res = nl.ndarray(
                    shape=(nl.par_dim(TILE_COUT), TILE_H, n_tiles_w * TILE_W), 
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                num_input_rows = TILE_H + filter_height - 1

                x_idx = h_tile * TILE_H + nl.arange(num_input_rows)[None, :, None]
                mask = (x_idx < input_height)
                
                X_rowchunk = nl.zeros((n_tiles_c_in, nl.par_dim(TILE_CIN), TILE_H+filter_height-1, n_tiles_w*TILE_W + filter_width - 1), dtype=X.dtype, buffer=nl.sbuf)
                for c_in in nl.affine_range(n_tiles_c_in):
                    X_rowchunk[c_in, :, :, :input_width] = nl.load(X_re[b, c_in, :, h_tile*TILE_H : h_tile*TILE_H + num_input_rows], mask=mask)
                    #    FYI: X_re = X.reshape([batch_size, n_tiles_c_in, c_in_pmax, input_height, input_width])

                for w_tile in nl.affine_range(n_tiles_w):
                    res_sbuf = nl.ndarray(shape=(nl.par_dim(TILE_COUT), TILE_H, TILE_W), dtype=nl.float32, buffer=nl.sbuf)
                    for out_row_tile in nl.affine_range(TILE_H):
                        res_psum = nl.ndarray(shape=(nl.par_dim(TILE_COUT), TILE_W), dtype=nl.float32, buffer=nl.psum)
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                for c_in_tile in nl.affine_range(n_tiles_c_in):
                                    lhsT_tile = w_sbuf[i, j, c_out_tile, c_in_tile]
                                    rhs_tile = X_rowchunk[c_in_tile, :, out_row_tile + i, w_tile*TILE_W + j:(w_tile+1)*TILE_W + j]
                                    res_psum += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)
                        res_sbuf[:, out_row_tile, :] = nl.copy(res_psum, dtype=nl.float32)
                    res_sbuf = nisa.tensor_scalar(res_sbuf, np.add, bias_sbuf[c_out_tile])
                    X_res[:, :, w_tile * TILE_W:(w_tile+1) * TILE_W] = res_sbuf

                # fused maxpool
                maxpool_height = TILE_H // pool_size
                if pool_size == 1:
                    maxpool_res = X_res[:, :, :out_width]
                elif pool_size == 2:
                    X_reshaped = X_res.reshape(
                        nl.par_dim(TILE_COUT),
                        TILE_H // pool_size,
                        pool_size,
                        TILE_W // pool_size,
                        pool_size
                    )
                    maxpool_res = nl.max(X_reshaped, axis=(2, 4))
                
                maxpool_out_height = min(maxpool_height, out_pool_height - h_tile * maxpool_height)
                mask = (nl.arange(maxpool_height)[None, :, None] < maxpool_out_height)
                nl.store(X_out[b, c_out_tile * TILE_COUT:(c_out_tile + 1) * TILE_COUT, h_tile * maxpool_height:(h_tile + 1) * maxpool_height, :], 
                         value=maxpool_res, mask=mask)

    return X_out