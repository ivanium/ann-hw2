import numpy as np


def im2col(A, kernel_size):
    '''
    Args:
        A: shape = n (#sample) x c_in (#input channel) x h_in (#height + 2*pad) x w_in (#width + 2*pad)

    Returns:
        input_cols: shape = (h_blocks * w_blocks * batch_size) x (channel)
            where h_out, w_out is the height and width of output, after convolution
    '''
    batch_size, channel, height, width = A.shape
    s3, s2, s1, s0 = A.strides
    height_blocks = height-kernel_size + 1
    width_blocks = width-kernel_size + 1
    shp = batch_size, height_blocks, width_blocks, channel, kernel_size,kernel_size
    strd = s3, s1, s0, s2, s1, s0

    input_cols = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return input_cols.reshape(height_blocks * width_blocks, -1)

def col2im(A, height_blocks, width_blocks, batch_size, output_channel):
    s1, s0 = A.strides
    shp = batch_size, output_channel, height_blocks, width_blocks
    strd = (height_blocks*width_blocks)*s1, s0, width_blocks*s1, s1

    output = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return output

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    npad = ((0,0),(0,0),(pad,pad),(pad,pad))
    padded_input = np.pad(input,npad, "constant", constant_values=0)

    input_cols = im2col(padded_input, kernel_size)
    output_channel = W.shape[0]
    W_cols = W.reshape(output_channel, -1) # o_c x (in_c * k_s * k_s)
    output_cols = input_cols * W_cols.T + b # (h_b * w_b * b_s) x o_c

    batch_size, channel, height, width = padded_input.shape
    height_blocks = height-kernel_size + 1
    width_blocks = width-kernel_size + 1
    output = col2im(output_cols, height_blocks, width_blocks, batch_size,output_channel)
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
    pass


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    pass


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    pass
