import numpy as np
# from conv_func import im2col, col2im

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
    return input_cols.reshape(batch_size * height_blocks * width_blocks, -1)

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
    padded_input = np.pad(input, npad, "constant", constant_values=0)

    input_cols = im2col(padded_input, kernel_size)
    output_channel = W.shape[0]
    W_cols = W.reshape(output_channel, -1) # o_c x (in_c * k_s * k_s)
    output_cols = np.dot(input_cols, W_cols.T) + b # (h_b * w_b * b_s) x o_c

    batch_size, channel, height, width = padded_input.shape
    height_blocks = height-kernel_size + 1
    width_blocks = width-kernel_size + 1
    output = col2im(output_cols, height_blocks, width_blocks, batch_size, output_channel)
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
        grad_W: gradient of W, shape = c_out (#output channel) x c_in x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
    go_pad = ((0,0), (0,0), (kernel_size-1,kernel_size-1), (kernel_size-1,kernel_size-1))
    padded_grad_output = np.pad(grad_output,go_pad, "constant", constant_values=0)
    in_pad = ((0,0), (0,0), (pad,pad), (pad,pad))
    padded_input = np.pad(input, in_pad, "constant", constant_values=0)

    grad_output_cols = im2col(padded_grad_output, kernel_size) # (h_b * w_b * b_s) x (out_c * k_s * k_s)
    
    output_channel = W.shape[0]
    input_channel = W.shape[1]
    W_rot = np.rot90(W.transpose(2,3,0,1), 2).transpose(2,3,0,1)
    W_cols = W_rot.transpose(1,0,2,3).reshape(input_channel, -1) # (in_c) x (out_c*k_s*k_s)
    grad_input_cols = np.dot(grad_output_cols, W_cols.T)

    batch_size, input_channel, height, width = padded_input.shape
    height_blocks = height - kernel_size + 1
    width_blocks = width - kernel_size + 1
    grad_padded_input = col2im(grad_input_cols, height, width, batch_size, input_channel)

    grad_output_tran = grad_output.transpose(1,0,2,3)
    grad_output_cols = grad_output_tran.reshape(output_channel, -1) # out_c x (h_b * w_b * n)

    input_cols = im2col(padded_input.transpose(1,0,2,3), height_blocks) # (in_c*ks*ks) x (hb*wb*n)
    grad_W_cols = np.dot(input_cols, grad_output_cols.T)
    grad_W = col2im(grad_W_cols, kernel_size, kernel_size, input_channel, output_channel).transpose(1,0,2,3)


    grad_b = np.sum(np.sum(np.sum(grad_output_tran, axis=1), axis=1), axis=1)

    if(pad == 0):
        return grad_padded_input, grad_W, grad_b
    else:
        grad_input = grad_padded_input[ : , : , pad:-pad, pad:-pad]
        return grad_input, grad_W, grad_b

def im2col_1c(A, kernel_size):
    '''
    Args:
        A: shape = n (#sample) x c_in (#input channel) x h_in (#height + 2*pad) x w_in (#width + 2*pad)

    Returns:
        input_cols: shape = (h_blocks * w_blocks * batch_size) x (channel)
            where h_out, w_out is the height and width of output, after convolution
    '''
    batch_size, channel, height, width = A.shape
    s3, s2, s1, s0 = A.strides
    height_blocks = height / kernel_size
    width_blocks = width / kernel_size
    shp = batch_size, channel, height_blocks, width_blocks, kernel_size, kernel_size
    strd = s3, s2, s1 * kernel_size, s0 * kernel_size, s1, s0

    input_cols = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return input_cols.reshape(batch_size * channel * height_blocks * width_blocks, -1)


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
    npad = ((0,0),(0,0),(pad,pad),(pad,pad))
    padded_input = np.pad(input,npad, "constant", constant_values=0)
    batch_size, channel, height, width = padded_input.shape
    height_blocks = height / kernel_size
    width_blocks = width / kernel_size

    input_cols = im2col_1c(padded_input, kernel_size)
    output_cols = np.mean(input_cols, axis=1)

    s1 = output_cols.strides[0]
    shp = batch_size, channel, height_blocks, width_blocks
    strd = (height_blocks*width_blocks*channel)*s1, (height_blocks*width_blocks)*s1, width_blocks*s1, s1

    output = np.lib.stride_tricks.as_strided(output_cols, shape=shp, strides=strd)

    return output


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
    batch_size, channel, height, width = grad_output.shape
    s3, s2, s1, s0 = grad_output.strides
    shp = batch_size, channel, kernel_size * height, kernel_size * width
    strd = s3, s2, int(s1/kernel_size), int(s0/kernel_size)

    grad_input = np.repeat(np.repeat(grad_output.transpose(2,3,0,1), kernel_size, axis=1), kernel_size, axis=0).transpose(2,3,0,1) / (kernel_size * kernel_size)
    if(pad == 0):
        return grad_input
    else:
        return grad_input[:,:,pad:-pad, pad:-pad]
