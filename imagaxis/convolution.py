from numpy import *
import fourier


def basic_conv(a, b, indices_list, axes, circular_list, op='...,...', izero=None):
    '''
    a and b must be the same size
    the length of each axis must be even
    
    indices_list specifies the way in which to convovle for the corresponding axes in 'axes'
    the repeated index is summed over
    the single index is the remaining index after convolution
    for example, indices = ['x,y-x'] will perform the sum over x of a(x)*b(y-x)
    
    options for each entry of indices_list is one of four forms (you can change the letter names):
    'x,y-x' or 'x,x+y' or 'y-x,x'  or 'x+y,x' 
    
    every convolution can be put into one of these forms by relabeling the index which is summed over
    note that 'x,x-y' is not in one of these forms but after the relabeling x->x+y it becomes 'x+y,x'
    
    axes specify the axes over which convolutions occur
    
    circular_list (True or False) specifies whether to do a circular or a regular convolution
    
    op specifies the tensor operation which to perform to the intermediate ffts
    it is a string which becomes the first argument to numpy.einsum
    by default it is elementwise multiplication
    
    izero is the index of x=0 in the arrays a and b
    '''
    
    if izero is None:
        izero = len(a)//2
    
    a_ = a[:]
    b_ = b[:]
    
    dims = array([min(d1,d2) for d1,d2 in zip(shape(a), shape(b))])
    #maxdims = array([max(d1,d2) for d1,d2 in zip(shape(a), shape(b))])
    
    for indices, axis, circular in zip(indices_list, axes, circular_list):
        if circular: continue
            
        comma = indices.index(',')
        
        a_ = concatenate((a_, zeros(2*dims-array(shape(a)))), axis=axis)
        b_ = concatenate((b_, zeros(2*dims-array(shape(b)))), axis=axis) 
    
    for axis in axes:
        a_ = roll(a_, -izero, axis=axis)
        b_ = roll(b_, -izero, axis=axis)
        
    a_ = fft.fftn(a_, axes=axes)
    b_ = fft.fftn(b_, axes=axes)
        
    for indices, axis in zip(indices_list, axes):
        if '+' in indices:
            comma = indices.index(',')
            if comma==1:
                a_ = roll(flip(a_, axis=axis), 1, axis=axis)
            elif comma==3:
                b_ = roll(flip(b_, axis=axis), 1, axis=axis)
            else: raise ValueError
            
    x = fft.ifftn(einsum(op, a_, b_), axes=axes)
    for axis in axes:
        x = roll(x, izero, axis=axis)
    
    return x



def basic_conv0(a, b, indices_list, axes, circular_list, op='...,...'):
    '''
    todo: write optional zeros argument
      - if x is the repeated index, then yz-xz=0 and xz=0 and yz=0
    
    a and b must be the same size
    the length of each axis must be even
    
    indices_list specifies the way in which to convovle for the corresponding axes in 'axes'
    the repeated index is summed over
    the single index is the remaining index after convolution
    for example, indices = ['x,y-x'] will perform the sum over x of a(x)*b(y-x)
    
    options for each entry of indices_list is one of four forms (you can change the letter names):
    'x,y-x' or 'x,x+y' or 'y-x,x'  or 'x+y,x' 
    
    every convolution can be put into one of these forms by relabeling the index which is summed over
    note that 'x,x-y' is not in one of these forms but after the relabeling x->x+y it becomes 'x+y,x'
    
    axes specify the axes over which convolutions occur
    
    circular_list (True or False) specifies whether to do a circular or a regular convolution
    
    op specifies the tensor operation which to perform to the intermediate ffts
    it is a string which becomes the first argument to numpy.einsum
    by default it is elementwise multiplication
    '''
    
    a_ = a[:]
    b_ = b[:]
    
    dims = array([min(d1,d2) for d1,d2 in zip(shape(a), shape(b))])
    #maxdims = array([max(d1,d2) for d1,d2 in zip(shape(a), shape(b))])
    
    for indices, axis, circular in zip(indices_list, axes, circular_list):
        if circular: continue
            
        comma = indices.index(',')
        
        a_ = concatenate((a_, zeros(2*dims-array(shape(a)))), axis=axis)
        b_ = concatenate((b_, zeros(2*dims-array(shape(b)))), axis=axis) 
    
    for axis in axes:
        a_ = roll(a_, shape(a)[axis]//2, axis=axis)
        b_ = roll(b_, shape(b)[axis]//2, axis=axis)
        
    a_ = fft.fftn(a_, axes=axes)
    b_ = fft.fftn(b_, axes=axes)
        
    for indices, axis in zip(indices_list, axes):
        if '+' in indices:
            comma = indices.index(',')
            if comma==1:
                a_ = roll(flip(a_, axis=axis), 1, axis=axis)
            elif comma==3:
                b_ = roll(flip(b_, axis=axis), 1, axis=axis)
            else: raise ValueError
            
    x = fft.ifftn(einsum(op, a_, b_), axes=axes)
    for axis in axes:
        x = roll(x, shape(a)[axis]//2, axis=axis)
    
    return x


def conv(a, b, indices_list, axes, circular_list, beta=None, kinds=(None,None,None), op='...,...', jumps=(None,None)):
    '''
    
    todo: write optional zeros argument
      - if x is the repeated index, then yz-xz=0 and xz=0 and yz=0
    
    a and b must be the same size
    the length of each axis must be even
    
    indices_list specifies the way in which to convovle for the corresponding axes in 'axes'
    the repeated index is summed over
    the single index is the remaining index after convolution
    for example, indices = ['x,y-x'] will perform the sum over x of a(x)*b(y-x)
    
    options for each entry of indices_list is one of four forms (you can change the letter names):
    'x,y-x' or 'x,x+y' or 'y-x,x'  or 'x+y,x' 
    
    every convolution can be put into one of these forms by relabeling the index which is summed over
    note that 'x,x-y' is not in one of these forms but after the relabeling x->x+y it becomes 'x+y,x'
    
    axes specify the axes over which convolutions occur
    
    circular_list specifies whether to do a circular or a regular convolution for each axis

    kinds = (kind of a, kind of b, kind of out). Either 'fermion' or 'boson'    

    op specifies the tensor operation which to perform to the ffts
    it is a string which becomes the first argument to numpy.einsum
    by default it is elementwise multiplication

    '''

    for indices, axis, circular in zip(indices_list, axes, circular_list):
        if circular:
            a = fft.fft(a, axis=axis)
            b = fft.fft(b, axis=axis)
        else:
            a = fourier.w2t(a, beta, axis, kind=kinds[0], jump=jumps[0])
            b = fourier.w2t(b, beta, axis, kind=kinds[1], jump=jumps[1])


        comma = indices.index(',')
                    
        if '+' in indices:
            if comma==1:
                a = flip(a, axis=axis)
                if not circular and kinds[0]=='fermion': a *= -1.0
                if circular: a = roll(a, 1, axis=axis)
            elif comma==3:
                b = flip(b, axis=axis)
                if not circular and kinds[1]=='fermion': b *= -1.0
                if circular: b = roll(b, 1, axis=axis)
            else:
                raise ValueError            
            
    x = einsum(op, a, b)
    for axis, circular in zip(axes, circular_list):
        if circular:
            x = fft.ifft(x, axis=axis)
        else:
            x = fourier.t2w(x, beta, axis, kind=kinds[2])[0]
            
    for axis, circular in zip(axes, circular_list):
        if circular:
            x = roll(x, shape(a)[axis]//2, axis=axis)
    
    return x
