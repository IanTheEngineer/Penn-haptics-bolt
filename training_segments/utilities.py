import numpy as np
import scipy.interpolate
import pylab
import tables
import scipy.ndimage

adjectives = ['sticky',
              'deformable',
              'hard',
              'hollow',
              'springy',
              'fuzzy',
              'rough',
              'thick',
              'compact',
              'elastic',
              'smooth',
              'metallic',
              'unpleasant',
              'plasticky',
              'meshy',
              'nice',
              'hairy',
              'compressible',
              'fibrous',
              'squishy',
              'gritty',
              'textured',
              'bumpy',
              'grainy',
              'scratchy',
              'cool',
              'absorbant',
              'stiff',
              'solid',
              'crinkly',
              'porous',
              #'warm',
              'slippery',
              'thin',
              #'sparse',
              'soft']
phases = ["SQUEEZE_SET_PRESSURE_SLOW", "HOLD_FOR_10_SECONDS", "SLIDE_5CM", "MOVE_DOWN_5CM"]
sensors = ["electrodes", "pac", "pdc", "tac"]

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def plot_database(database):
    """
    
    database: a dictionary as saved by aggregate_data
    """
    features = database.keys()
    sensors = database[features[0]].keys()
    
    rows = int(pylab.sqrt(len(sensors)))
    cols = len(sensors) / rows + 1
    
    #hold(True)
    for feature in features:
        pylab.figure()
        pylab.suptitle(feature, fontsize=12)
        for i, sensor in enumerate(sensors):
            pylab.subplot(rows, cols, i)
            data = database[feature][sensor]
            [pylab.plot(x[:,0]) for x in data]
            pylab.title(sensor)

def resample(a, dimensions, method='linear', center=False, minusone=False):
    """Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    
    -----------
    | http://www.scipy.org/Cookbook/Rebinning (Original source, 2011/11/19)
    """
    if a.ndim > 1:
        if dimensions[1] != a.shape[1]:
            raise ValueError("The new shape should keep the number of columns")
        
        ret = [_resample(col, (dimensions[0],), method, center, minusone)
                        for col in a.T]
        return np.array(ret).T
    else:
        return _resample(a, dimensions, method, center, minusone)
               

def _resample(a, dimensions, method='linear', center=False, minusone=False):
    orig_data = np.asarray(a)
    
    # Verify that number dimensions requested matches original shape
    if len(dimensions) != a.ndim:
        raise ValueError("Dimensions are not equal!")

    if not orig_data.dtype in [np.float64, np.float32]:
        orig_data = orig_data.astype(np.float64)

    dimensions = np.asarray(dimensions, dtype=np.float64)
    m1 = np.array(minusone, dtype=np.int64) # array(0) or array(1)
    offset = np.float64(center * 0.5) # float64(0.) or float64(0.5)

    # Resample data
    if method == 'neighbor':
        data = _resample_neighbor(orig_data, dimensions, offset, m1)
    elif method in ['nearest','linear']:
        data = _resample_nearest_linear(orig_data, dimensions, method,
                                        offset, m1)
    elif method == 'spline':
        data = _resample_spline(orig_data, dimensions, offset, m1)
    else:
        raise ValueError("Unknown sampling method")
    
    return data

def _resample_nearest_linear(orig, dimensions, method, offset, m1):
    """Resample using either linear or nearest interpolation"""


    dimlist = []
    
    # calculate new dims
    for i in range(orig.ndim):
        base = np.arange(dimensions[i])
        dimlist.append((orig.shape[i] - m1) / (dimensions[i] - m1) *
                       (base + offset) - offset)

    # specify old coordinates
    old_coords = [np.arange(i, dtype=np.float) for i in orig.shape]

    # first interpolation - for ndims = any
    
    mint = scipy.interpolate.interp1d(old_coords[-1], orig, kind=method)
    new_data = mint(dimlist[-1])

    trorder = [orig.ndim - 1] + range(orig.ndim - 1)
    for i in xrange(orig.ndim - 2, -1, -1):
        new_data = new_data.transpose(trorder)

        mint = scipy.interpolate.interp1d(old_coords[i], new_data,
                                          kind=method)
        new_data = mint(dimlist[i])

    if orig.ndim > 1:
        # need one more transpose to return to original dimensions
        new_data = new_data.transpose(trorder)

    return new_data

def _resample_neighbor(orig, dimensions, offset, m1):
    """Resample using closest-value interpolation"""
    dimlist = []
    
    for i in xrange(orig.ndim):
        base = np.indices(dimensions)[i]
        dimlist.append((orig.shape[i] - m1) / (dimensions[i] - m1) *
                       (base + offset) - offset)
    cd = np.array(dimlist).round().astype(int)
    
    return orig[list(cd)]

def _resample_spline(orig, dimensions, offset, m1):
    """Resample using spline-based interpolation"""
    
    oslices = [slice(0, j) for j in orig.shape]
    old_coords = np.ogrid[oslices] #pylint: disable=W0612
    nslices = [slice(0, j) for j in list(dimensions)]
    newcoords = np.mgrid[nslices]

    newcoords_dims = range(np.rank(newcoords))
    
    #make first index last
    newcoords_dims.append(newcoords_dims.pop(0))
    newcoords_tr = newcoords.transpose(newcoords_dims) #pylint: disable=W0612

    # makes a view that affects newcoords
    newcoords_tr += offset

    deltas = (np.asarray(orig.shape) - m1) / (dimensions - m1)
    newcoords_tr *= deltas

    newcoords_tr -= offset

    return scipy.ndimage.map_coordinates(orig, newcoords)

def dict_from_h5_group(group, phases, sensors):
    """
    Creates a dictionary from an h5 group. The dictionary will have fields:
    name: the name of the object
    adjectives: a list of strings
    data: a dictionary with keys being the phases, and values:
          one dictionary for each sensor, where the key is the sensor and the
          value is the data in the sensor
    """
    assert isinstance(group, tables.Group)
    ret_d = dict()
    ret_d["adjectives"] = group.adjectives[:]
    ret_d["name"] = group._v_name
    data = dict()
    ret_d["data"] = data
    for phase in phases:
        phase_data = {}
        for sensor in sensors:
            
            #getting the indexes for the phase
            indexed = (group.state.controller_detail_state.read() == phase)
            
            #finger 0
            finger_0 = group.biotacs.finger_0
            data_0 = getattr(finger_0, sensor).read()
            nrows = data_0.shape[0]
            data_0 = data_0.reshape((nrows,-1))
            data_0 = data_0[indexed, :]

            #finger_1
            finger_1 = group.biotacs.finger_1
            data_1 = getattr(finger_1, sensor).read()
            nrows = data_1.shape[0]
            data_1 = data_1.reshape((nrows,-1))
            data_1 = data_1[indexed, :]        
            
            phase_data[sensor] = np.hstack((data_0, data_1))
        data[phase] = phase_data
    
    return ret_d
            
def iterator_over_object_groups(database):
    """Returns an iterator over all the objects (groups) in the h5 database.
    If database is a string it will be interpreted as a filename, otherwise
    as an open pytables file.
    """
    if type(database) is str:
        database = tables.openFile(database,"r")
    
    return (g for g in database.root._v_children.values()
                   if g._v_name != "adjectives")
    