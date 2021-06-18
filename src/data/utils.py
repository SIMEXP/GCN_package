import numpy.lib.format

# https://stackoverflow.com/questions/64226337/is-there-a-way-to-read-npy-header-without-loading-the-whole-file
def read_npy_array_header(filepath):
    with open(filepath, 'rb') as fobj:
      version = numpy.lib.format.read_magic(fobj)
      func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
      func = getattr(numpy.lib.format, func_name)
      header = func(fobj)
    
    return header