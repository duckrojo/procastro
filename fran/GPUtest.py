__author__ = 'fran'

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sys
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

platforms = cl.get_platforms()
if len(platforms) == 0:
    print("Failed to find any OpenCL platforms.")

devices = platforms[0].get_devices(cl.device_type.GPU)
if len(devices) == 0:
    print("Could not find GPU device, trying CPU...")
    devices = platforms[0].get_devices(cl.device_type.CPU)
    if len(devices) == 0:
        print("Could not find OpenCL GPU or CPU device.")

ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)
MF = cl.mem_flags

# Host variables
img = np.arange(100, dtype='int32').reshape(10, 10)
#img_g = cl.image_from_array(ctx, img, 1)
#img_g = cl_array.Array(queue, img.shape, dtype='int32')
img_g = cl_array.to_device(queue, img)
#img_g = cl_array.set(img)
img_g2 = cl.image_from_array(ctx, img, 1)

mask = np.zeros((3, 3), dtype='int32')
mask[1, 1] = 1
#mask_g = cl.image_from_array(ctx, mask, 1)
mask_g = cl_array.Array(queue, mask.shape, dtype='int32')

out = np.arange(100, dtype='int32').reshape(10, 10)
#out_g = cl.image_from_array(ctx, out, 1)
out_g = cl_array.Array(queue, out.shape, dtype='int32')

# Device variable allocation
#img_g = cl.Buffer(ctx, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=img)
#mask_g = cl.Buffer(ctx, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=mask)
#out_g = cl.Buffer(ctx, MF.WRITE_ONLY, img.nbytes)

f = open('gaussian.cl', 'r')
programName = "".join(f.readlines())

program = cl.Program(ctx, programName).build()
program.gaussian(queue, img.shape, None, img_g2, mask_g, out_g, np.int32(10), np.int32(2))

print(out_g)
out2 = np.empty_like(out)
#cl.enqueue_copy(queue, out2, out_g).wait()
out2 = out_g.get()