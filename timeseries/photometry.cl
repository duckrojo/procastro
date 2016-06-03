#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable

__kernel void photometry(__global float* stamp, __global float* dark,
                        __global float* flat, __global int* output)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int i = y*SIZE + x;

    int s = n * n;
    int px = x / n;
    int py = x % n;

    float2 curr_px = (float2)(px, py);
    float2 center = (float2)(centerX, centerY);
    int dist = (int)distance(center, curr_px);

    if(dist < aperture){
        atomic_add(&output[0], (stamp[x]-dark[x]));
        atomic_add(&output[1], 1);
    }else if (dist > sky_inner && dist < sky_outer){
        atomic_add(&output[2], (stamp[x]-dark[x]));
        atomic_add(&output[3], 1);
    }
}
