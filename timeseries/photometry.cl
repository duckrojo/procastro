__kernel void photometry(__global float* stamp, __constant float* dark,
                        __constant float* flat, __global float* output,
                        __local float* local_stamp)
{
    int x = get_global_id(0);
    int px = x / n;
    int py = x % n;

    int s = n * n;
    int lid = get_local_id(0);

    float sum = 0;
    int px_count = 0;
    float sky_sum = 0;
    int sky_count = 0;

    if(x < s){
        //local_stamp[lid] = stamp[x];
        float2 curr_px = (float2)((x / n), (x % n));
        float2 center = (float2)(centerX, centerY);
        int dist = (int)fast_distance(center, curr_px);
        if(dist < aperture){
            output[0] += stamp[x]-dark[x];
            output[1]++;
            }else if (dist > sky_inner && dist < sky_outer){
            output[2] += stamp[x]-dark[x];
            output[3]++;
            }
     }
}