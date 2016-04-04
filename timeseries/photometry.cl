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
        output[x] = flat[x / n];
        printf("%d", flat[x / n]);
     }

     //output[0] += output[1] - output[3];


   //output[0] = sky_sum;
   //output[0] = (float)(sum - (sky_sum / sky_count)*px_count);

  /* barrier(CLK_LOCAL_MEM_FENCE);

   float2 center = (float2)(centerX, centerY);
   float sum = 0;
   int px_count = 0;
   float sky_sum = 0;
   int sky_count = 0;

   float2 curr_px = (float2)((x / n), (x % n));
   int dist = (int)fast_distance(center, curr_px);
   //printf("px: (%d, %d)\n", (x / n), (x % n));
   //printf("dist: %d\n", dist);
//   printf("cX: %d, cY: %d\n", centerX, centerY);

    if(dist < aperture){
    sum += (stamp[x] - dark[px*n + py])/flat[px*n + py];
    }else{
    sky_sum += (stamp[x] - dark[px*n + py])/flat[px*n + py];
    }

    output[x] = sum;

   /*for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float2 curr_px = (float2)(i, j);
            int dist = (int)fast_distance(center, curr_px);
            //output[x] = dist;
            //printf("dist: %f, aperture: %f\n", dist, aperture);
            if(dist < aperture){
                //printf("local_stamp: %f\n", local_stamp[i*n + j]);
                //printf("yas\n");
                sum += (local_stamp[ltid*n + j] - dark[i*n + j])/flat[i*n + j];
                px_count++;
            }
            else if(dist > sky_inner && dist < sky_outer){
                //printf("no\n");
                sky_sum += (local_stamp[ltid*n + j]-dark[i*n + j])/flat[i*n + j];
                sky_count++;
            }
        }
    }*/

    //printf("o = %f\n", sum);
    //output[x] = sum - (sky_sum / sky_count)*px_count;
    //output[x] = 25.;

}