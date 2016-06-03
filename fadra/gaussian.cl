__kernel void gaussian(__read_only image2d_t inputImage,
                            __read_only image2d_t filterImage,
                            __write_only image2d_t outputImage,
                            const int nInWidth,
                            const int nFilterWidth){

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    float4 sum = (float4)(0.0, 0.0, 0.0, 1.0);

    for(int r = 0; r < nFilterWidth; r++){
        for(int c = 0; c < nFilterWidth; c++){

            int2 location = (xOut + r, yOut + c);

            float4 filterVal = read_imagef(filterImage, sampler, location);
            float4 inputVal = read_imagef(inputImage, sampler, location);

            sum.x += filterVal.x * inputVal.x;
            sum.y += filterVal.y * inputVal.y;
            sum.z += filterVal.z * inputVal.z;
            sum.w = 1.0;
        }
    }

    int2 outLocation = (xOut, yOut);
    write_imagef(outputImage, outLocation, sum);
}