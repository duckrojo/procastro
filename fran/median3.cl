#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void s (int a, int b){
    int tmp;
    if (a > b){
        tmp = b;
        b = a;
        a = tmp;}
}

#define min3(a, b, c) s(a, b); s(a, c);
#define max3(a, b, c) s(b, c); s(a, c);
#define minmax3(a, b, c) max3(a, b, c); s(a, b);
#define minmax4(a, b, c, d) s(a, b); s(c, d); s(a, c); s(b, d);
#define minmax5(a, b, c, d, e) s(a, b); s(c, d); min3(a, c, e); max3(b, d, e);
#define minmax6(a, b, c, d, e, f) s(a, d); s(b, e); s(c, f); min3(a, b, c); max3(d, e, f);

__kernel void median3(__read_only image2d_t inputImage,
                        __write_only image2d_t output, int i_dim, int j_dim){

    int j = (int)get_global_id(0);
    int i = (int)get_global_id(1);
    int a0, a1, a2, a3, a4, a5;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int2 coord = (int2)(j-1, i-1);
    a0 = read_imagef(inputImage, sampler, coord).x;

    coord = (int2)(j, i-1);
    a1 = read_imagef(inputImage, sampler, coord).x;

    coord = (int2)(j+1, i-1);
    a2 = read_imagef(inputImage, sampler, coord).x;

    coord = (int2)(j-1, i);
    a3 = read_imagef(inputImage, sampler, coord).x;

    coord = (int2)(j, i);
    a4 = read_imagef(inputImage, sampler, coord).x;

    coord = (int2)(j+1, i);
    a5 = read_imagef(inputImage, sampler, coord).x;

    minmax6(a0, a1, a2, a3, a4, a5);
    coord = (int2)(j-1, i+1);
    a5 = read_imagef(inputImage, sampler, coord).x;

    minmax5(a1, a2, a3, a4, a5);
    coord = (int2)(j, i+1);
    a5 = read_imagef(inputImage, sampler, coord).x;

    minmax4(a2, a3, a4, a5);
    coord = (int2)(j+1, i+1);
    a5 = read_imagef(inputImage, sampler, coord).x;

    minmax3(a3, a4, a5);

    output = a4;
}