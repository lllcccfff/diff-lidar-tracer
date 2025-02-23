/**
 * @file config.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef OPTIX_TRACER_CONFIG_H_INCLUDED
#define OPTIX_TRACER_CONFIG_H_INCLUDED

// Some global parameters
#define CHUNK_SIZE 16  // Chunk size for one traversal
#define STEP_EPSILON 0.00001
#define MAX_INTERSECT 1000  // Usually around 200

#define RGB_OFFSET 0 // 3
#define DEPTH_OFFSET 3 // 1
#define INTENSITY_OFFSET 4 // 1
#define RAYHIT_OFFSET 5 // 1
#define RAYDROP_OFFSET 6 // 1
#define ACCUM_OFFSET 7 // 1
#define NORMAL_OFFSET 8  // 3
#define DEEPEST_OFFSET 11 // 1
#define DISTORTION_OFFSET 12 // 1
#define FINALT_OFFSET 13 // 1
#define DISDEPTH_OFFSET 14 // 1
#define DISDEPTH2_OFFSET 15 // 1
#define MIDDEPTH_OFFSET 16 // 1
#define NUM_CHANNELS_F 17// Default 3, RGB

#define N_CONTRIB_OFFSET 0
#define MID_CONTRIB_OFFSET 1
#define M_BUFFER 2
#define NUM_CHANNELS_I 2 + MAX_INTERSECT


#define BLOCK_X 16
#define BLOCK_Y 16

#endif
