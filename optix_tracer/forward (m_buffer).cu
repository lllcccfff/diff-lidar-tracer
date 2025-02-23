/**
 * @file forward.cu
 * @author xbillowy
 * @brief 
 * @version 0.1
 * @date 2024-08-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#define OPTIXU_MATH_DEFINE_IN_NAMESPACE

#include <optix.h>
#include <math_constants.h>

#include "params.h"
#include "auxiliary.h"

// Make the parameters available to the device code
extern "C" {
    __constant__ Params params;
}

// Unpack two 32-bit payload from a 64-bit pointer
static __forceinline__ __device__
void *unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}
// Pack a 64-bit pointer from two 32-bit payload
static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}
// Get the payload pointer
template<typename T>
static __forceinline__ __device__ T *getPayload() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

// Call optixTrace to trace a single ray
__device__ void traceStep(float3 ray_o, float3 ray_d, uint32_t payload_u0, uint32_t payload_u1)
{
    optixTrace(
        params.handle,
        ray_o,
        ray_d,
        0.0f,  // Min intersection distance
        1e16,  // Max intersection distance
        0.0f,  // rayTime, used for motion blur, disable
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0,  // SBT offset
        0,  // SBT stride
        0,  // missSBTIndex
        payload_u0, payload_u1);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	return glm::max(result, 0.0f);
}


// Compute a 2D-to-2D mapping matrix from world to splat space,
// given a 2D gaussian parameters
__device__ void compute_transmat_uv(
	const glm::vec3 p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* viewmatrix,
    const float3 xyz,
	glm::mat3x4& world2splat,
	float3& normal,
    float2& uv
) {
    // Convert the quaternion and scale vector to matrices
    // * NOTE: R here is the row-major rotation matrix, namely R as in Python,
    // * NOTE: the original quat_to_rotmat(rot) will return the column-major R^T
    // * NOTE: S here is the inverse of the scale matrix
	glm::mat3 R = quat_to_rotmat_transpose(rot);
	glm::mat3 S = scale_to_mat_inverse(scale, mod);
	glm::mat3 L = S * R;
    glm::vec3 T = -1.f * L * p_orig;

	// Compute the world to splat transformation matrix
	world2splat = glm::mat3x4(
        glm::vec4(L[0].x, L[1].x, L[2].x, T.x),
        glm::vec4(L[0].y, L[1].y, L[2].y, T.y),
        glm::vec4(L[0].z, L[1].z, L[2].z, T.z)
	);

    // Compute the normal in world space
	normal = make_float3(L[0].z, L[1].z, L[2].z);

    // Convert the intersection point from world to splat space
    glm::vec3 uv1 = glm::vec4(glm::vec3(xyz.x, xyz.y, xyz.z), 1.0f) * world2splat;
    uv = make_float2(uv1.x, uv1.y);
}


__device__ bool compute_transmat_xy(
	const glm::vec3 p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const int W,
	const int H,
	float2& xy
) {
    // Convert the quaternion and scale vector to matrices
	glm::mat3 R = quat_to_rotmat(rot);  // R
	glm::mat3 S = scale_to_mat(scale, mod);  // S
	glm::mat3 L = R * S;

    // The splat2world matrix
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);
    // The world2ndc matrix
	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[ 8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[ 9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);
    // The ndc2pix matrix
	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

    glm::mat3 T;
    // Compute the final transformation matrix from splat space to pixel space
    // NOTE: The matrix multiplication is performed from right to left
	T = glm::transpose(splat2world) * world2ndc * ndc2pix;  // (3, 4) * (4, 4) * (4, 3) -> (3, 3)

    // Computing the projected center of each 2D Gaussian
    // The projected center of the 2DGS is used to create a low pass filter
	float3 T0 = {T[0][0], T[0][1], T[0][2]};
	float3 T1 = {T[1][0], T[1][1], T[1][2]};
	float3 T3 = {T[2][0], T[2][1], T[2][2]};
	// Compute AABB
	float3 f = {1.0f, 1.0f, -1.0f};
	float a = sumf3(T3 * T3 * f);
    if (a == 0.0) return false;

    float3 c = f / a;
    // Compute the projected center as the center of the AABB
	xy = {sumf3(c * T0 * T3), sumf3(c * T1 * T3)};
    return true;
}


// TODO: throw error if chunk size exceeds
// Core __raygen__ program
extern "C" __global__ void __raygen__ot()
{
    // Lookup current location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t tidx = idx.x * dim.y + idx.y;
    // Fetch the ray origin and direction of the current pixel
    float3 ray_om = params.ray_o[tidx];
    float3 ray_dm = params.ray_d[tidx];
    // Store a copy of the original ray origin and direction
    float3 ray_oc = ray_om;
    float3 ray_dc = ray_dm;
    // bool flag=false;
    // if (idx.x == 25 && idx.y == 1000)
    // {
    //     flag = true;
    // }

    // Creat and initialize the ray payload data
    RayPayload payload;
    IntersectionInfo buffer[CHUNK_SIZE];
    for (int i = 0; i < CHUNK_SIZE; i++) buffer[i].tmx = 1e16f;
    payload.buffer = buffer;
    payload.dpt = 0.f;
    payload.cnt = 0.f;

    // Pack the pointer, the values we store the payload pointer in
    uint32_t payload_u0, payload_u1;
    packPointer(&payload, payload_u0, payload_u1);

    // Initialize the volume rendering data
	float3 N = make_float3(0.0f, 0.0f, 0.0f);
    float C[3] = {0.0f};
    float D = 0.0f;
	float W = 0.0f;
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
    float T = 1.0f;
    float test_T = 1.0f;
    float last_dpt = 1e16f;
    int* m_buffer = params.out_attr_uint32 + NUM_CHANNELS_I * tidx + M_BUFFER;

	int contributor = 0;
	int median_contributor = -1;


    // Prepare rendering data
    float dpt = 0.0f;
    float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
    glm::mat3x4 world2splat = glm::mat3x4(0.0f);
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);
    float2 uv = make_float2(0.0f, 0.0f);
    // float2 xy;
    int last_gidx = -1;

    while (1)
    {
        // Actual optixTrace
        traceStep(ray_om, ray_dm, payload_u0, payload_u1);
        // Volume rendering

        for (int i = 0; i < CHUNK_SIZE; i++)
        {
            // Break if the intersection depth is invalid
            if (i >= payload.cnt)
                break;

            // Get the primitive index and Gaussian index
            int pidx = payload.buffer[i].idx;  // intersection primitive index
            int gidx = pidx / 2;  // Gaussian index is half of the primitive index
            // Compute the actual intersection depth and coordinates in world space
            dpt = payload.buffer[i].tmx + payload.dpt;
            xyz = ray_oc + dpt * ray_dc;

            // // Re-initialize payload data
            payload.buffer[i].tmx = 1e16f;
            payload.buffer[i].idx = 0;
            if (gidx == last_gidx)
            {
                continue;
            }
            last_gidx = gidx;
            // last_d = dpt;
            // last_xyz = xyz;

            // Build the world to splat transformation matrix
            // and compute the normal vector
            compute_transmat_uv(params.means3D[gidx], params.scales[gidx],
                                params.scale_modifier, params.rotations[gidx], params.viewmatrix,
                                xyz, world2splat, normal, uv);

            // Adjust the normal vector direction
#if DUAL_VISIABLE
            float3 dir = make_float3(params.means3D[gidx].x - ray_oc.x, params.means3D[gidx].y - ray_oc.y, params.means3D[gidx].z - ray_oc.z);
            // float3 dir = ray_dm;
            float cos = -sumf3(dir * normal);
            if (cos == 0) continue;
            normal = cos > 0 ? normal : -normal;
#endif

            // // Compute the projected center of each 2D Gaussian
            // bool ok = compute_xy(params.means3D[gidx], params.scales[gidx],
            //                      params.scale_modifier, params.rotations[gidx],
            //                      params.projmatrix, params.W, params.H, xy);
            // if (!ok) continue;
            // // TODO (xbillowy): Determine the use of Eq. (11) of 2DGS
            // float rho2d = FilterInvSquare * ((xy.x - idx.x) * (xy.x - idx.x) + (xy.y - idx.y) * (xy.y - idx.y));

            // Get weights
            float rho3d = uv.x * uv.x + uv.y * uv.y;
            float rho2d = rho3d;
            float power = -0.5f * min(rho3d, rho2d);
            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
            float alpha = min(0.99f, params.opacities[gidx] * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;

            test_T = T * (1 - alpha);
            if (test_T < 0.001f)
            {
                break;
            }
            float w = alpha * T;

            // Render colors
            if (params.colors_precomp == nullptr)
            {
                glm::vec3 result = computeColorFromSH(gidx, params.D, params.M,
                                                      params.means3D, *params.campos,
                                                      params.shs);
                C[0] += w * result.x;
                C[1] += w * result.y;
                C[2] += w * result.z;
            }
            else
            {
                for (int ch = 0; ch < 3; ch++)
                    C[ch] += w * params.colors_precomp[ch];
            }
            
			if (T > 0.5) {
				median_depth = dpt;
				// median_weight = w;
				median_contributor = gidx;
			}
            // Render other componments
            atomicAdd(params.accum_gaussian_weights + gidx, w);
            N += w * normal;
            D += w * dpt;
            W += w;
            float A = 1-T;
			float m = dpt;
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			M1 += m * w;
			M2 += m * m * w;

            // Update transmittence
            T = test_T;

            // Keep track of the current position and last range entry to update this pixel
            m_buffer[contributor] = gidx;
			contributor++;
            // if (flag)
            //     printf("gidx: %d\n", gidx);

        }

        if (test_T < 0.001f || payload.cnt < CHUNK_SIZE)
        {
            break;
        }

        // Re-initialize payload data
        payload.dpt = dpt;
        payload.cnt = 0;
        // Update Ray origin
        ray_om = ray_oc + (payload.dpt + STEP_EPSILON) * ray_dc;
    }

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
    for (int ch = 0; ch < 3; ch++)
        params.out_attr_float32[NUM_CHANNELS_F * tidx + RGB_OFFSET + ch] = 0;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + DEPTH_OFFSET] = D;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + INTENSITY_OFFSET] = C[0] + T * params.background[0];
    params.out_attr_float32[NUM_CHANNELS_F * tidx + RAYHIT_OFFSET] = C[1] + T * params.background[1];
    params.out_attr_float32[NUM_CHANNELS_F * tidx + RAYDROP_OFFSET] = C[2] + T * params.background[2];
    params.out_attr_float32[NUM_CHANNELS_F * tidx + ACCUM_OFFSET] = W;

    params.out_attr_float32[NUM_CHANNELS_F * tidx + NORMAL_OFFSET + 0] = N.x;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + NORMAL_OFFSET + 1] = N.y;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + NORMAL_OFFSET + 2] = N.z;
    
    params.out_attr_float32[NUM_CHANNELS_F * tidx + MIDDEPTH_OFFSET] = median_depth;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + DISTORTION_OFFSET] = distortion;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + FINALT_OFFSET] = T;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + DISDEPTH_OFFSET] = M1;
    params.out_attr_float32[NUM_CHANNELS_F * tidx + DISDEPTH2_OFFSET] = M2;
    
    params.out_attr_uint32[NUM_CHANNELS_I * tidx + MID_CONTRIB_OFFSET] = median_contributor;
    params.out_attr_uint32[NUM_CHANNELS_I * tidx + N_CONTRIB_OFFSET] = contributor;

}


// Core __anyhit__ program
extern "C" __global__ void __anyhit__ot()
{
    // https://forums.developer.nvidia.com/t/some-confusion-on-anyhit-shader-in-optix/223336
    // Get the payload pointer
    RayPayload &payload = *getPayload<RayPayload>();

    // Get the intersection tmax and the primitive index
    float tmx = optixGetRayTmax();
    uint32_t idx = optixGetPrimitiveIndex();

    // Increment the number of intersections
    if (tmx < payload.buffer[CHUNK_SIZE - 1].tmx)
    {
        // Enter this branch means current intersection is closer, we need to update the buffer
        // Increment the counter, the counter only increases when the intersection is closer
        payload.cnt += 1;

        // Temporary variable for swapping
        float tmp_tmx;
        float cur_tmx = tmx;
        uint32_t tmp_idx;
        uint32_t cur_idx = idx;

        // Insert the new primitive into the ascending t sorted list
        for (int i = 0; i < CHUNK_SIZE; ++i)
        {
            // Swap if the new intersection is closer
            if (payload.buffer[i].tmx > cur_tmx)
            {
                // Store the original buffer info
                tmp_tmx = payload.buffer[i].tmx;
                tmp_idx = payload.buffer[i].idx;
                // Update the current intersection info
                payload.buffer[i].tmx = cur_tmx;
                payload.buffer[i].idx = cur_idx;
                // Swap
                cur_tmx = tmp_tmx;
                cur_idx = tmp_idx;
            }
        }
    }

    // Ignore the intersection to continue traversal
    optixIgnoreIntersection();
}
