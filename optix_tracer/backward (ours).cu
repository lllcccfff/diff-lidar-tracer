/**
 * @file backward.cu
 * @author xbillowy
 * @brief 
 * @version 0.1
 * @date 2024-08-26
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
__device__ glm::vec3 computeColorFromSHForward(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, float* clamped)
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

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[0] = (result.x < 0);
	clamped[1] = result.y;
	clamped[2] = result.z;
	return glm::vec3(max(result.x, 0.0f), result.y, result.z);

}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian
__device__ void computeColorFromSHBackward(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const float* clamped, const float* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	// glm::vec3 dL_dRGB = dL_dcolor[idx];
	glm::vec3 dL_dRGB = glm::vec3(dL_dcolor[0], dL_dcolor[1], dL_dcolor[2]);
	dL_dRGB.x *= clamped[0] ? 0 : 1;
	// dL_dRGB.y *= clamped[1] ? 0 : 1;
	// dL_dRGB.z *= clamped[2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	glm::vec3 dL_dsh0 = dRGBdsh0 * dL_dRGB;
	atomicAdd(&(dL_dsh[0].x), dL_dsh0.x);
	atomicAdd(&(dL_dsh[0].y), dL_dsh0.y);
	atomicAdd(&(dL_dsh[0].z), dL_dsh0.z);
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		glm::vec3 dL_dsh1 = dRGBdsh1 * dL_dRGB;
		glm::vec3 dL_dsh2 = dRGBdsh2 * dL_dRGB;
		glm::vec3 dL_dsh3 = dRGBdsh3 * dL_dRGB;
		atomicAdd(&(dL_dsh[1].x), dL_dsh1.x);
		atomicAdd(&(dL_dsh[1].y), dL_dsh1.y);
		atomicAdd(&(dL_dsh[1].z), dL_dsh1.z);
		atomicAdd(&(dL_dsh[2].x), dL_dsh2.x);
		atomicAdd(&(dL_dsh[2].y), dL_dsh2.y);
		atomicAdd(&(dL_dsh[2].z), dL_dsh2.z);
		atomicAdd(&(dL_dsh[3].x), dL_dsh3.x);
		atomicAdd(&(dL_dsh[3].y), dL_dsh3.y);
		atomicAdd(&(dL_dsh[3].z), dL_dsh3.z);

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			glm::vec3 dL_dsh4 = dRGBdsh4 * dL_dRGB;
			glm::vec3 dL_dsh5 = dRGBdsh5 * dL_dRGB;
			glm::vec3 dL_dsh6 = dRGBdsh6 * dL_dRGB;
			glm::vec3 dL_dsh7 = dRGBdsh7 * dL_dRGB;
			glm::vec3 dL_dsh8 = dRGBdsh8 * dL_dRGB;
			atomicAdd(&(dL_dsh[4].x), dL_dsh4.x);
			atomicAdd(&(dL_dsh[4].y), dL_dsh4.y);
			atomicAdd(&(dL_dsh[4].z), dL_dsh4.z);
			atomicAdd(&(dL_dsh[5].x), dL_dsh5.x);
			atomicAdd(&(dL_dsh[5].y), dL_dsh5.y);
			atomicAdd(&(dL_dsh[5].z), dL_dsh5.z);
			atomicAdd(&(dL_dsh[6].x), dL_dsh6.x);
			atomicAdd(&(dL_dsh[6].y), dL_dsh6.y);
			atomicAdd(&(dL_dsh[6].z), dL_dsh6.z);
			atomicAdd(&(dL_dsh[7].x), dL_dsh7.x);
			atomicAdd(&(dL_dsh[7].y), dL_dsh7.y);
			atomicAdd(&(dL_dsh[7].z), dL_dsh7.z);
			atomicAdd(&(dL_dsh[8].x), dL_dsh8.x);
			atomicAdd(&(dL_dsh[8].y), dL_dsh8.y);
			atomicAdd(&(dL_dsh[8].z), dL_dsh8.z);

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				glm::vec3 dL_dsh9 = dRGBdsh9 * dL_dRGB;
				glm::vec3 dL_dsh10 = dRGBdsh10 * dL_dRGB;
				glm::vec3 dL_dsh11 = dRGBdsh11 * dL_dRGB;
				glm::vec3 dL_dsh12 = dRGBdsh12 * dL_dRGB;
				glm::vec3 dL_dsh13 = dRGBdsh13 * dL_dRGB;
				glm::vec3 dL_dsh14 = dRGBdsh14 * dL_dRGB;
				glm::vec3 dL_dsh15 = dRGBdsh15 * dL_dRGB;
				atomicAdd(&(dL_dsh[9].x), dL_dsh9.x);
				atomicAdd(&(dL_dsh[9].y), dL_dsh9.y);
				atomicAdd(&(dL_dsh[9].z), dL_dsh9.z);
				atomicAdd(&(dL_dsh[10].x), dL_dsh10.x);
				atomicAdd(&(dL_dsh[10].y), dL_dsh10.y);
				atomicAdd(&(dL_dsh[10].z), dL_dsh10.z);
				atomicAdd(&(dL_dsh[11].x), dL_dsh11.x);
				atomicAdd(&(dL_dsh[11].y), dL_dsh11.y);
				atomicAdd(&(dL_dsh[11].z), dL_dsh11.z);
				atomicAdd(&(dL_dsh[12].x), dL_dsh12.x);
				atomicAdd(&(dL_dsh[12].y), dL_dsh12.y);
				atomicAdd(&(dL_dsh[12].z), dL_dsh12.z);
				atomicAdd(&(dL_dsh[13].x), dL_dsh13.x);
				atomicAdd(&(dL_dsh[13].y), dL_dsh13.y);
				atomicAdd(&(dL_dsh[13].z), dL_dsh13.z);
				atomicAdd(&(dL_dsh[14].x), dL_dsh14.x);
				atomicAdd(&(dL_dsh[14].y), dL_dsh14.y);
				atomicAdd(&(dL_dsh[14].z), dL_dsh14.z);
				atomicAdd(&(dL_dsh[15].x), dL_dsh15.x);
				atomicAdd(&(dL_dsh[15].y), dL_dsh15.y);
				atomicAdd(&(dL_dsh[15].z), dL_dsh15.z);

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	atomicAdd(&(dL_dmeans[idx].x), dL_dmean.x);
	atomicAdd(&(dL_dmeans[idx].y), dL_dmean.y);
	atomicAdd(&(dL_dmeans[idx].z), dL_dmean.z);
}


__device__ void compute_transmat_uv_forward(
	const glm::vec3 p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* viewmatrix,
    float3& ray_oc,
    float3& ray_dc,
    float& dpt,
    float3& xyz,
	glm::mat3x4& world2splat,
	float3& normal,
    float2& uv
) {
    // Convert the quaternion and scale vector to matrices
    // * NOTE: R here is the row-major rotation matrix, namely R as in Python,
    // * NOTE: We take it as column-major R^T
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

    // float3 mean = make_float3(p_orig.x, p_orig.y, p_orig.z);
    // dpt = sumf3(normal * (mean - ray_oc)) / sumf3(ray_dc * normal);
    // xyz = ray_oc + dpt * ray_dc;

    // Convert the intersection point from world to splat space
    glm::vec3 uv1 = glm::vec4(glm::vec3(xyz.x, xyz.y, xyz.z), 1.0f) * world2splat;
    uv = make_float2(uv1.x, uv1.y);
}


// Backward pass for rotation matrix to quaternion
inline __device__ float4
quat_to_rotmat_vjp_test(const float4& quat, const float3* v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	float4 v_quat;
	// v_R is the gradient w.r.t the trasnpose of rotation matrix
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1].z - v_R[2].y) + y * (v_R[2].x - v_R[0].z) +
				  z * (v_R[0].y - v_R[1].x)
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1].y + v_R[2].z) + y * (v_R[0].y + v_R[1].x) +
			z * (v_R[0].z + v_R[2].x) + w * (v_R[1].z - v_R[2].y)
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0].y + v_R[1].x) - 2.f * y * (v_R[0].x + v_R[2].z) +
			z * (v_R[1].z + v_R[2].y) + w * (v_R[2].x - v_R[0].z)
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0].z + v_R[2].x) + y * (v_R[1].z + v_R[2].y) -
			2.f * z * (v_R[0].x + v_R[1].y) + w * (v_R[0].y - v_R[1].x)
		);
	return v_quat;
}
__forceinline__ __device__ float3 to_float3(const float4& a) {return make_float3(a.x, a.y, a.z);}


__device__ void compute_transmat_uv_backward_2(
	const float3 p_orig,
	const float2 scale, 
	float mod,
	const float4 rot,
	const float* viewmatrix,
    const float3 xyz,
	const float4* world2splat,
	const float normal_sign,
    const float2 uv,
	const float3 v1,
	const float3 v2,
	const float3 v3,
	const float3 ray_o,
	const float3 ray_d,
	const float G,
	const float3& dL_dN,
	const float dL_dD,
	const float dL_dG,
	float2& dL_dscale,
	float4& dL_drot,
	float3& dL_dmean3D,
	const uint3 idx,
	int gidx
)
{
	// Compute the gradient w.r.t. the uv
	float2 dL_duv = dL_dG  * -G * uv;

	float3 dL_dR[3];
	// Compute the gradient w.r.t. the transposed rotation matrix
	dL_dR[0] = dL_duv.x * (xyz - p_orig) / scale.x;
	dL_dR[1] = dL_duv.y * (xyz - p_orig) / scale.y;
	dL_dR[2] = dL_dN * normal_sign;

	// Update the gradient w.r.t. the scale
	dL_dscale = dL_dG  * (G * uv * uv / scale);

	// Update the gradient w.r.t. the mean3D
	float3 dG_dmean3D = G * (to_float3(world2splat[0]) * uv.x / scale.x + to_float3(world2splat[1]) * uv.y / scale.y);
	dL_dmean3D = dL_dG  * dG_dmean3D;

	// Compute the gradient flow through the ray-triangle intersection
	float3 dL_dxyz = make_float3(
		dL_duv.x / scale.x * world2splat[0].x + dL_duv.y / scale.y * world2splat[1].x,
		dL_duv.x / scale.x * world2splat[0].y + dL_duv.y / scale.y * world2splat[1].y,
		dL_duv.x / scale.x * world2splat[0].z + dL_duv.y / scale.y * world2splat[1].z
	);
	float dL_dd = dot(dL_dxyz, ray_d) + dL_dD;
	// Compute the gradient w.r.t. the triangle vertices
	// Define some useful middle variables
	float3 n = cross(v2 - v1, v3 - v1);
	float3 c = v1 - ray_o;
	// Numerator and denominator aliases
	float p = dot(n, c);
	float q = dot(n, ray_d);
	// Chain rule for the gradient
	float3 dL_dn = (c - p / q * ray_d) / q;
	float3 dL_dv1 = cross(v2 - v3, dL_dn) * dL_dd + n / q * dL_dd;
	float3 dL_dv2 = cross(v3 - v1, dL_dn) * dL_dd;
	float3 dL_dv3 = cross(v1 - v2, dL_dn) * dL_dd;

	// if ((idx.x == 420) && (idx.y == 600)) {
	// 	float t = P / Q;
	// 	float3 sec = ray_o + t * ray_d;
	// 	printf("gidx: %u, optix xyz: %.18f %.18f %.18f\n", gidx, xyz.x, xyz.y, xyz.z);
	// 	printf("gidx: %u, optix sec: %.18f %.18f %.18f\n", gidx, sec.x, sec.y, sec.z);
	// 	printf("dL_dv1: %.18f %.18f %.18f\n", dL_dv1.x, dL_dv1.y, dL_dv1.z);
	// 	printf("dL_dd: %.18f, Q: %.18f, N/Q: %.18f %.18f %.18f\n", dL_dd, Q, (N / Q).x, (N / Q).y, (N / Q).z);
	// 	printf("dL_dv2: %.18f %.18f %.18f\n", dL_dv2.x, dL_dv2.y, dL_dv2.z);
	// 	printf("dL_dv3: %.18f %.18f %.18f\n", dL_dv3.x, dL_dv3.y, dL_dv3.z);
	// }

	// Update the gradient w.r.t. the transposed rotation matrix R
	dL_dR[0].x += scale.x * (dL_dv1.x + dL_dv2.x + dL_dv3.x);
	dL_dR[0].y += scale.x * (dL_dv1.y + dL_dv2.y + dL_dv3.y);
	dL_dR[0].z += scale.x * (dL_dv1.z + dL_dv2.z + dL_dv3.z);
	dL_dR[1].x += scale.y * (dL_dv1.x + dL_dv2.x + dL_dv3.x);
	dL_dR[1].y += scale.y * (dL_dv1.y + dL_dv2.y + dL_dv3.y);
	dL_dR[1].z += scale.y * (dL_dv1.z + dL_dv2.z + dL_dv3.z);
	// Update gradient w.r.t. rotation
	dL_drot = quat_to_rotmat_vjp_test(rot, dL_dR);

	// Update the gradient w.r.t. the scale
	dL_dscale.x += dot(to_float3(world2splat[0]), dL_dv1 + dL_dv2 + dL_dv3);
	dL_dscale.y += dot(to_float3(world2splat[1]), dL_dv1 + dL_dv2 + dL_dv3);

	// Update the gradient w.r.t. the mean3D
	dL_dmean3D += dL_dv1 + dL_dv2 + dL_dv3;
}

__device__ void compute_transmat_uv_backward(
	const float3* p_orig,
	const glm::vec2 scale, 
	float mod,
	const glm::vec4 rot,
	const float* viewmatrix,
	const float3& dir,
    const float3& xyz,
	const glm::mat3x4 world2splat,
	const float3& normal,
    const float2 uv,
	const float3& ray_o,
	const float3& ray_d,
	const float* dL_dnorm,
	const float dL_ddpt,
	const float2 dL_duv,
	glm::vec2& dL_dscale,
	glm::vec4& dL_drot,
	glm::vec3& dL_dmean3D)
{
    // Convert the quaternion and scale vector to matrices
    // * NOTE: R here is the row-major rotation matrix, namely R as in Python,
    // * NOTE: We take it as column-major R^T
    // * NOTE: S here is the inverse of the scale matrix
	glm::mat3 R = quat_to_rotmat_transpose(rot);
	glm::mat3 S = scale_to_mat_inverse(scale, mod);
	glm::mat3 L = S * R;

	// Compute the gradient w.r.t. the world2splat matrix
	glm::mat3x4 dL_dworld2splat = glm::mat3x4(
		glm::vec4(xyz.x, xyz.y, xyz.z, 1.0) * dL_duv.x,
		glm::vec4(xyz.x, xyz.y, xyz.z, 1.0) * dL_duv.y,
		glm::vec4(0.0, 0.0, 0.0, 0.0)
	);

	// Compute the gradient w.r.t. the original normal first
	float3 dL_dtw = make_float3(dL_dnorm[0], dL_dnorm[1], dL_dnorm[2]);
#if DUAL_VISIABLE
	float cos = -sumf3(dir * normal);
	dL_dtw = cos > 0 ? dL_dtw : -dL_dtw;
#endif

	float3 dL_dxyz = make_float3(
		dL_duv.x * world2splat[0].x + dL_duv.y * world2splat[1].x,
		dL_duv.x * world2splat[0].y + dL_duv.y * world2splat[1].y,
		dL_duv.x * world2splat[0].z + dL_duv.y * world2splat[1].z
	);
	float dL_dt = dot(dL_dxyz, ray_d) + dL_ddpt;
	float3 mean = *p_orig;
	float3 mu_o = mean - ray_o;
	float nd = dot(normal, ray_d);
	float3 dt_dnormal = (mu_o - (dot(normal, mu_o) / nd * ray_d)) / nd;
	dL_dtw += dL_dt * dt_dnormal;
	float3 dL_dmu = dL_dt * normal / nd;

	// Compute the gradient w.r.t. L
	glm::mat3 dL_dL_col = glm::mat3(
		glm::vec3(
			dL_dworld2splat[0].x - dL_dworld2splat[0].w * mean.x,
			dL_dworld2splat[1].x - dL_dworld2splat[1].w * mean.x,
			dL_dworld2splat[2].x - dL_dworld2splat[2].w * mean.x
		),
		glm::vec3(
			dL_dworld2splat[0].y - dL_dworld2splat[0].w * mean.y,
			dL_dworld2splat[1].y - dL_dworld2splat[1].w * mean.y,
			dL_dworld2splat[2].y - dL_dworld2splat[2].w * mean.y
		),
		glm::vec3(
			dL_dworld2splat[0].z - dL_dworld2splat[0].w * mean.z,
			dL_dworld2splat[1].z - dL_dworld2splat[1].w * mean.z,
			dL_dworld2splat[2].z - dL_dworld2splat[2].w * mean.z
		)
	);

	// Update gradient w.r.t. scale, rotation and mean3D
	glm::mat3 dL_dR_inv_col = glm::mat3(
		dL_dL_col[0] / glm::vec3(scale, 1.f),
		dL_dL_col[1] / glm::vec3(scale, 1.f),
		dL_dL_col[2] / glm::vec3(scale, 1.f)
	);
	glm::mat3 R_col = glm::transpose(R);
	glm::mat3 dL_dR_col = - R_col * dL_dR_inv_col * R_col;
	dL_dR_col[2][0] = dL_dtw.x;
	dL_dR_col[2][1] = dL_dtw.y;
	dL_dR_col[2][2] = dL_dtw.z;
	dL_drot = quat_to_rotmat_vjp(rot, dL_dR_col);
	dL_dscale = glm::vec2(
		-(dL_dL_col[0].x * R[0].x + dL_dL_col[1].x * R[1].x + dL_dL_col[2].x * R[2].x) / scale.x / scale.x,
		-(dL_dL_col[0].y * R[0].y + dL_dL_col[1].y * R[1].y + dL_dL_col[2].y * R[2].y) / scale.y / scale.y
	);
	dL_dmean3D = glm::vec3(
		-(dL_dworld2splat[0].w * L[0].x + dL_dworld2splat[1].w * L[0].y + dL_dworld2splat[2].w * L[0].z) + dL_dmu.x,
		-(dL_dworld2splat[0].w * L[1].x + dL_dworld2splat[1].w * L[1].y + dL_dworld2splat[2].w * L[1].z) + dL_dmu.y,
		-(dL_dworld2splat[0].w * L[2].x + dL_dworld2splat[1].w * L[2].y + dL_dworld2splat[2].w * L[2].z) + dL_dmu.z
	);
}


// Core __raygen__ program
extern "C" __global__ void __raygen__ot()
{
    // Lookup current location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t tidx = idx.x * dim.y + idx.y;
    // bool flag=false;
    // if (idx.x == 25 && idx.y == 1000)
    // {
	// 	printf("backward\n");
    //     flag = true;
    // }

    // Fetch the ray origin and direction of the current pixel
    float3 ray_om = params.ray_o[tidx];
    float3 ray_dm = params.ray_d[tidx];
    // Store a copy of the original ray origin and direction
    float3 ray_oc = ray_om;
    float3 ray_dc = ray_dm;

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
	
    // Prepare rendering data
	float C[3] = {0.0f};
	float clamped[3];
    float dpt = 0.0f;
	glm::mat3x4 world2splat;
    float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal;
    float2 uv;

	float dL_drgb[3];
	dL_drgb[0] = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + INTENSITY_OFFSET];
	dL_drgb[1] = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + RAYHIT_OFFSET];
	dL_drgb[2] = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + RAYDROP_OFFSET];
	float dL_ddpt = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + DEPTH_OFFSET];
	float dL_dacc = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + ACCUM_OFFSET];
	float dL_dnorm[3];
	for (int i = 0; i < 3; i++)
		dL_dnorm[i] = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + NORMAL_OFFSET + i];
	const float T_final = params.out_attr_float32[NUM_CHANNELS_F * tidx + FINALT_OFFSET];
	const float final_D = params.out_attr_float32[NUM_CHANNELS_F * tidx + DISDEPTH_OFFSET];
	const float final_D2 = params.out_attr_float32[NUM_CHANNELS_F * tidx + DISDEPTH2_OFFSET];
	float dL_dmedian_depth = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + MIDDEPTH_OFFSET];
	float dL_ddistortion = params.dL_dout_attr_float32[NUM_CHANNELS_F * tidx + DISTORTION_OFFSET];
    float* chunk_buffer = params.out_attr_float32 + NUM_CHANNELS_F * tidx + CHUNK_DPT_OFFSET;

	const int median_contributor = params.out_attr_uint32[NUM_CHANNELS_I * tidx + MID_CONTRIB_OFFSET];
	const int chunk_cnt = params.out_attr_uint32[NUM_CHANNELS_I * tidx + CHUNK_NUM_OFFSET];
	const int first_idx_forward = params.out_attr_uint32[NUM_CHANNELS_I * tidx + FIRST_IDX_OFFSET];
	const int last_idx_forward = params.out_attr_uint32[NUM_CHANNELS_I * tidx + LAST_IDX_OFFSET];

	// TODO (xbillowy): Implement this?
	// float dL_ddist = params.dL_ddist[tidx];
	// Prepare gradients computation data
	const float final_A = 1 - T_final;
	float T = T_final;
	float last_dL_dT = 0;
	float last_color[3] = {0};
	float acc_colors[3] = {0};
	float last_depth = 0;
	float acc_depths = 0;
	float last_alpha = 0;
	float acc_alphas = 0;
	float last_normal[3] = {0};
	float acc_normals[3] = {0};
	// // What's this?
	// float last_dL_dT = 0;
	// Per-Gaussian gradient
	float dL_dcolor[3];
	glm::vec2 dL_dscale;
	glm::vec4 dL_drot;
	glm::vec3 dL_dmean3D;

    int last_gidx = -1;

    for (int i = chunk_cnt - 1; i >= 0; i--)
	{
        // Re-initialize payload data
        payload.dpt = chunk_buffer[i];
        payload.cnt = 0;
        // Update Ray origin
		if (i == 0)
			ray_om = ray_oc;
		else
			ray_om = ray_oc + (payload.dpt + STEP_EPSILON) * ray_dc;

        // Actual optixTrace
        traceStep(ray_om, ray_dm, payload_u0, payload_u1);
		
        // Volume rendering
        for (int i = CHUNK_SIZE - 1; i >= 0; i--)
        {
            // Break if the intersection depth is invalid
            if (i >= payload.cnt)
                continue;

            // Get the primitive index and Gaussian index
            int pidx = payload.buffer[i].idx;  // intersection primitive index
            int gidx = pidx / 2;  // Gaussian index is half of the primitive index
            // Compute the actual intersection depth and coordinates in world space
            dpt = payload.buffer[i].tmx + payload.dpt;
            xyz = ray_oc + dpt * ray_dc;
            // if (dpt < 0.2f) continue;

            // Re-initialize payload data
            payload.buffer[i].tmx = 1e16f;
            payload.buffer[i].idx = 0;

			if (last_gidx == -1 && gidx != last_idx_forward)
				continue;
            if (gidx == last_gidx)
                continue;
            last_gidx = gidx;

			// Build the world to splat transformation matrix
			compute_transmat_uv_forward(params.means3D[gidx], params.scales[gidx],
										params.scale_modifier, params.rotations[gidx], params.viewmatrix,
										ray_oc, ray_dc, dpt, xyz, world2splat, normal, uv);

			// Get weights
			float rho3d = uv.x * uv.x + uv.y * uv.y;
			float rho2d = rho3d;
			// Get particle response
			float power = -0.5f * min(rho3d, rho2d);
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean
			const float G = exp(power);
			float alpha = min(0.99f, params.opacities[gidx] * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			// TODO (xbillowy): What's this?
			// const float w = alpha * T;
            // if (flag)
            //     printf("gidx: %d, dpt %f\n", gidx, dpt);


			// Compute or fetch forward color or feature first
			if (params.colors_precomp == nullptr)
			{
				glm::vec3 result = computeColorFromSHForward(gidx, params.D, params.M,
															params.means3D, *params.campos,
															params.shs, clamped);
				C[0] = result.x;
				C[1] = result.y;
				C[2] = result.z;
			}
			const float* feature_ptr = params.colors_precomp != nullptr ? params.colors_precomp : C;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair)
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < 3; ch++)
			{
				const float c = feature_ptr[ch];
				acc_colors[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * acc_colors[ch];
				// Update last color (to be used in the next iteration)
				last_color[ch] = c;

				const float dL_dchannel = dL_drgb[ch];
				dL_dalpha += (c - acc_colors[ch]) * dL_dchannel;
				dL_dcolor[ch] = dchannel_dcolor * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian
				atomicAdd(&(params.dL_dcolors[ch + 3 * gidx]), dchannel_dcolor * dL_dchannel);
			}

			float dL_ddpt_gs = 0.0f;
			float dL_dnormal_gs[3] = {0.0f};
#if RENDER_AXUTILITY
			// Propagate gradients w.r.t mid T depths
			if (gidx == median_contributor) {
				dL_ddpt_gs += dL_dmedian_depth;
			}

			// propagate the current weight W_{i} to next weight W_{i-1}
			const float m_d = dpt;
			const float dmd_dd = 1;
#if DETACH_WEIGHT 
			// if not detached weight, sometimes will bia toward creating extragated 2D Gaussians near front
			float dL_dweight = 0;
#else
			float dL_dweight = (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_ddistortion;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_ddistortion;
			dL_ddpt_gs += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t. color ray-splat alphas
			acc_alphas = last_alpha * 1.0 + (1.f - last_alpha) * acc_alphas;
			dL_dalpha += (1 - acc_alphas) * dL_dacc;

			// Propagate gradients w.r.t ray-splat depths
			acc_depths = last_alpha * last_depth + (1.f - last_alpha) * acc_depths;
			last_depth = dpt;
			dL_dalpha += (dpt - acc_depths) * dL_ddpt;
			dL_ddpt_gs += alpha * T * dL_ddpt; 

			// Propagate gradients to per-Gaussian normals
			float normal_tmp[3] = {normal.x, normal.y, normal.z};
			for (int ch = 0; ch < 3; ch++) {
				acc_normals[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * acc_normals[ch];
				last_normal[ch] = normal_tmp[ch];
				dL_dalpha += (normal_tmp[ch] - acc_normals[ch]) * dL_dnorm[ch];
				dL_dnormal_gs[ch] = alpha * T * dL_dnorm[ch];
			}
#endif
			// Update dL_dalpha of current Gaussian
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_x_drgb = 0;
			for (int ch = 0; ch < 3; ch++)
				bg_x_drgb += params.background[ch] * dL_drgb[ch];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_x_drgb;

			// Helpful reusable temporary variables
			const float dL_dG = params.opacities[gidx] * dL_dalpha;
			// float dL_dz = 0.0f;
			// dL_dz += alpha * T * dL_ddpt;

			// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
			const float2 dL_duv = {dL_dG * -G * uv.x, dL_dG * -G * uv.y};

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(params.dL_dopacities[gidx]), G * dL_dalpha);
			
			// Compute gradients w.r.t. scaling, rotation, position of the Gaussian
#if DUAL_VISIABLE
            float3 dir = make_float3(params.means3D[gidx].x - ray_oc.x, params.means3D[gidx].y - ray_oc.y, params.means3D[gidx].z - ray_oc.z);
            // float3 dir = ray_dm;
#endif
			compute_transmat_uv_backward((float3*)(params.means3D + gidx), params.scales[gidx],
										params.scale_modifier, params.rotations[gidx], params.viewmatrix,
										dir, xyz, world2splat, normal, uv, ray_oc, ray_dc, 
										dL_dnormal_gs, dL_ddpt_gs, dL_duv,
										dL_dscale, dL_drot, dL_dmean3D);
										
// 			float3 v1, v2, v3;
// 			// Prepare primitive vertices and local coordinate
// 			if (pidx % 2 == 0)
// 			{
// 				v1 = params.vertices[gidx * 4 + 0];
// 				v2 = params.vertices[gidx * 4 + 1];
// 				v3 = params.vertices[gidx * 4 + 2];
// 			}
// 			else
// 			{
// 				v1 = params.vertices[gidx * 4 + 1];
// 				v2 = params.vertices[gidx * 4 + 2];
// 				v3 = params.vertices[gidx * 4 + 3];
// 			}
// #if DUAL_VISIABLE
// 	float cos = -sumf3(dir * normal);
// 	float normal_sign = cos > 0 ? 1 : -1;
// #endif
// 			compute_transmat_uv_backward_2(*(float3*)(params.means3D + gidx), *(float2*)(params.scales+gidx),
// 										params.scale_modifier, *(float4*)(params.rotations+gidx), params.viewmatrix,
// 										xyz, (float4*)&world2splat, normal_sign, uv, v1, v2, v3, ray_oc, ray_dc,
// 										G, *(float3*)dL_dnormal_gs, dL_ddpt_gs, dL_dG,
// 										*(float2*)&dL_dscale, *(float4*)&dL_drot, *(float3*)&dL_dmean3D, idx, gidx);

			// Update gradients w.r.t. scaling
			atomicAdd(&(params.dL_dscales[gidx].x), dL_dscale.x);
			atomicAdd(&(params.dL_dscales[gidx].y), dL_dscale.y);
			// Update gradients w.r.t. rotation
			atomicAdd(&(params.dL_drotations[gidx].x), dL_drot.x);
			atomicAdd(&(params.dL_drotations[gidx].y), dL_drot.y);
			atomicAdd(&(params.dL_drotations[gidx].z), dL_drot.z);
			atomicAdd(&(params.dL_drotations[gidx].w), dL_drot.w);
			// Update gradients w.r.t. position of the Gaussian
			atomicAdd(&(params.dL_dmeans3D[gidx].x), dL_dmean3D.x);
			atomicAdd(&(params.dL_dmeans3D[gidx].y), dL_dmean3D.y);
			atomicAdd(&(params.dL_dmeans3D[gidx].z), dL_dmean3D.z);

			// Compute the gradient w.r.t. the SHs if they are present
			if (params.colors_precomp == nullptr)
				computeColorFromSHBackward(gidx, params.D, params.M, params.means3D, *params.campos,
										params.shs, clamped, dL_dcolor,
										params.dL_dmeans3D, params.dL_dshs);
			
		}
	}
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
