/**
 * HMQ1725 algorithm
 * @author tpruvot@github 02-2017
 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include <miner.h>
#include <cuda_helper.h>

#include "x11/cuda_x11.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_hash_br2[MAX_GPUS];
static uint32_t *d_tempBranch[MAX_GPUS];

extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen);

struct hmq_contexts
{
	sph_blake512_context    blake1, blake2;
	sph_bmw512_context      bmw1, bmw2, bmw3;
	sph_groestl512_context  groestl1, groestl2;
	sph_skein512_context    skein1, skein2;
	sph_jh512_context       jh1, jh2;
	sph_keccak512_context   keccak1, keccak2;
	sph_luffa512_context    luffa1, luffa2;
	sph_cubehash512_context	cubehash;
	sph_shavite512_context	shavite1, shavite2;
	sph_simd512_context     simd1, simd2;
	sph_echo512_context     echo1, echo2;
	sph_hamsi512_context    hamsi;
	sph_fugue512_context    fugue1, fugue2;
	sph_shabal512_context   shabal;
	sph_whirlpool_context   whirlpool1, whirlpool2, whirlpool3, whirlpool4;
	sph_sha512_context      sha1, sha2;
	sph_haval256_5_context  haval1, haval2;
};

static __thread hmq_contexts base_contexts;
static __thread bool hmq_context_init = false;

static void init_contexts(hmq_contexts *ctx)
{
	sph_bmw512_init(&ctx->bmw1);
	sph_bmw512_init(&ctx->bmw2);
	sph_bmw512_init(&ctx->bmw2);
	sph_bmw512_init(&ctx->bmw3);
	sph_whirlpool_init(&ctx->whirlpool1);
	sph_whirlpool_init(&ctx->whirlpool2);
	sph_whirlpool_init(&ctx->whirlpool3);
	sph_whirlpool_init(&ctx->whirlpool4);
	sph_groestl512_init(&ctx->groestl1);
	sph_groestl512_init(&ctx->groestl2);
	sph_skein512_init(&ctx->skein1);
	sph_skein512_init(&ctx->skein2);
	sph_jh512_init(&ctx->jh1);
	sph_jh512_init(&ctx->jh2);
	sph_keccak512_init(&ctx->keccak1);
	sph_keccak512_init(&ctx->keccak2);
	sph_blake512_init(&ctx->blake1);
	sph_blake512_init(&ctx->blake2);
	sph_luffa512_init(&ctx->luffa1);
	sph_luffa512_init(&ctx->luffa2);
	sph_cubehash512_init(&ctx->cubehash);
	sph_shavite512_init(&ctx->shavite1);
	sph_shavite512_init(&ctx->shavite2);
	sph_simd512_init(&ctx->simd1);
	sph_simd512_init(&ctx->simd2);
	sph_echo512_init(&ctx->echo1);
	sph_echo512_init(&ctx->echo2);
	sph_hamsi512_init(&ctx->hamsi);
	sph_fugue512_init(&ctx->fugue1);
	sph_fugue512_init(&ctx->fugue2);
	sph_shabal512_init(&ctx->shabal);
	sph_sha512_init(&ctx->sha1);
	sph_sha512_init(&ctx->sha2);
	sph_haval256_5_init(&ctx->haval1);
	sph_haval256_5_init(&ctx->haval2);
}

// CPU Check
extern "C" void hmq17hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[32];

	const uint32_t mask = 24;

	hmq_contexts ctx;
	if (!hmq_context_init) {
		init_contexts(&base_contexts);
		hmq_context_init = true;
	}
	memcpy(&ctx, &base_contexts, sizeof(hmq_contexts));

	sph_bmw512(&ctx.bmw1, input, 80);
	sph_bmw512_close(&ctx.bmw1, hash);

	sph_whirlpool(&ctx.whirlpool1, hash, 64);
	sph_whirlpool_close(&ctx.whirlpool1, hash);

	if (hash[0] & mask) {
		sph_groestl512(&ctx.groestl1, hash, 64);
		sph_groestl512_close(&ctx.groestl1, hash);
	} else {
		sph_skein512(&ctx.skein1, hash, 64);
		sph_skein512_close(&ctx.skein1, hash);
	}

	sph_jh512(&ctx.jh1, hash, 64);
	sph_jh512_close(&ctx.jh1, hash);
	sph_keccak512(&ctx.keccak1, hash, 64);
	sph_keccak512_close(&ctx.keccak1, hash);

	if (hash[0] & mask) {
		sph_blake512(&ctx.blake1, hash, 64);
		sph_blake512_close(&ctx.blake1, hash);
	} else {
		sph_bmw512(&ctx.bmw2, hash, 64);
		sph_bmw512_close(&ctx.bmw2, hash);
	}

	sph_luffa512(&ctx.luffa1, hash, 64);
	sph_luffa512_close(&ctx.luffa1, hash);

	sph_cubehash512(&ctx.cubehash, hash, 64);
	sph_cubehash512_close(&ctx.cubehash, hash);

	if (hash[0] & mask) {
		sph_keccak512(&ctx.keccak2, hash, 64);
		sph_keccak512_close(&ctx.keccak2, hash);
	} else {
		sph_jh512(&ctx.jh2, hash, 64);
		sph_jh512_close(&ctx.jh2, hash);
	}

	sph_shavite512(&ctx.shavite1, hash, 64);
	sph_shavite512_close(&ctx.shavite1, hash);

	sph_simd512(&ctx.simd1, hash, 64);
	sph_simd512_close(&ctx.simd1, hash);
	//applog_hash(hash);

	if (hash[0] & mask) {
		sph_whirlpool(&ctx.whirlpool2, hash, 64);
		sph_whirlpool_close(&ctx.whirlpool2, hash);
	} else {
		sph_haval256_5(&ctx.haval1, hash, 64);
		sph_haval256_5_close(&ctx.haval1, hash);
		memset(&hash[8], 0, 32);
	}

	sph_echo512(&ctx.echo1, hash, 64);
	sph_echo512_close(&ctx.echo1, hash);

	sph_blake512(&ctx.blake2, hash, 64);
	sph_blake512_close(&ctx.blake2, hash);
	//applog_hash(hash);

	if (hash[0] & mask) {
		sph_shavite512(&ctx.shavite2, hash, 64);
		sph_shavite512_close(&ctx.shavite2, hash);
	} else {
		sph_luffa512(&ctx.luffa2, hash, 64);
		sph_luffa512_close(&ctx.luffa2, hash);
	}

	sph_hamsi512(&ctx.hamsi, hash, 64);
	sph_hamsi512_close(&ctx.hamsi, hash);

	sph_fugue512(&ctx.fugue1, hash, 64);
	sph_fugue512_close(&ctx.fugue1, hash);
	//applog_hash(hash);

	if (hash[0] & mask) {
		sph_echo512(&ctx.echo2, hash, 64);
		sph_echo512_close(&ctx.echo2, hash);
	} else {
		sph_simd512(&ctx.simd2, hash, 64);
		sph_simd512_close(&ctx.simd2, hash);
	}

	sph_shabal512(&ctx.shabal, hash, 64);
	sph_shabal512_close(&ctx.shabal, hash);

	sph_whirlpool(&ctx.whirlpool3, hash, 64);
	sph_whirlpool_close(&ctx.whirlpool3, hash);
	//applog_hash(hash);

	if (hash[0] & mask) {
		sph_fugue512(&ctx.fugue2, hash, 64);
		sph_fugue512_close(&ctx.fugue2, hash);
	} else {
		sph_sha512(&ctx.sha1, hash, 64);
		sph_sha512_close(&ctx.sha1, hash);
	}

	sph_groestl512(&ctx.groestl2, hash, 64);
	sph_groestl512_close(&ctx.groestl2, hash);

	sph_sha512(&ctx.sha2, hash, 64);
	sph_sha512_close(&ctx.sha2, hash);
	//applog_hash(hash);

	if (hash[0] & mask) {
		sph_haval256_5(&ctx.haval2, hash, 64);
		sph_haval256_5_close(&ctx.haval2, hash);
		memset(&hash[8], 0, 32);
	} else {
		sph_whirlpool(&ctx.whirlpool4, hash, 64);
		sph_whirlpool_close(&ctx.whirlpool4, hash);
	}
	//applog_hash(hash);

	sph_bmw512(&ctx.bmw3, hash, 64);
	sph_bmw512_close(&ctx.bmw3, hash);

	memcpy(output, hash, 32);
}

__global__ __launch_bounds__(128, 8)
void hmq_filter_gpu(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch2, uint32_t* d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint4 *psrc = (uint4*) (&d_hash[offset]);
		d_NonceBranch[thread] = ((uint8_t*)psrc)[0] & 24U;
		if (d_NonceBranch[thread]) return;
		// uint4 = 4x uint32_t = 16 bytes
		uint4 *pdst = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__global__ __launch_bounds__(128, 8)
void hmq_merge_gpu(const uint32_t threads, uint32_t* d_hash, uint32_t* d_branch2, uint32_t* const d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads && !d_NonceBranch[thread])
	{
		const uint32_t offset = thread * 16U;
		uint4 *pdst = (uint4*) (&d_hash[offset]);
		uint4 *psrc = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__host__
uint32_t hmq_filter_cpu(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// extract algo permution hashes to a second branch buffer
	hmq_filter_gpu <<<grid, block>>> (threads, inpHashes, d_branch2, d_tempBranch[thr_id]);
	return threads;
}

__host__
void hmq_merge_cpu(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back second branch hashes to the common buffer d_hash
	hmq_merge_gpu <<<grid, block>>> (threads, outpHashes, d_branch2, d_tempBranch[thr_id]);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "hmq-"
#include "cuda_debug.cuh"

