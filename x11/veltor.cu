extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_shavite.h"
#include "sph/sph_shabal.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// for SM3.x
extern void streebog_sm3_set_target(uint32_t* ptarget);
extern void streebog_sm3_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

// for latest cards only
extern void skunk_cpu_init(int thr_id, uint32_t threads);
extern void skunk_streebog_set_target(uint32_t* ptarget);
extern void skunk_cuda_streebog(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

#include <stdio.h>
#include <memory.h>

#define NBN 2
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// veltor CPU Hash
extern "C" void veltorhash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_gost512_context ctx_gost;
	sph_shabal512_context ctx_shabal;
	sph_shavite512_context ctx_shavite;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*) hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*) hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

// cleanup
extern "C" void free_veltor(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
