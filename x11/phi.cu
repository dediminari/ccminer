//
//
//  PHI1612 algo
//  Skein + JH + CubeHash + Fugue + Gost + Echo
//
//  Implemented by anorganix @ bitcointalk on 01.10.2017
//  Feel free to send some satoshis to 1Bitcoin8tfbtGAQNFxDRUVUfFgFWKoWi9
//
//

extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int swap);
extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void streebog_hash_64_maxwell(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void phihash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_jh512_context ctx_jh;
	sph_cubehash512_context ctx_cubehash;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;
	sph_echo512_context ctx_echo;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*)hash, 64);
	sph_gost512_close(&ctx_gost, (void*)hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	memcpy(output, hash, 32);
}

#define _DEBUG_PREFIX "phi"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

