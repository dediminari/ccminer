/**
 * Tribus Algo for Denarius
 *
 * tpruvot@github 09 2017 - GPLv3
 *
 */
extern "C" {
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);
void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target);

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// cpu hash

extern "C" void tribus_hash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];

	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_echo512_context ctx_echo;

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, input, 80);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

// ressources cleanup
extern "C" void free_tribus(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
