/*
 * whirlpool routine
 */
extern "C" {
#include <sph/sph_whirlpool.h>
#include <miner.h>
}

#include <cuda_helper.h>

//#define SM3_VARIANT

#ifdef SM3_VARIANT
static uint32_t *d_hash[MAX_GPUS];
extern void whirlpool512_init_sm3(int thr_id, uint32_t threads, int mode);
extern void whirlpool512_free_sm3(int thr_id);
extern void whirlpool512_setBlock_80_sm3(void *pdata, const void *ptarget);
extern void whirlpool512_hash_64_sm3(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void whirlpool512_hash_80_sm3(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern uint32_t whirlpool512_finalhash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
//#define _DEBUG
#define _DEBUG_PREFIX "whirl"
#include <cuda_debug.cuh>
#else
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_free(int thr_id);
extern void whirlpool512_setBlock_80(void *pdata, const void *ptarget);
extern void whirlpool512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, const uint64_t target);
#endif


// CPU Hash function
extern "C" void wcoinhash(void *state, const void *input)
{
	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, input, 80);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hashB);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hashB, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	memcpy(state, hash, 32);
}

void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };

