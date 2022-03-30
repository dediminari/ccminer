/**
 * X11EVO algo implementation
 * Cuda implementation by tpruvot@github - May 2016
 */
#include <stdio.h>
#include <memory.h>

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
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	SKEIN,
	JH,
	KECCAK,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HASH_FUNC_COUNT
};

static void swap8(uint8_t *a, uint8_t *b)
{
	uint8_t t = *a;
	*a = *b;
	*b = t;
}

static void initPerm(uint8_t n[], int count)
{
	for (int i = 0; i < count; i++)
		n[i] = i;
}

static int nextPerm(uint8_t n[], int count)
{
	int tail, i, j;

	if (count <= 1)
		return 0;

	for (i = count - 1; i>0 && n[i - 1] >= n[i]; i--);
	tail = i;

	if (tail > 0) {
		for (j = count - 1; j>tail && n[j] <= n[tail - 1]; j--);
		swap8(&n[tail - 1], &n[j]);
	}

	for (i = tail, j = count - 1; i<j; i++, j--)
		swap8(&n[i], &n[j]);

	return (tail != 0);
}

static void getAlgoString(char *str, int seq)
{
	uint8_t algoList[HASH_FUNC_COUNT];
	char *sptr;

	initPerm(algoList, HASH_FUNC_COUNT);

	for (int k = 0; k < seq; k++) {
		nextPerm(algoList, HASH_FUNC_COUNT);
	}

	sptr = str;
	for (int j = 0; j < HASH_FUNC_COUNT; j++) {
		if (algoList[j] >= 10)
			sprintf(sptr, "%c", 'A' + (algoList[j] - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoList[j]);
		sptr++;
	}
	*sptr = '\0';
	//applog(LOG_DEBUG, "nextPerm %s", str);
}

static __thread uint32_t s_ntime = 0;
static char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };
static int  s_sequence = -1;

#define INITIAL_DATE 0x57254700
static inline int getCurrentAlgoSeq(uint32_t current_time)
{
	// change once per day
	return (int) (current_time - INITIAL_DATE) / (60 * 60 * 24);
}

static void evo_twisted_code(uint32_t ntime, char *permstr)
{
	int seq = getCurrentAlgoSeq(ntime);
	if (s_sequence != seq) {
		getAlgoString(permstr, seq);
		s_sequence = seq;
	}
}

// X11evo CPU Hash
extern "C" void x11evo_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[64/4] = { 0 };

	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_skein512_context     ctx_skein;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_luffa512_context     ctx_luffa1;
	sph_cubehash512_context  ctx_cubehash1;
	sph_shavite512_context   ctx_shavite1;
	sph_simd512_context      ctx_simd1;
	sph_echo512_context      ctx_echo1;

	if (s_sequence == -1) {
		uint32_t *data = (uint32_t*) input;
		const uint32_t ntime = data[17];
		evo_twisted_code(ntime, hashOrder);
	}

	void *in = (void*) input;
	int size = 80;

	const int hashes = (int) strlen(hashOrder);

	for (int i = 0; i < hashes; i++)
	{
		const char elem = hashOrder[i];
		uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		if (i > 0) {
			in = (void*) hash;
			size = 64;
		}

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa1);
			sph_luffa512(&ctx_luffa1, in, size);
			sph_luffa512_close(&ctx_luffa1, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash1);
			sph_cubehash512(&ctx_cubehash1, in, size);
			sph_cubehash512_close(&ctx_cubehash1, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite1);
			sph_shavite512(&ctx_shavite1, in, size);
			sph_shavite512_close(&ctx_shavite1, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd1);
			sph_simd512(&ctx_simd1, in, size);
			sph_simd512_close(&ctx_simd1, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo1);
			sph_echo512(&ctx_echo1, in, size);
			sph_echo512_close(&ctx_echo1, hash);
			break;
		}
	}

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "evo"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

