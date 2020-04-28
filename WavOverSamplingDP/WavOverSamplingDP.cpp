
#include <iostream>
#include <bitset>
#include <immintrin.h>

#include <Windows.h>

#include <VersionHelpers.h>


// reomve comment out below to use Boost
#include <boost/multiprecision/cpp_dec_float.hpp>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Tap size; change this number if necessary. Must be an odd number
//#define TAP_SIZE 4095
//#define TAP_SIZE 524287
#define TAP_SIZE 65535
//#define TAP_SIZE 16383

#define DATA_UNIT_SIZE (1024 * 512)

using namespace boost::multiprecision;
using boost::math::constants::pi;

double* arrayedNormalCoeff[8];
double* arrayedDiffCoeff[8];

void setCoeff(int tapNum, double* coeff1, double* coeff2)
{
	int coeffNum = (tapNum + 1) / 2;
	for (int i = 0; i < coeffNum; ++i)
	{
		int x = i % 8;
		int y = i / 8;
		arrayedNormalCoeff[x][y] = coeff1[coeffNum - 1 + i];
		arrayedDiffCoeff[x][y] = coeff2[coeffNum - 1 + i];
	}
}

void createHannCoeff(int tapNum, double* dest, double* dest2)
{
	int coeffNum = (tapNum + 1) / 2;
	cpp_dec_float_100* coeff1 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);
	cpp_dec_float_100* coeff2 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);
	cpp_dec_float_100* coeff3 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);

	cpp_dec_float_100 piq = pi<cpp_dec_float_100>();

	coeff1[0] = cpp_dec_float_100(2) * 22050 / 352800;
	for (int i = 1; i < coeffNum; ++i)
	{
		cpp_dec_float_100 x = cpp_dec_float_100(i) * 2 * piq * 22050 / 352800;
		coeff1[i] = (cpp_dec_float_100)boost::multiprecision::sin(x) / (piq * cpp_dec_float_100(i));
	}

	#pragma omp parallel for
	for (int i = 0; i < coeffNum; ++i)
	{
		cpp_dec_float_100 x = cpp_dec_float_100(2) * piq * i / (tapNum - 1);
		coeff2[i] = cpp_dec_float_100("0.5")  + cpp_dec_float_100("0.5") * boost::multiprecision::cos(x);
	}
	coeff2[coeffNum - 1] = 0;

	long long scale = 352800 / 22050 * 32768;

	#pragma omp parallel for
	for (int i = 0; i < coeffNum; ++i)
	{
		coeff3[i] = coeff1[i] * coeff2[i] * scale;
	}

	dest[coeffNum - 1] = (double)coeff3[0];
	dest2[coeffNum - 1] = (double)(coeff3[0] - (cpp_dec_float_100)((double)coeff3[0]));

	#pragma omp parallel for
	for (int i = 1; i < coeffNum; ++i)
	{
		double x = (double)coeff3[i];
		dest[coeffNum - 1 + i] = x;
		dest[coeffNum - 1 - i] = x;
		dest2[coeffNum - 1 + i] = (double)(coeff3[i] - (cpp_dec_float_100)x);
		dest2[coeffNum - 1 - i] = (double)(coeff3[i] - (cpp_dec_float_100)x);
	}
	::GlobalFree(coeff1);
	::GlobalFree(coeff2);
	::GlobalFree(coeff3);
}

__inline static int writeRaw32bitPCM(long long left, long long right, int* buffer)
{
	buffer[0] = (int)left;
	buffer[1] = (int)right;

	return buffer[0];
}

__inline int do_oversample(short* src, unsigned int length, double* coeff, double* coeff2, int tapNum, int* dest, int x8pos)
{
	int half_size = (tapNum - 1) / 2;

	__m256d tmp256Left2;
	__m256d tmp256Right2;

	__m256d max256d = _mm256_set_pd(0, 0, 2147483647, 2147483647);
	__m256d min256d = _mm256_set_pd(0, 0, -2147483647-1, -2147483647-1);

	for (unsigned int i = 0; i < length; ++i)
	{
		tmp256Left2 = _mm256_setzero_pd();
		tmp256Right2 = _mm256_setzero_pd();

		short* srcPtr = src + 2;
		double* coeffPtr = arrayedNormalCoeff[8 - x8pos]; //coeff + half_size - x8pos + 8;
		double* coeff2Ptr = arrayedDiffCoeff[8 - x8pos];// coeff2 + half_size - x8pos + 8;

		for (int j = 1; j * 8 <= half_size; j += 8)
		{
			__m256i mSrc256V = _mm256_load_si256((__m256i*)srcPtr); // RL RL RL RL   RL RL RL RL   16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16  256bit
			__m256i mSrc256U = _mm256_srli_si256(mSrc256V, 4);      //    RL RL RL      RL RL RL 
			__m256i mSrc256X = _mm256_unpacklo_epi16(mSrc256V, mSrc256U); // -- -- RR LL --  -- RR LL
			mSrc256V = _mm256_srli_si256(mSrc256V, 8);  //  -- -- RL RL -- -- RL RL  AVX2
			mSrc256U = _mm256_srli_si256(mSrc256V, 4);  //  -- -- -- RL -- -- -- RL
			__m256i mSrc256Y = _mm256_unpacklo_epi16(mSrc256V, mSrc256U); //  -- -- RR LL -- -- RR LL   // AVX2
			__m256i mSrc256LLLL = _mm256_unpacklo_epi32(mSrc256X, mSrc256Y); // RR RR LL LL RR RR LL LL
			mSrc256LLLL = _mm256_permute4x64_epi64(mSrc256LLLL, _MM_SHUFFLE(3,1,2,0)); // RR RR RR RR LL LL LL LL  // AVX2
			__m256i mSrc256RRRR = _mm256_permute4x64_epi64(mSrc256LLLL, _MM_SHUFFLE(1,0,3,2)); // LL LL LL LL RR RR RR RR
			mSrc256LLLL = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mSrc256LLLL)); // L32L32  L32L32  L32L32  L32L32  //  AVX2
			mSrc256RRRR = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mSrc256RRRR)); // R32R32  R32R32  R32R32  R32R32

			//// first 4 samples 

			__m256d mDiffCoeff1 = _mm256_load_pd(coeffPtr);
			__m256d mDiffCoeff2 = _mm256_load_pd(coeffPtr + 4);
			__m128i mSrcLLLL1 = _mm256_extractf128_si256(mSrc256LLLL, 0); // AVX
			__m128i mSrcRRRR1 = _mm256_extractf128_si256(mSrc256RRRR, 0);
			__m128i mSrcLLLL2 = _mm256_extractf128_si256(mSrc256LLLL, 1);
			__m128i mSrcRRRR2 = _mm256_extractf128_si256(mSrc256RRRR, 1);


			__m256d m256SrcLLLL1 = _mm256_cvtepi32_pd(mSrcLLLL1); // L64D L64D L64D L64D // AVX
			__m256d m256SrcRRRR1 = _mm256_cvtepi32_pd(mSrcRRRR1);
			__m256d m256SrcLLLL2 = _mm256_cvtepi32_pd(mSrcLLLL2); // L64D L64D L64D L64D // AVX
			__m256d m256SrcRRRR2 = _mm256_cvtepi32_pd(mSrcRRRR2);

			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL1, mDiffCoeff1, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR1, mDiffCoeff1, tmp256Right2);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL2, mDiffCoeff2, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR2, mDiffCoeff2, tmp256Right2);

			srcPtr += 16;
			coeffPtr += 8;

		}

		srcPtr = src;
		coeffPtr = arrayedNormalCoeff[x8pos]; //coeff + half_size + x8pos;
		coeff2Ptr = arrayedDiffCoeff[x8pos];//coeff2 + half_size + x8pos;

		for (int j = 0; j * 8 <= half_size; j += 8)
		{
			__m256i mSrc256V = _mm256_load_si256((__m256i*)(srcPtr-14)); // RL RL RL RL   RL RL RL RL   16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16  256bit
			//    AA BB   CC DD   EE FF   GG HH
			mSrc256V = _mm256_permute4x64_epi64(mSrc256V, _MM_SHUFFLE(1, 0, 3, 2)); //  EE FF GG HH   AA BB CC DD
			mSrc256V = _mm256_shuffle_epi32(mSrc256V, _MM_SHUFFLE(0, 1, 2, 3)); // HH GG FF EE DD CC BB AA
			__m256i mSrc256U = _mm256_srli_si256(mSrc256V, 4);      //    RL RL RL      RL RL RL 
			__m256i mSrc256X = _mm256_unpacklo_epi16(mSrc256V, mSrc256U); // -- -- RR LL --  -- RR LL
			mSrc256V = _mm256_srli_si256(mSrc256V, 8);  //  -- -- RL RL -- -- RL RL 
			mSrc256U = _mm256_srli_si256(mSrc256V, 4);  //  -- -- -- RL -- -- -- RL
			__m256i mSrc256Y = _mm256_unpacklo_epi16(mSrc256V, mSrc256U); //  -- -- RR LL -- -- RR LL
			__m256i mSrc256LLLL = _mm256_unpacklo_epi32(mSrc256X, mSrc256Y); // RR RR LL LL RR RR LL LL
			mSrc256LLLL = _mm256_permute4x64_epi64(mSrc256LLLL, _MM_SHUFFLE(3, 1, 2, 0)); // RR RR RR RR LL LL LL LL
			__m256i mSrc256RRRR = _mm256_permute4x64_epi64(mSrc256LLLL, _MM_SHUFFLE(1, 0, 3, 2)); // LL LL LL LL RR RR RR RR
			mSrc256LLLL = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mSrc256LLLL)); // L32L32  L32L32  L32L32  L32L32
			mSrc256RRRR = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mSrc256RRRR)); // R32R32  R32R32  R32R32  R32R32

			//////////////
			__m256d mDiffCoeff1 = _mm256_load_pd(coeffPtr);
			__m256d mDiffCoeff2 = _mm256_load_pd(coeffPtr + 4);
			__m128i mSrcLLLL1 = _mm256_extractf128_si256(mSrc256LLLL, 0); // AVX
			__m128i mSrcRRRR1 = _mm256_extractf128_si256(mSrc256RRRR, 0);
			__m128i mSrcLLLL2 = _mm256_extractf128_si256(mSrc256LLLL, 1);
			__m128i mSrcRRRR2 = _mm256_extractf128_si256(mSrc256RRRR, 1);


			__m256d m256SrcLLLL1 = _mm256_cvtepi32_pd(mSrcLLLL1); // L64D L64D L64D L64D // AVX
			__m256d m256SrcRRRR1 = _mm256_cvtepi32_pd(mSrcRRRR1);
			__m256d m256SrcLLLL2 = _mm256_cvtepi32_pd(mSrcLLLL2); // L64D L64D L64D L64D // AVX
			__m256d m256SrcRRRR2 = _mm256_cvtepi32_pd(mSrcRRRR2);

			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL1, mDiffCoeff1, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR1, mDiffCoeff1, tmp256Right2);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL2, mDiffCoeff2, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR2, mDiffCoeff2, tmp256Right2);

			srcPtr -= 16;
			coeffPtr += 8;
		}

		tmp256Left2 = _mm256_hadd_pd(tmp256Left2, tmp256Right2); // c'+d'  c+d  a'+b'  a+b
		tmp256Left2 = _mm256_permute4x64_pd(tmp256Left2, _MM_SHUFFLE(3, 1, 2, 0)); // c'+d' a'+b' c+d a+b
		tmp256Left2 = _mm256_hadd_pd(tmp256Left2, tmp256Left2); //  ---- a'+b'+c'+d' ---- a+b+c+d
		tmp256Left2 = _mm256_permute4x64_pd(tmp256Left2, _MM_SHUFFLE(3, 1, 2, 0)); // a'+b'+c'+d' a+b+c+d

		tmp256Left2 = _mm256_min_pd(tmp256Left2, max256d);
		tmp256Left2 = _mm256_max_pd(tmp256Left2, min256d);
		tmp256Left2 = _mm256_round_pd(tmp256Left2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		__m128i f = _mm_cvtpd_epi32(_mm256_castpd256_pd128(tmp256Left2));
		_mm_storel_epi64((__m128i*)(dest + x8pos * 2), f);


		src += 2;
		dest += 8 * 2;
	}
	return 0;
}


int  oversample(short* src, unsigned int length, double* coeff, double* coeff2, int tapNum, int* dest, unsigned int option)
{
	if (option == 0)
	{
		int half_size = (tapNum - 1) / 2;
		
		for (unsigned int i = 0; i < length; ++i)
		{
			short *srcLeft = src;
			short *srcRight = src + 1;
			long long tmpLeft, tmpRight;

			tmpLeft = *srcLeft * coeff[half_size];
			tmpRight = *srcRight * coeff[half_size];
			writeRaw32bitPCM(tmpLeft, tmpRight, dest);

			src += 2;
			dest += 8 * 2;
		}
	}
	else
	{
		do_oversample(src,  length, coeff, coeff2, tapNum, dest, option);
	}
	return 0;
}


struct oversample_info
{
	short* src;
	unsigned int length;
	double* coeff;
	double* coeff2;
	int tapNum;
	int* dest;
	int option;
};

DWORD WINAPI ThreadFunc(LPVOID arg)
{
	struct oversample_info* info = (struct oversample_info*)arg;
	oversample(info->src, info->length, info->coeff, info->coeff2, info->tapNum, info->dest, info->option);
	return 0;
}

unsigned int searchFmtDataChunk(wchar_t* fileName, WAVEFORMATEX* wf, DWORD* offset, DWORD* size)
{
	HANDLE fileHandle;
	fileHandle = CreateFileW(fileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (fileHandle == INVALID_HANDLE_VALUE)
	{
		return 0;
	}

	DWORD header[2];
	DWORD readSize;
	WORD  wav[8];
	DWORD riffSize, pos = 0;
	DWORD dataOffset, dataSize;
	::ReadFile(fileHandle, header, 8, &readSize, NULL);
	bool fmtFound = false, dataFound = false;

	if (readSize != 8)
	{
		CloseHandle(fileHandle);
		return 0;
	}

	if (header[0] != 0x46464952)
	{
		// not "RIFF"
		CloseHandle(fileHandle);
		return 0;
	}
	riffSize = header[1];

	::ReadFile(fileHandle, header, 4, &readSize, NULL);
	if (readSize != 4)
	{
		CloseHandle(fileHandle);
		return 0;
	}
	if (header[0] != 0x45564157)
	{
		// not "WAVE"
		CloseHandle(fileHandle);
		return 0;
	}
	pos += 4;

	while (pos < riffSize)
	{
		::ReadFile(fileHandle, header, 8, &readSize, NULL);
		if (readSize != 8)
		{
			break;
		}
		pos += 8;

		if (header[0] == 0x20746d66)
		{
			// "fmt "
			if (header[1] >= 16)
			{
				::ReadFile(fileHandle, wav, 16, &readSize, NULL);
				if (readSize != 16)
				{
					break;
				}
				fmtFound = true;
				if (header[1] > 16)
				{
					::SetFilePointer(fileHandle, header[1] - 16, 0, FILE_CURRENT);
				}
				pos += header[1];
			}
			else
			{
				::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
				pos += header[1];
			}
		}
		else if (header[0] == 0x61746164)
		{
			// "data"
			dataFound = true;
			dataOffset = ::SetFilePointer(fileHandle, 0, 0, FILE_CURRENT);
			dataSize = header[1];
			::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
			pos += header[1];
		}
		else
		{
			::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
			pos += header[1];
		}
		if (GetLastError() != NO_ERROR)
		{
			break;
		}
	}
	CloseHandle(fileHandle);

	if (dataFound && fmtFound)
	{
		*offset = dataOffset;
		*size = dataSize;
		wf->wFormatTag = wav[0]; //  1:LPCM   3:IEEE float
		wf->nChannels = wav[1]; //  1:Mono  2:Stereo
		wf->nSamplesPerSec = *(DWORD*)(wav + 2);  // 44100, 48000, 176400, 19200, 352800, 384000...
		wf->nAvgBytesPerSec = *(DWORD*)(wav + 4);
		wf->nBlockAlign = wav[6]; // 4@16bit/2ch,  6@24bit/2ch,   8@32bit/2ch   
		wf->wBitsPerSample = wav[7]; // 16bit, 24bit, 32bit
		wf->cbSize = 0;
		return 1;
	}
	return 0;
}

DWORD readWavFile(wchar_t* fileName, void* readMem, DWORD readPos, DWORD readLength)
{
	HANDLE fileHandle;
	DWORD wavDataOffset, wavDataSize, readSize = 0;
	WAVEFORMATEX wf;

	if (!searchFmtDataChunk(fileName, &wf, &wavDataOffset, &wavDataSize))
	{
		return 0;
	}

	fileHandle = CreateFileW(fileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (fileHandle == INVALID_HANDLE_VALUE)
	{
		return 0;
	}

	if (::SetFilePointer(fileHandle, wavDataOffset + readPos, 0, FILE_BEGIN) == INVALID_SET_FILE_POINTER)
	{
		if (GetLastError() != NO_ERROR)
		{
			// fail
			return 0;
		}
	}
	::ReadFile(fileHandle, readMem, readLength, &readSize, NULL);
	::CloseHandle(fileHandle);

	return readSize;
}

static int writePCM352_32_header(HANDLE fileHandle, unsigned long dataSize)
{
	WAVEFORMATEX wf;
	wf.wFormatTag = 0x01;
	wf.nChannels = 2;
	wf.nSamplesPerSec = 352800;
	wf.nAvgBytesPerSec = 352800 * 8; // 352800 * 4byte(32bit) * 2ch
	wf.nBlockAlign = 8; // 8bytes (32bit, 2ch) per sample
	wf.wBitsPerSample = 32;
	wf.cbSize = 0; // ignored. not written.

	DWORD writtenSize = 0;
	WriteFile(fileHandle, "RIFF", 4, &writtenSize, NULL);
	DWORD size = (dataSize + 44) - 8;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);
	WriteFile(fileHandle, "WAVE", 4, &writtenSize, NULL);
	WriteFile(fileHandle, "fmt ", 4, &writtenSize, NULL);
	size = 16;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);
	WriteFile(fileHandle, &wf, size, &writtenSize, NULL);
	WriteFile(fileHandle, "data", 4, &writtenSize, NULL);
	size = (DWORD)dataSize;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);

	return 0;
}

int wmain(int argc, wchar_t *argv[], wchar_t *envp[])
{
	DWORD wavDataOffset, wavDataSize, writtenSize, length, readSize = 0;
	WAVEFORMATEX wf;
	wchar_t* fileName;
	wchar_t* destFileName;

	if (!IsWindowsVistaOrGreater())
	{
		return 0;
	}

	if (argc < 2) return 0;
	fileName = argv[1];
	destFileName = argv[2];

	SYSTEM_INFO si;
	GetNativeSystemInfo(&si);
	int logicalProcessorCount = 0; // only current processor group;
	int physicalProcessorCount = 0;
	DWORD dw = 0;
	if (IsWindowsVistaOrGreater())
	{
		GetLogicalProcessorInformation(NULL, &dw);
		SYSTEM_LOGICAL_PROCESSOR_INFORMATION* logicalProcessorInfoPtr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)::GlobalAlloc(GPTR, dw);
		GetLogicalProcessorInformation(logicalProcessorInfoPtr, &dw);
		for (int i = 0; i < dw / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
		{
			if (logicalProcessorInfoPtr[i].Relationship == 0)
			{
				++physicalProcessorCount;
				std::bitset<64> a(logicalProcessorInfoPtr[i].ProcessorMask);
				logicalProcessorCount += (int)a.count();
			}
		}
		::GlobalFree(logicalProcessorInfoPtr);
	}

	// FMA3 and AVX2 are requied; Intel Core i3(Haswell)/i5(Haswell)/i7(Haswell)/i9 and newer, AMD Ryzen and newer
	int cpuinfo[4];
	int isFma3Supported = 0;
	int isAvxSupported = 0;
	int isAvx2Supported = 0;
	__cpuid(cpuinfo, 1);
	if (cpuinfo[2] & (1 << 12)) // ECX 
	{
		//FMA3 supported
		isFma3Supported = 1;
	}
	if (cpuinfo[2] & (1 << 28)) // ECX 
	{
		//AVX supported
		isAvxSupported = 1;
	}
	if (isAvxSupported)
	{
		__cpuid(cpuinfo, 7);
		if (cpuinfo[1] & (1 << 5))
		{
			//AVX2 supported
			isAvx2Supported = 1;
		}
	}

	ULONGLONG startTime = GetTickCount64();
	ULONGLONG elapsedTime, calcStartTime;

	if (!searchFmtDataChunk(fileName, &wf, &wavDataOffset, &wavDataSize))
	{
		return 0;
	}
	int part = wavDataSize / DATA_UNIT_SIZE;
	if ((wavDataSize %  DATA_UNIT_SIZE) != 0) part += 1;

	void* memWorkBuffer = _mm_malloc(DATA_UNIT_SIZE * 3 + 1024, 32);
	void* mem1 = memWorkBuffer;
	void* mem2 = (char*)mem1 + DATA_UNIT_SIZE;
	void* mem3 = (char*)mem2 + DATA_UNIT_SIZE;
	void* memOut = _mm_malloc(DATA_UNIT_SIZE * 8 * 2 + 1024, 32);

	HANDLE fileOut = CreateFileW(destFileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	writePCM352_32_header(fileOut, wavDataSize * 8 * 2);

	double* firCoeff = (double* )_mm_malloc(sizeof(double) * TAP_SIZE + 4096, 32);
	double* firCoeff2 = (double*)_mm_malloc(sizeof(double) * TAP_SIZE + 4096, 32);

	createHannCoeff(TAP_SIZE, firCoeff, firCoeff2);
	for (int i = 0; i < 256; ++i)
	{
		firCoeff[TAP_SIZE + i] = 0;
		firCoeff2[TAP_SIZE + i] = 0;
	}

	for (int i = 0; i < 8; ++i)
	{
		int coeffNum = (TAP_SIZE + 1) / 2 + 1024;
		arrayedNormalCoeff[i] = (double*) _mm_malloc(coeffNum * sizeof(double), 32);
		arrayedDiffCoeff[i] = (double*)_mm_malloc(coeffNum * sizeof(double), 32);
		for (int j= 0; j < coeffNum; ++j)
		{
			*(arrayedNormalCoeff[i] + j) = 0;
			*(arrayedDiffCoeff[i] + j) = 0;
		}
	}
	setCoeff(TAP_SIZE, firCoeff, firCoeff2);

	elapsedTime = GetTickCount64() - startTime;
	calcStartTime = GetTickCount64();
	std::cout << "WavOverSampling: Initialization finished:  " << (elapsedTime / 1000) << "." << (elapsedTime % 1000) << " sec  \n";

	float total = 0.0f;

	for (int i = 0; i <= part; ++i)
	{
		::SetThreadExecutionState(ES_SYSTEM_REQUIRED);
		
		length = readSize;
		::CopyMemory(mem1, mem2, DATA_UNIT_SIZE);
		::CopyMemory(mem2, mem3, DATA_UNIT_SIZE);
		::SecureZeroMemory(mem3, DATA_UNIT_SIZE);
		if (i != part)
		{
			readSize = readWavFile(fileName, mem3, DATA_UNIT_SIZE * i, DATA_UNIT_SIZE);
		}
		if (i == 0) continue;
	
		struct oversample_info info[8];
		info[0].src = (short* )mem2;
		info[0].length = length / 4;
		info[0].coeff = firCoeff;
		info[0].coeff2 = firCoeff2;
		info[0].tapNum = TAP_SIZE;
		info[0].dest = (int* )memOut;
		info[0].option = 0;
		
		HANDLE thread[8];
		DWORD threadId[8];
		for (int j = 0; j < 8; ++j)
		{
			info[j] = info[0];
			info[j].option = j;
			thread[j] = CreateThread(NULL, 0, ThreadFunc, (LPVOID)&info[j], 0, &threadId[j]);
		}
		::WaitForMultipleObjects(8, thread, TRUE, INFINITE);
	
		::WriteFile(fileOut, memOut, length * 8 * 2, &writtenSize, NULL);

		total += (DATA_UNIT_SIZE / 4) * 1000;
		elapsedTime = GetTickCount64() - startTime;
		std::cout << "WavOverSampling: Progress  " << (i * 100) / part << "%    " << (total / 44100 / (GetTickCount64() - calcStartTime)) <<  "x    " << (elapsedTime/1000/60) << " min " << (elapsedTime /1000)%60 << " sec  \r";
	}
	elapsedTime = GetTickCount64() - startTime;
	std::cout << "\nWavOverSampling: Completed.   " << (elapsedTime/1000) << "." << (elapsedTime % 1000) <<  " sec  \n";

	::FlushFileBuffers(fileOut);
	::CloseHandle(fileOut);

	for (int i = 0; i < 8; ++i)
	{
		_mm_free(arrayedNormalCoeff[i]);
		_mm_free(arrayedDiffCoeff[i]);
	}

	_mm_free(memWorkBuffer);
	_mm_free(memOut);
	_mm_free(firCoeff);
	_mm_free(firCoeff2);
	return 0;
}
