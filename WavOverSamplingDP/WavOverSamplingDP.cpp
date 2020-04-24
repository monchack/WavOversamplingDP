
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

// 16(15+1)bit  X  scale: 48(47+1)bit =  63(62+1)bit -> 32bit (31bit shift)
#define COEFF_SCALE 47
#define SCALE_SHIFT 31

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

	long long scale = 1LL << (COEFF_SCALE + 3);

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
	int shift = SCALE_SHIFT;


	int add = 1 << (shift - 1);
	int x = -1 << shift;

	if (left >= 0) left += add;
	else
	{
		left -= add;
		if (left > x) left = 0;
	}
	if (right >= 0) right += add;
	else
	{
		right -= add;
		if (right > x) right = 0;
	}

	
	if (left >= 4611686018427387904) left = 4611686018427387904 - 1; // over 63bit : limitted to under [1 << 62]   62bit + 1bit
	if (right >= 4611686018427387904) right = 4611686018427387904 - 1;

	if (left < -4611686018427387904) left = -4611686018427387904;
	if (right < -4611686018427387904) right = -4611686018427387904;
	

	left = left >> shift;
	right = right >> shift;

	buffer[0] = (int)left;
	buffer[1] = (int)right;

	return buffer[0];
}

__inline int do_oversample(short* src, unsigned int length, double* coeff, double* coeff2, int tapNum, int* dest, int x8pos)
{
	int half_size = (tapNum - 1) / 2;

	__declspec(align(32)) long long tmpLR[4];

	__m256d tmp256Left2;
	__m256d tmp256Right2;

	for (unsigned int i = 0; i < length; ++i)
	{
		tmp256Left2 = _mm256_setzero_pd();
		tmp256Right2 = _mm256_setzero_pd();

		short* srcPtr = src + 2;
		double* coeffPtr = arrayedNormalCoeff[8 - x8pos]; //coeff + half_size - x8pos + 8;
		double* coeff2Ptr = arrayedDiffCoeff[8 - x8pos];// coeff2 + half_size - x8pos + 8;

		for (int j = 1; j * 8 <= half_size; j += 4)
		{
			__m256d mDiffCoeff = _mm256_load_pd(coeffPtr);
			//__m256d mDiffCoeff = _mm256_load_pd(coeff2Ptr);
			__m128i mSrcV = _mm_loadu_si128((__m128i*)srcPtr); // RL RL RL RL   16 16 16 16 16 16 16 16
			__m128i mSrcU = _mm_srli_si128(mSrcV, 4);          //    RL RL RL
			__m128i mSrcX = _mm_unpacklo_epi16(mSrcV, mSrcU); //  lower RR LL
			mSrcV = _mm_srli_si128(mSrcV, 8);  //  RL RL
			mSrcU = _mm_srli_si128(mSrcV, 4);  //     RL
			__m128i mSrcY = _mm_unpacklo_epi16(mSrcV, mSrcU); // upper RR LL
			__m128i mSrcLLLL = _mm_unpacklo_epi32(mSrcX, mSrcY); // RR RR LL LL
			__m128i mSrcRRRR = _mm_srli_si128(mSrcLLLL, 8); // RR RR
			mSrcLLLL = _mm_cvtepi16_epi32(mSrcLLLL); // L32 L32 L32 L32 <- L16 L16 L16 L16
			mSrcRRRR = _mm_cvtepi16_epi32(mSrcRRRR);
			__m256d m256SrcLLLL = _mm256_cvtepi32_pd(mSrcLLLL); // L64D L64D L64D L64D // AVX
			__m256d m256SrcRRRR = _mm256_cvtepi32_pd(mSrcRRRR);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL, mDiffCoeff, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR, mDiffCoeff, tmp256Right2);

			mDiffCoeff = _mm256_load_pd(coeff2Ptr);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL, mDiffCoeff, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR, mDiffCoeff, tmp256Right2);

			srcPtr += 8;
			coeffPtr += 4;
			coeff2Ptr += 4;
		}

		srcPtr = src;
		coeffPtr = arrayedNormalCoeff[x8pos]; //coeff + half_size + x8pos;
		coeff2Ptr = arrayedDiffCoeff[x8pos];//coeff2 + half_size + x8pos;

		for (int j = 0; j * 8 <= half_size; j += 4)
		{
			__m256d mDiffCoeff = _mm256_load_pd(coeffPtr);
			__m128i mSrcV = _mm_loadu_si128((__m128i*)(srcPtr-6)); // RL RL RL RL   16 16 16 16 16 16 16 16
			mSrcV = _mm_shuffle_epi32(mSrcV, _MM_SHUFFLE(0,1,2,3)); // from(0), from(1), from(2), from(3)      3:2:1:0
			__m128i mSrcU = _mm_srli_si128(mSrcV, 4);          //    RL RL RL
			__m128i mSrcX = _mm_unpacklo_epi16(mSrcV, mSrcU); //  RR LL
			mSrcV = _mm_srli_si128(mSrcV, 8);  //  RL RL
			mSrcU = _mm_srli_si128(mSrcV, 4);  //     RL
			__m128i mSrcY = _mm_unpacklo_epi16(mSrcV, mSrcU); //  RR LL
			__m128i mSrcLLLL = _mm_unpacklo_epi32(mSrcX, mSrcY); //RR RR LL LL
			__m128i mSrcRRRR = _mm_srli_si128(mSrcLLLL, 8); // RR RR
			mSrcLLLL = _mm_cvtepi16_epi32(mSrcLLLL); // L32 L32 L32 L32 <- L16 L16 L16 L16
			mSrcRRRR = _mm_cvtepi16_epi32(mSrcRRRR);
			__m256d m256SrcLLLL = _mm256_cvtepi32_pd(mSrcLLLL); // L64D L64D L64D L64D
			__m256d m256SrcRRRR = _mm256_cvtepi32_pd(mSrcRRRR);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL, mDiffCoeff, tmp256Left2);
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR, mDiffCoeff, tmp256Right2);

			mDiffCoeff = _mm256_load_pd(coeff2Ptr);
			tmp256Left2 = _mm256_fmadd_pd(m256SrcLLLL, mDiffCoeff, tmp256Left2); // FMA
			tmp256Right2 = _mm256_fmadd_pd(m256SrcRRRR, mDiffCoeff, tmp256Right2);

			srcPtr -= 8;
			coeffPtr += 4;
			coeff2Ptr += 4;
		}

		double x = (tmp256Left2.m256d_f64[0] + tmp256Left2.m256d_f64[1] + tmp256Left2.m256d_f64[2] + tmp256Left2.m256d_f64[3]);
		double y = (tmp256Right2.m256d_f64[0] + tmp256Right2.m256d_f64[1] + tmp256Right2.m256d_f64[2] + tmp256Right2.m256d_f64[3]);
		tmpLR[0] = round(x);
		tmpLR[1] = round(y);

		writeRaw32bitPCM(tmpLR[0], tmpLR[1], dest + x8pos * 2);

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
				logicalProcessorCount += a.count();
			}
		}
		::GlobalFree(logicalProcessorInfoPtr);
	}

	// FMA3 and AVX are requied; Core i3/5/7/9 (Haswell) and later, AMD FX (Piledriver) and later
	int cpuinfo[4];
	int isFma3Supported = 0;
	int isAvxSupported = 0;
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
