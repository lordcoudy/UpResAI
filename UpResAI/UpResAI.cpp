

#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "CmdArgsMap.hpp"
#include "image_io_util.hpp"

#include "nvsdk_ngx.h"

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
	int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(
			stderr,
			"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

void AboutDevice(int argc, char **argv)
{
	int *pArgc = &argc;
	char **pArgv = argv;
	printf("%s Starting...\n\n", argv[0]);
	printf(
		" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);
		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
			memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize);
		}

#endif

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total shared memory per multiprocessor:        %zu bytes\n",
			deviceProp.sharedMemPerMultiprocessor);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %zu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %zu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Managed Memory:                %s\n",
			deviceProp.managedMemory ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown", NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	// If there are 2 or more GPUs, query to determine whether RDMA is supported
	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

			// Only boards based on Fermi or later can support P2P
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				// on Windows (64-bit), the Tesla Compute Cluster driver for windows
				// must be enabled to support this
				&& prop[i].tccDriver
#endif
				) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		// Show all the combinations of support P2P GPUs
		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
						prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
						can_access_peer ? "Yes" : "No");
				}
			}
		}
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	printf("%s", sProfileString.c_str());

}

typedef struct _appParams {
	std::string wd;
	std::string input_image_filename;
	std::string output_image_filename;
	uint32_t uprez_factor;
} appParams;

NVSDK_NGX_Handle *DUHandle{ nullptr };
NVSDK_NGX_Parameter *params{ nullptr };

long long app_id = 0x0;

void NGXTestCallback(float InProgress, bool &OutShouldCancel)
{
	//Perform progress handling here.
	//For long running cases.
	//e.g. LOG("Progress callback %.2f%", InProgress * 100.0f);
	OutShouldCancel = false;
}

int NGXSuperResolution(appParams &AppParams)
{
	// Read input image into host memory
	std::string input_image_file_path = AppParams.wd + AppParams.input_image_filename;

	int image_width, image_height;
	const auto rgba_bitmap_ptr = getRgbImage(input_image_file_path, image_width, image_height);
	if (nullptr == rgba_bitmap_ptr)
	{
		std::cerr << "Error reading Image " << input_image_file_path << std::endl;
		return 1;
	}
	// Copy input image to GPU device memory
	size_t in_image_row_bytes = image_width * 3;
	size_t in_image_width = image_width;
	size_t in_image_height = image_height;
	void *in_image_dev_ptr;

	if (cudaMalloc(&in_image_dev_ptr, in_image_row_bytes * in_image_height) != cudaSuccess)
	{
		std::cerr << "Error allocating output image CUDA buffer" << std::endl;
		return 1;
	}

	if (cudaMemcpy(in_image_dev_ptr, rgba_bitmap_ptr.get(), in_image_row_bytes * in_image_height,
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "Error copying input RGBA image to CUDA buffer" << std::endl;
		return -1;
	}

	// Calculate output image paramters
	size_t out_image_row_bytes = image_width * AppParams.uprez_factor * 3;
	size_t out_image_width = image_width * AppParams.uprez_factor;
	size_t out_image_height = image_height * AppParams.uprez_factor;
	void * out_image_dev_ptr;

	if (cudaMalloc(&out_image_dev_ptr, out_image_row_bytes * out_image_height) != cudaSuccess)
	{
		std::cout << "Error allocating output image CUDA buffer" << std::endl;
		return 1;
	}

	// Initialize NGX.
	NVSDK_NGX_Result rslt = NVSDK_NGX_Result_Success;
	rslt = NVSDK_NGX_CUDA_Init(app_id, L"./", NVSDK_NGX_Version_API);
	if (rslt != NVSDK_NGX_Result_Success) {
		std::cerr << "Error Initializing NGX. " << std::endl;
		return 1;
	}

	// Get the parameter block.
	NVSDK_NGX_CUDA_GetParameters(&params);

	// Verify feature is supported
	int Supported = 0;
	params->Get(NVSDK_NGX_Parameter_ImageSuperResolution_Available, &Supported);
	if (!Supported)
	{
		std::cerr << "NVSDK_NGX_Feature_ImageSuperResolution Unavailable on this System" << std::endl;
		return 1;
	}

	// Set the default hyperparameters for inferrence.
	params->Set(NVSDK_NGX_Parameter_Width, in_image_width);
	params->Set(NVSDK_NGX_Parameter_Height, in_image_height);
	params->Set(NVSDK_NGX_Parameter_Scale, AppParams.uprez_factor);

	// Get the scratch buffer size and create the scratch allocation.
	// (if required)
	size_t byteSize{ 0u };
	void *scratchBuffer{ nullptr };
	rslt = NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_ImageSuperResolution, params, &byteSize);
	if (rslt != NVSDK_NGX_Result_Success) {
		std::cerr << "Error Getting NGX Scratch Buffer Size. " << std::endl;
		return 1;
	}
	cudaMalloc(&scratchBuffer, byteSize > 0u ? byteSize : 1u); //cudaMalloc, unlike malloc, fails on 0 size allocations

	// Update the parameter block with the scratch space metadata.:
	params->Set(NVSDK_NGX_Parameter_Scratch, scratchBuffer);
	params->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, (uint32_t)byteSize);

	// Create the feature
	NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature_ImageSuperResolution, params, &DUHandle);

	// Pass the pointers to the GPU allocations to the
	// parameter block along with the format and size.
	params->Set(NVSDK_NGX_Parameter_Color_SizeInBytes, in_image_row_bytes * in_image_height);
	params->Set(NVSDK_NGX_Parameter_Color_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
	params->Set(NVSDK_NGX_Parameter_Color, in_image_dev_ptr);
	params->Set(NVSDK_NGX_Parameter_Output_SizeInBytes, out_image_row_bytes * out_image_height);
	params->Set(NVSDK_NGX_Parameter_Output_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
	params->Set(NVSDK_NGX_Parameter_Output, out_image_dev_ptr);

	//Synchronize the device.
	cudaDeviceSynchronize();
	cudaEventSynchronize;

	//Execute the feature.
	NVSDK_NGX_CUDA_EvaluateFeature(DUHandle, params, NGXTestCallback);

	//Synchronize once more.
	cudaDeviceSynchronize();
	cudaEventSynchronize;

	// Copy output image from GPU device memory
	std::unique_ptr<unsigned char[] > out_image{};
	out_image = std::unique_ptr<unsigned char[]>(new unsigned char[out_image_row_bytes * out_image_height]);

	if (cudaMemcpy(out_image.get(), out_image_dev_ptr, out_image_row_bytes * out_image_height,
		cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Error copying output image from CUDA buffer" << std::endl;
		return 1;
	}

	// Write output image from host memory
	std::string output_image_file_path = AppParams.wd + AppParams.output_image_filename;
	putRgbImage(output_image_file_path, out_image.get(), (int)out_image_width, (int)out_image_height);

	// Tear down the feature.
	NVSDK_NGX_CUDA_ReleaseFeature(DUHandle);

	// Shutdown NGX
	NVSDK_NGX_CUDA_Shutdown();

	//Clean up device buffers.
	cudaFree(in_image_dev_ptr);
	cudaFree(out_image_dev_ptr);

	in_image_dev_ptr = NULL;
	out_image_dev_ptr = NULL;

	return 0;
}

int main(int argc, char *argv[])
{
	AboutDevice(argc, argv);

	appParams myAppParams{ "","","",0 };

	// Process command line arguments
	bool show_help = false;
	int uprez_factor_arg;
	CmdArgsMap cmdArgs = CmdArgsMap(argc, argv, "--")
		("help", "Produce help message", &show_help)
		("wd", "Base directory for image input and output files", &myAppParams.wd, myAppParams.wd)
		("input", "Input image filename", &myAppParams.input_image_filename, myAppParams.input_image_filename)
		("output", "Output image filename", &myAppParams.output_image_filename, myAppParams.output_image_filename)
		("factor", "Super resolution factor (2, 4, 8)", &uprez_factor_arg, uprez_factor_arg);

	if (argc == 1 || show_help)
	{
		std::cout << cmdArgs.help();
		return 1;
	}

	// Verify that specified super resolution factor is valid
	if ((uprez_factor_arg != 2) && (uprez_factor_arg != 4) && (uprez_factor_arg != 8))
	{
		std::cerr << "Image super resolution factor (--factor) must be one of 2, 4 or 8." << std::endl;
		return 1;
	}
	else
	{
		myAppParams.uprez_factor = uprez_factor_arg;
	}

	// Verify input image file specified.
	if (myAppParams.input_image_filename.empty())
	{
		std::cerr << "Input image filename must be specified." << std::endl;
		return 1;
	}

	// Verify output image file specified.
	if (myAppParams.output_image_filename.empty())
	{
		std::cerr << "Output image filename must be specified." << std::endl;
		return 1;
	}

	// Append trailing '/' to working directory if not specified to reduce user errors.
	if (!myAppParams.wd.empty())
	{
		switch (myAppParams.wd[myAppParams.wd.size() - 1])
		{
#ifdef _MSC_VER
		case '\\':
#endif // _MSC_VER
		case '/':
			break;
		default:
			myAppParams.wd += '/';
			break;
		}
	}

	std::cout << myAppParams.wd << myAppParams.input_image_filename << " * " << myAppParams.uprez_factor << " -> " << myAppParams.output_image_filename << std::endl;
	std::cout << "Upscaling..." << std::endl;
	if (NGXSuperResolution(myAppParams) == 0)
	{
		std::cout << "Image upscaling completed!" << std::endl;
	}
	else
	{
		std::cout << "Image upscaling gone wrong!" << std::endl;
	}
	

	return 0;
}
