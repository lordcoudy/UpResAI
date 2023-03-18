// UpResAI.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include "nvsdk_ngx.h"
#include "CmdArgsMap.hpp"
#include "image_io_util.hpp"


#define debug 0

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

void NGXSuperResolution(appParams ImageParams)
{

	// Read input image into host memory
	std::string input_image_file_path = ImageParams.wd + ImageParams.input_image_filename;

	int image_width, image_height;
	const auto rgba_bitmap_ptr = getRgbImage(input_image_file_path, image_width, image_height);
	if (nullptr == rgba_bitmap_ptr)
	{
		std::cerr << "Error reading Image " << input_image_file_path << std::endl;
		return;
	}

	// Copy input image to GPU device memory
	size_t in_image_row_bytes = image_width * 3;
	size_t in_image_width = image_width;
	size_t in_image_height = image_height;
	void* in_image_dev_ptr;

	if (cudaMalloc(&in_image_dev_ptr, in_image_row_bytes * in_image_height) != cudaSuccess)
	{
		std::cerr << "Error allocating output image CUDA buffer" << std::endl;
		return;
	}

	if (cudaMemcpy(in_image_dev_ptr, rgba_bitmap_ptr.get(), in_image_row_bytes * in_image_height,
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cerr << "Error copying input RGBA image to CUDA buffer" << std::endl;
		return;
	}

	// Calculate output image paramters
	size_t out_image_row_bytes = image_width * ImageParams.uprez_factor * 3;
	size_t out_image_width = image_width * ImageParams.uprez_factor;
	size_t out_image_height = image_height * ImageParams.uprez_factor;
	void* out_image_dev_ptr;

	if (cudaMalloc(&out_image_dev_ptr, out_image_row_bytes * out_image_height) != cudaSuccess)
	{
		std::cout << "Error allocating output image CUDA buffer" << std::endl;
		return;
	}

	// ------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------
	// Initialize NGX.

	NVSDK_NGX_Result rslt = NVSDK_NGX_Result_Success;
	rslt = NVSDK_NGX_CUDA_Init(app_id, L"./", NVSDK_NGX_Version_API);
	if (rslt != NVSDK_NGX_Result_Success) {
		std::cerr << "Error Initializing NGX. " << std::endl;
		return;
	}

	// Get the parameter block.
	NVSDK_NGX_CUDA_GetParameters(&params);

	// Verify feature is supported
	int Supported = 0;
	params->Get(NVSDK_NGX_Parameter_ImageSuperResolution_Available, &Supported);
	if (!Supported)
	{
		std::cerr << "NVSDK_NGX_Feature_ImageSuperResolution Unavailable on this System" << std::endl;
		return;
	}

	// ------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------


	// Set the default hyperparameters for inferrence.
	params->Set(NVSDK_NGX_Parameter_Width, in_image_width);
	params->Set(NVSDK_NGX_Parameter_Height, in_image_height);
	params->Set(NVSDK_NGX_Parameter_Scale, ImageParams.uprez_factor);

	// Get the scratch buffer size and create the scratch allocation.
	// (if required)
	unsigned long long byteSize{ 0u };
	void* scratchBuffer{ nullptr };
	rslt = NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_ImageSuperResolution, params, &byteSize);
	if (rslt != NVSDK_NGX_Result_Success) {
		std::cerr << "Error Getting NGX Scratch Buffer Size. " << std::endl;
		return;
	}
	cudaMalloc(&scratchBuffer, byteSize > 0u ? byteSize : 1u); //cudaMalloc, unlike malloc, fails on 0 size allocations

	// Update the parameter block with the scratch space metadata.:
	params->Set(NVSDK_NGX_Parameter_Scratch, scratchBuffer);
	params->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, (uint32_t)byteSize);

	// Create the feature
	NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature_ImageSuperResolution, params, &DUHandle);

	// Pass the pointers to the GPU allocations to the
	// parameter block along with the format and size.
	params->Set(NVSDK_NGX_Parameter_Color_SizeInBytes, in_image_row_bytes* in_image_height);
	params->Set(NVSDK_NGX_Parameter_Color_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
	params->Set(NVSDK_NGX_Parameter_Color, in_image_dev_ptr);
	params->Set(NVSDK_NGX_Parameter_Output_SizeInBytes, out_image_row_bytes* out_image_height);
	params->Set(NVSDK_NGX_Parameter_Output_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
	params->Set(NVSDK_NGX_Parameter_Output, out_image_dev_ptr);

	//Synchronize the device.
	cudaDeviceSynchronize();

	//Execute the feature.
	NVSDK_NGX_CUDA_EvaluateFeature(DUHandle, params, NGXTestCallback);

	//Synchronize once more.
	cudaDeviceSynchronize();

	// Copy output image from GPU device memory
	std::unique_ptr<unsigned char[] > out_image{};
	out_image = std::unique_ptr<unsigned char[]>(new unsigned char[out_image_row_bytes * out_image_height]);

	if (cudaMemcpy(out_image.get(), out_image_dev_ptr, out_image_row_bytes * out_image_height,
		cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Error copying output image from CUDA buffer" << std::endl;
		return;
	}

	// Write output image from host memory
	std::string output_image_file_path = ImageParams.wd + ImageParams.output_image_filename;
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

}

int main(int argc, char* argv[])
{

	appParams myAppParams{ "","","",4 };

	bool exitQ = false;
	// Process command line arguments
	bool show_help = false;
	CmdArgsMap cmdArgs = CmdArgsMap(argc, argv, "--")
		("help", "Produce help message", &show_help)
		("wd", "Base directory for image input and output files", &myAppParams.wd, myAppParams.wd)
		("input", "Input image filename", &myAppParams.input_image_filename, myAppParams.input_image_filename)
		("output", "Output image filename", &myAppParams.output_image_filename, myAppParams.output_image_filename);

	if (argc == 1 || show_help)
	{
		std::cout << cmdArgs.help();
		return 1;
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
	NGXSuperResolution(myAppParams);

	std::cout << "Image upscaling completed!" << std::endl;

	return 0;
}