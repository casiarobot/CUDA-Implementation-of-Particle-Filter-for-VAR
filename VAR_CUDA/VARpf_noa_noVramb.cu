#include "VAR_kernel.cu" 
#include "scan_kernel.cu"

#include <time.h>
#include <random>

std::random_device rd;
std::default_random_engine generator( rd() );
std::normal_distribution<double> distribution(0, 1.0);

//#include "normalize_kernel.cu"
#include <algorithm>
#define ENTER printf("\n");
#define TEST printf("test?\n");

using namespace std;

///The observations
cv_obs * y;
///The inputs
cv_inputs * u;

int main (void)
{

	//**********************************************************************//
	// Initialise CUDA Device                                               //
	//**********************************************************************//
	if(!InitCUDA()) return 0;

	//**********************************************************************//
	// Read Configurations                                                  //
	//**********************************************************************//
	SimParams *hostParams;
	hostParams = (SimParams *)malloc(sizeof(SimParams));
	ReadConfig(hostParams);
	CUDA_CALL(cudaMemcpyToSymbol(deviceParams, hostParams, sizeof(SimParams), 0, cudaMemcpyHostToDevice));

	unsigned int NP = 1<<14; //
	unsigned int iter = 1202; // Number of iterations
	FILE *fd = fopen(".\\16_BPF_estimation_.txt", "w");
	FILE *fidESS = fopen(".\\16_BPF_ESS_.txt", "w");
	//**********************************************************************//
	// Allocate Device Memory                                               //
	//**********************************************************************//
	// Allocate X(state) particles on Device Memory
	unsigned int size_X_particles = sizeof(double) * NP * SIZEOFX;
	double *device_X_particles;
	double *device_X_particles_copy;
	CUDA_CALL(cudaMalloc((void **) &device_X_particles , size_X_particles));
	CUDA_CALL(cudaMalloc((void **) &device_X_particles_copy , size_X_particles));
	// Allocate Y(observation) particles on Device Memory
	unsigned int size_Y_particles = sizeof(double) * NP * SIZEOFY;
	double *device_Y_particles;
	CUDA_CALL(cudaMalloc((void **) &device_Y_particles , size_Y_particles));
	// Allocate W(weight) particles on Device Memory
	unsigned int size_W_particles = sizeof(double) * NP;
	double *device_W_particles, *host_W_particles;
	double *device_W_particles_scanned;
	host_W_particles = (double *)malloc(size_W_particles);
	CUDA_CALL(cudaMalloc((void **) &device_W_particles , size_W_particles));
	CUDA_CALL(cudaMalloc((void **) &device_W_particles_scanned , size_W_particles));
	//**********************************************************************//
	// Load y and u from data.csv                                           //
	//**********************************************************************//
	load_data("data.csv", &y, &u);
	//**********************************************************************//
	// Initialise y and u on Device Memory                                  //
	//**********************************************************************//
	double *device_y, *host_y;
	CUDA_CALL(cudaMalloc((void **) &device_y, sizeof(double) * SIZEOFY));
	host_y = (double*)malloc(sizeof(double) * SIZEOFY);
	double *device_u, *host_u;
	CUDA_CALL(cudaMalloc((void **) &device_u, sizeof(double) * SIZEOFU));
	host_u = (double*)malloc(sizeof(double) * SIZEOFU);
	host_y[0] = y[0].G;
	host_y[1] = y[0].Xram;
	host_y[2] = y[0].I;
	host_y[3] = y[0].Me;
	host_y[4] = y[0].Scl;
	host_y[5] = y[0].Smr;
	host_y[6] = y[0].phe;
	host_u[0] = u[0].Ic;
	host_u[1] = u[0].Vramc;
	CUDA_CALL(cudaMemcpy(device_y , host_y , sizeof(double) * SIZEOFY , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(device_u , host_u , sizeof(double) * SIZEOFU , cudaMemcpyHostToDevice));
	//**********************************************************************//
	// Allocate Device Memory for SCANFAN	     	                        //
	//**********************************************************************//
	double** g_scanBlockSums;
	int level = 0;
	int numEle = NP;
	while(numEle > BLOCK_SIZE){
		level ++;
		numEle = numEle / BLOCK_SIZE;
	}
	g_scanBlockSums = (double**) malloc(level * sizeof(double*));

	numEle = NP;
	level = 0;
	while(numEle > BLOCK_SIZE){
		numEle = numEle / BLOCK_SIZE;
		CUDA_CALL(cudaMalloc((void**) &g_scanBlockSums[level],  numEle * sizeof(double)));
		level ++;
	}
	/************************************************************************/
	/* Assign memory space to Index Array                                   */
	/************************************************************************/
	unsigned int sizeMaxj = sizeof(int) * (NP + 1);
	int *device_maxj, *host_maxj;
	CUDA_CALL(cudaMalloc((void**)&device_maxj, sizeMaxj));
	CUDA_CALL(cudaMallocHost((void **)&host_maxj, sizeof(int) * (NP + 1)));
	host_maxj[NP] = NP - 1;
	CUDA_CALL(cudaMemcpy(device_maxj , host_maxj , sizeof(int) * (NP + 1) , cudaMemcpyHostToDevice));
	//**********************************************************************//
	// Allocate memory for REDUCE               	                        //
	//**********************************************************************//
	unsigned int reduceThreads, reduceBlocks;
	reduceThreads = (NP < BLOCK_SIZE*2) ? nextPow2((NP + 1)/ 2) : BLOCK_SIZE;
	reduceBlocks = (NP + (reduceThreads * 2 - 1)) / (reduceThreads * 2);
	if (reduceBlocks >= 8)
		reduceBlocks = reduceBlocks/8;
	else if (reduceBlocks >= 4)
		reduceBlocks = reduceBlocks/4;
	else if (reduceBlocks >= 2)
		reduceBlocks = reduceBlocks/2;
	unsigned int sizeStatesBlockSumArray = sizeof(double) * reduceBlocks * SIZEOFX;
	double *device_statesBlockSumArray;
	CUDA_CALL(cudaMalloc((void**)&device_statesBlockSumArray, sizeStatesBlockSumArray));
	double *device_weightsBlockSumArray,*device_weights_squareBlockSumArray;
	CUDA_CALL(cudaMalloc((void**)&device_weightsBlockSumArray, sizeof(double) * reduceBlocks));
	CUDA_CALL(cudaMalloc((void**)&device_weights_squareBlockSumArray, sizeof(double) * reduceBlocks));
	double *sum_weight = (double *)malloc(sizeof(double));
	double *sum_weight_square = (double *)malloc(sizeof(double));
	//**********************************************************************//
	// Allocate memory for XE                   	                        //
	//**********************************************************************//
	double *device_Xe, *host_Xe;
	CUDA_CALL(cudaMalloc((void **) &device_Xe, sizeof(double) * SIZEOFX));
	host_Xe = (double *)malloc(sizeof(double) * SIZEOFX);
	//**********************************************************************//
	// Initialise the particles and compute their weights                   //
	//**********************************************************************//
	double *host_rand_init = (double*) malloc(sizeof(double)*NP*SIZEOFX);
	double *host_rand_move = (double*) malloc(sizeof(double)*NP*4);
	double *device_rand_init, *device_rand_move;
	
	CUDA_CALL(cudaMalloc((void **) &device_rand_init, sizeof(double) * NP * SIZEOFX));
	CUDA_CALL(cudaMalloc((void **) &device_rand_move, sizeof(double) * NP * 4));
	for (int i = 0; i < SIZEOFX * NP; i++)
			host_rand_init[i] = distribution(generator);
	CUDA_CALL(cudaMemcpy(device_rand_init, host_rand_init,sizeof(double) * NP * SIZEOFX,
				cudaMemcpyHostToDevice));	
	// Manually-produced data
	double Sdot_i = 4.0*hostParams->mu0*(hostParams->Vc+(hostParams->Ri+hostParams->Rg*y[0].G)
					*u[0].Ic)*u[0].Ic/hostParams->hs/pi/hostParams->De/hostParams->De;
	double delta_i = 7.0*hostParams->alphar*(0.5+hostParams->betam/3.0)/Sdot_i;
	Initialise<<< NP/BLOCK_SIZE, BLOCK_SIZE >>>(/* Input and also output */
												device_X_particles, 
												/* Inputs*/
												device_y,			device_u,			/* Observations and Inputs*/
												delta_i,	device_rand_init,	/* Random numbers*/
												NP,										/* Number of particles*/
												/* Output */
												device_W_particles);

	//**********************************************************************//
	// Array for ESS			                                            //
	//**********************************************************************//
	double *ESS_Array = (double *) malloc(iter * sizeof(double));

	//**********************************************************************//
	// Iterations start								                        //
	//**********************************************************************//
	unsigned int iterate = 0;
	StopWatchInterface *timer = NULL;
	cudaEvent_t start, stop;
	float time_part1 = 0.0f;
	float time_part2 = 0.0f;
	float GPUtime = 0;
	float CPUtime = 0;

	float time_sampling = 0.0f;
	float time_scan = 0.0f;
	float time_normalise = 0.0f;
	float time_resampling = 0.0f;
	float time_reduce = 0.0f;

	float time_sampling_total = 0.0f;
	float time_scan_total = 0.0f;
	float time_normalise_total = 0.0f;
	float time_resampling_total = 0.0f;
	float time_reduce_total = 0.0f;

	clock_t start_CPU, end_CPU;
	
	for (iterate = 1; iterate < iter; iterate ++)
	{
		printf("Iteration %d ......\n",iterate);

		// Prepare device_y and device_u
		host_y[0] = y[iterate].G;
		host_y[1] = y[iterate].Xram;
		host_y[2] = y[iterate].I;
		host_y[3] = y[iterate].Me;
		host_y[4] = y[iterate].Scl;
		host_y[5] = y[iterate].Smr;
		host_y[6] = y[iterate].phe;
		host_u[0] = u[iterate-1].Ic;
		host_u[1] = u[iterate-1].Vramc;
		CUDA_CALL(cudaMemcpy(device_y , host_y , sizeof(double) * SIZEOFY , cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(device_u , host_u , sizeof(double) * SIZEOFU , cudaMemcpyHostToDevice));
		for (int i = 0; i < 4 * NP; i++)
			host_rand_move[i] = distribution(generator);
		CUDA_CALL(cudaMemcpy(device_rand_move, host_rand_move,sizeof(double) * NP * 4,cudaMemcpyHostToDevice));
		//**********************************************************************//
		// Start recording time							                        //
		//**********************************************************************//
		sdkCreateTimer(&timer);
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));
		// Call the SAMPLING AND IMPORTANCE module (kernel SI)
		SI<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(	/* Inputs*/
												device_X_particles,	
												device_y,			device_u,			/* Observations and Inputs*/
												device_rand_move,	/* Random numbers*/	
												NP,					/* Number of particles*/
												/* Output*/
												device_X_particles_copy,		device_W_particles);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);
		checkCudaErrors(cudaEventElapsedTime(&time_sampling, start, stop));
		
		// Normalise weights to be sensible values and calculate the ESS(on CPU)
		CUDA_CALL(cudaMemcpy(host_W_particles , device_W_particles , size_W_particles , cudaMemcpyDeviceToHost));
		
		start_CPU = clock();

		//double dMaxWeight = -std::numeric_limits<double>::infinity();
		//long double sum = 0;
		//long double sumsq = 0;
		//double nResampled = 0;
		//for(unsigned int i = 0; i < NP; i++)
		//	dMaxWeight = max(dMaxWeight, host_W_particles[i]);
		//for(unsigned int i = 0; i < NP; i++){
		//	host_W_particles[i] -= dMaxWeight;
		//	sum += expl(host_W_particles[i]);
		//	sumsq += expl(2.0 * host_W_particles[i]);
		//	host_W_particles[i] = expl(host_W_particles[i]);
		//}

		//ESS_Array[iterate] = expl(-log(sumsq) + 2*log(sum));
		//end_CPU = clock();

		sdkCreateTimer(&timer);
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));

		//CUDA_CALL(cudaMemcpy(device_W_particles , host_W_particles , size_W_particles , cudaMemcpyHostToDevice));
		// Do the RESAMPLING conditionally	
		//if (expl(-log(sumsq) + 2*log(sum)) < (NP / 2))
		//{
		//	nResampled = 1;
		scanFan<double>(device_W_particles_scanned , device_W_particles , NP , BLOCK_SIZE , 0 , 
							g_scanBlockSums);
		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);
		checkCudaErrors(cudaEventElapsedTime(&time_scan, start, stop));

		sdkCreateTimer(&timer);
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));

		normalize<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(device_W_particles_scanned, NP, device_maxj);	

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);
		checkCudaErrors(cudaEventElapsedTime(&time_normalise, start, stop));

		sdkCreateTimer(&timer);
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));

		resample<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(NP, device_X_particles_copy, device_X_particles, 
														device_maxj, device_W_particles);

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);
		checkCudaErrors(cudaEventElapsedTime(&time_resampling, start, stop));

		sdkCreateTimer(&timer);
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		sdkStartTimer(&timer);
		checkCudaErrors(cudaEventRecord(start, 0));
		//}
		//else
		//	nResampled = 0;

		// Calculate Xe
		reduce<double>(NP, reduceThreads, reduceBlocks, 6, device_X_particles, 
						device_statesBlockSumArray , NP, reduceBlocks);
		reduce<double>(reduceBlocks, reduceThreads, 1, 6, device_statesBlockSumArray, 
						device_statesBlockSumArray , reduceBlocks, reduceBlocks);
		calculateXe<<<1,64>>>(reduceBlocks, device_statesBlockSumArray, device_Xe);
		CUDA_CALL(cudaMemcpy(host_Xe , device_Xe , sizeof(double) * SIZEOFX , cudaMemcpyDeviceToHost));
		
		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);
		checkCudaErrors(cudaEventElapsedTime(&time_reduce, start, stop));	
		//**********************************************************************//
		// End recording time							                        //
		//**********************************************************************//
		//GPUtime += time_part1+time_part2;
		//CPUtime += ((double) (end_CPU - start_CPU)*1000) / CLOCKS_PER_SEC;

		time_sampling_total += time_sampling;
		time_scan_total += time_scan;
		time_normalise_total += time_normalise;
		time_resampling_total += time_resampling;
		time_reduce_total += time_reduce;

		for (int i = 0; i < SIZEOFX; i++)
			fprintf(fd, "%12.6f\t",host_Xe[i]/NP);
		fprintf(fd, "\n");
	} // end for (iterate = 1; iterate < iter; iterate ++)

	printf("\nNumber of particles: %d\n",NP);
	printf("Sampling Time: \t%f\n", time_sampling_total/(iter - 1.0));
	printf("Scan Time: \t%f\n", time_scan_total/(iter - 1.0));
	printf("Normalise Time: \t%f\n", time_normalise_total/(iter - 1.0));
	printf("Resampling Time: \t%f\n", time_resampling_total/(iter - 1.0));
	printf("resample time: \t%f\n", time_scan_total/(iter - 1.0)+time_normalise_total/(iter - 1.0)+time_resampling_total/(iter - 1.0));
	printf("Reduce Time: \t%f\n", time_reduce_total/(iter - 1.0));


	
	//for (int i = 1; i < iter; i++)
	//{
	//	fprintf(fidESS, "%d    %lf\n", i, ESS_Array[i]);
	//}
	
	//printf("Number of particles: %d\n",NP);
	//printf("Elapsed Time for one iteration :  \nGPUtime = %f\nCPUtime = %f\n",
	//			GPUtime/(iter - 1.0), CPUtime/(iter - 1.0));
	//printf("Total Time = %lf\n",GPUtime/(iter - 1.0)+CPUtime/(iter - 1.0));
	//**********************************************************************//
	// Free Memory space on both Host and Device                            //
	//**********************************************************************//
	free(host_W_particles);
	CUDA_CALL(cudaFree(device_X_particles));
	CUDA_CALL(cudaFree(device_X_particles_copy));
	CUDA_CALL(cudaFree(device_Y_particles));
	CUDA_CALL(cudaFree(device_W_particles));
	CUDA_CALL(cudaFree(device_W_particles_scanned));
	CUDA_CALL(cudaFree(device_rand_init));
	CUDA_CALL(cudaFree(device_rand_move));
	free(host_rand_init);
	free(host_rand_move);
	free(host_y);
	free(host_u);
	CUDA_CALL(cudaFree(device_y));
	CUDA_CALL(cudaFree(device_u));
	for (int i = 0; i < level; i++) CUDA_CALL(cudaFree(g_scanBlockSums[i]));
	free((void**)g_scanBlockSums);
	CUDA_CALL(cudaFree(device_maxj));
	CUDA_CALL(cudaFreeHost(host_maxj));
	CUDA_CALL(cudaFree(device_statesBlockSumArray));
	CUDA_CALL(cudaFree(device_weightsBlockSumArray));
	CUDA_CALL(cudaFree(device_weights_squareBlockSumArray));
	CUDA_CALL(cudaFree(device_Xe));
	free(host_Xe);
	free(ESS_Array);
	free(sum_weight);
	free(sum_weight_square);
	fclose(fidESS);
	fclose(fd);
	free(hostParams);
	return 0;
}

int Rng_Initialise(SimParams *hostParams, curandGenerator_t gen, double *device_rand_init , unsigned int NP , unsigned long long seed)
{
	// Generate random numbers for	delta
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init, NP, 0.0, sqrt(hostParams->var_delta0)));
	// Generate random numbers for	G
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+1));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+NP, NP, 0.0, sqrt(hostParams->var_G0)));
	// Generate random numbers for	Xram
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+2));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+2*NP, NP, 0.0, sqrt(hostParams->var_Xram0)));
	// Generate random numbers for	Me
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+3));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+3*NP, NP, 0.0, sqrt(hostParams->var_Me0)));
	// Generate random numbers for	mu
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+4));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+4*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmamur));
	// Generate random numbers for	Vb
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+5));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+5*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmaVb));
	// Generate random numbers for	Ib
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+6));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+6*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmaIb));
	// Generate random numbers for	phe
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+7));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_init+7*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmahe));

	return 0;

}

int Rng_LogLikelihood(SimParams *hostParams, curandGenerator_t gen, double *device_rand_log , unsigned int NP , unsigned long long seed)
{
	// Generate random numbers for	G: Electrode gap
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log, NP, 0.0, hostParams->sigmaG));
	// Generate random numbers for	Xram: Ram position
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+1));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+NP, NP, 0.0, hostParams->sigmaPos));
	// Generate random numbers for	I: Measured current
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+2));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+2*NP, NP, 0.0, hostParams->sigmaImeas));
	// Generate random numbers for	Me: Electrode mass
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+3));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+3*NP, NP, 0.0, hostParams->sigmaLC));
	// Generate random numbers for	V: Measured voltage
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+4));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+4*NP, NP, 0.0, hostParams->sigmaVmeas));
	// Generate random numbers for	Scl: Centerline pool depth
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+5));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+5*NP, NP, 0.0, hostParams->sigmaCL));
	// Generate random numbers for	Smr: Mid-radius pool depth
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+6));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+6*NP, NP, 0.0, hostParams->sigmaMR));
	// Generate random numbers for	phe: Helium pressure
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+7));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_log+7*NP, NP, 0.0, hostParams->sigmahemeas));

	return 0;

}

int Rng_Move(SimParams *hostParams, curandGenerator_t gen, double *device_rand_move , unsigned int NP , unsigned long long seed)
{
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move, NP, 0.0, hostParams->sigmaI));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+1));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+NP, NP, 0.0, hostParams->sigmaVram));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+2));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+2*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmamur));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+3));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+3*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmaVb));
	
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+4));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+4*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmaIb));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+5));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+5*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigmahe));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+6));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+6*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigma_pooldepth));

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed+7));
	CURAND_CALL(curandGenerateNormalDouble(gen, device_rand_move+7*NP, NP, 0.0, sqrt(hostParams->dt)*hostParams->sigma_pooldepth));

	return 0;

}

bool InitCUDA()
{
	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	cudaDeviceProp prop;
	for(i = 0; i < count; i++) {	
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major > DEVICE_CAPABILITY_MAJOR) break;
			else if(prop.major == DEVICE_CAPABILITY_MAJOR && prop.minor >= DEVICE_CAPABILITY_MINOR) break;
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);
//	printf("CUDA Device initialized.\n");
//	printf ("Running on device %s with compute capability %d.%d\n", prop.name,prop.major,prop.minor);
	return true;
}

void ReadConfig(SimParams *hostParams)
{
	hostParams->De = 17.0*2.54; //Electrode diameter (in)
	hostParams->Di = 20.0*2.54; //Ingot diameter (in)
	hostParams->I0 = 6000; //Nominal current (A)
	hostParams->G0 = 0.36*2.54; //Nominal gap (in)
	hostParams->phe0 = 3.0; //Nominal helium pressure (Torr)
	hostParams->mu0 = 0.440087; //Nominal melting efficiency
	hostParams->Vc = 21.18; //Cathode voltage fall (V)
	hostParams->Ri = 4.37e-4; //Gap-independent electric resistance (Ohm)
	hostParams->Rg = 0.0; //Gap-dependent electric resistance (Ohm/cm)
	hostParams->Scl0 = 15.702565; //Nominal centerline pool depth (cm)
	hostParams->Smr0 = 13.234694; //Nominal mid-radius pool depth (cm)
	hostParams->Acl = 1.909e-3; //A matrix (1)
	hostParams->Amr = 1.443e-3; //
	hostParams->Bdeltacl = 2.60e-5; //Bdelta matrix (1)
	hostParams->Bdeltamr = -1.29e-4; //
	hostParams->Bicl = 6.587e-6; //Bi matrix (cm/A)
	hostParams->Bimr = 3.165e-6; //
	hostParams->Bmucl = 6.899e-2; //Bmu matrix (cm)
	hostParams->Bmumr = 2.686e-2; //
	hostParams->Bhecl = -8.091e-4; //Bhe matrix (cm/Torr)
	hostParams->Bhemr = -6.541e-4; //
	hostParams->sigmaI = 20.0; //Current standard deviation (A)
	hostParams->sigmaVram = 5.0e-4; //Ram speed standard deviation (cm/s)
	hostParams->sigmamur = 1.0e-2; //Melting efficiency standard deviation (1)
//	hostParams->sigmaa = 1.0e-3; //Fill ratio standard deviation (1)
//	hostParams->sigmaVb = 1.0e-3; //Voltage bias standard deviation
//	hostParams->sigmaIb = 1.0e-3; //Current bias standard deviation
//	hostParams->sigmaVramb = 1.0e-3; //Ram speed bias standard deviation
	hostParams->sigmahe = 1.0e-3; //Helium pressure standard deviation
	hostParams->sigmaG = 0.2; //Measured electrode gap standard deviation (cm)
	hostParams->sigmaPos = 0.005; //Measured ram position standard deviation (cm)
	hostParams->sigmaImeas = 15.0; //Measured current standard deviation (A)
	hostParams->sigmaLC = 200.0; //Measured load cell standard deviation (g)
	hostParams->sigmaVmeas = 0.1; //Measured voltage standard deviation (V)
	hostParams->sigmaCL = 1.0; //Measured centerline pool depth standard deviation (cm)
	hostParams->sigmaMR = 1.0; //Measured mid-radius pool depth standard deviation (cm)
	hostParams->sigmahemeas = 1.0e-2; //Measured helium pressure standard deviation (Torr)
	hostParams->dt = 6; //Time step (s)
	// Other global variables
	hostParams->alphar = 0.023821;
	hostParams->alpham = 0.059553;
	hostParams->hr = 0.000000;
	hostParams->rhor = 7.750000;
	hostParams->hm = 698.750000 * hostParams->rhor;
	hostParams->hs = 1038.750000 * hostParams->rhor;
	double Lambda = (hostParams->hs-hostParams->hm)/hostParams->hm;
	hostParams->a0 = 1-(hostParams->De/hostParams->Di)*(hostParams->De/hostParams->Di);
	double V0 = hostParams->Vc + (hostParams->Ri+hostParams->Rg*hostParams->G0)*hostParams->I0;
	double Pm0 = hostParams->mu0*V0*hostParams->I0;
	double den = 11.0*Lambda+3.0;
	hostParams->betam = (hostParams->alpham-hostParams->alphar)/hostParams->alphar;
	hostParams->Cdd = 224.0*(Lambda+1.0)*(0.5+hostParams->betam/3.0)/den;
	hostParams->Cdp = 32.0/den;
	hostParams->Csd = 56.0*(0.5+hostParams->betam/3.0)/den;
	hostParams->Csp = 11.0/den;
	double mdot0, Sdot0;
	mdot0 = Pm0*hostParams->rhor/hostParams->hs;
	Sdot0 = 4*mdot0/hostParams->rhor/pi/hostParams->De/hostParams->De;
	hostParams->delta0 = 7.0*hostParams->alphar*(0.5+hostParams->betam/3.0)/Sdot0;
	hostParams->Vram0 = 4*hostParams->a0*mdot0/hostParams->rhor/pi/hostParams->De/hostParams->De;
	hostParams->sigmamur *= hostParams->mu0;
	hostParams->sigmaa *= hostParams->a0;
	hostParams->sigmaVb *= V0;
	hostParams->sigmaIb *= hostParams->I0;
	hostParams->sigmaVramb *= hostParams->Vram0;
	hostParams->sigmahe *= hostParams->phe0;

	hostParams->epsilon = 1.0e-10;
	hostParams->var_delta0 = 5.0;
//	hostParams->var_G0 = 1.0;
//	hostParams->var_Xram0 = 0.1;
//	hostParams->var_Me0 = 1.0;
//	hostParams->sigma_pooldepth = 0.01;
	hostParams->G11 = -4.0*hostParams->Cdp*hostParams->Vc*hostParams->mu0/pi/hostParams->De/hostParams->De/hostParams->hm - 8.0*hostParams->Cdp*hostParams->Ri*hostParams->mu0*hostParams->I0/pi/hostParams->De/hostParams->De/hostParams->hm;
	hostParams->G21 = 4.0*hostParams->Csp*hostParams->a0*hostParams->mu0*hostParams->Vc/pi/hostParams->De/hostParams->De/hostParams->hm + 8.0*hostParams->Csp*hostParams->a0*hostParams->mu0*hostParams->Ri*hostParams->I0/pi/hostParams->De/hostParams->De/hostParams->hm;
	hostParams->G41 = -hostParams->rhor*hostParams->Csp*hostParams->Vc*hostParams->mu0/hostParams->hm - 2.0*hostParams->rhor*hostParams->Csp*hostParams->Ri*hostParams->mu0*hostParams->I0/hostParams->hm;
}

long load_data(char const * szName, cv_obs** yp, cv_inputs** up)
{
  FILE * fObs = fopen(szName,"rt");
  if (fObs==NULL) {fputs ("File error: fObs",stderr); exit (1);}
  char* szBuffer = new char[1024];
  if ( fgets(szBuffer, 1024, fObs) == NULL) {
    perror("Need total number of observations");
    return 0;
  }
  long lIterates = strtol(szBuffer, NULL, 10);
  char * pch ;

  *yp = new cv_obs[lIterates];
  *up = new cv_inputs[lIterates];
  
  for(long i = 0; i < lIterates; ++i)
    {
      if ( fgets(szBuffer, 1024, fObs) == NULL ) {
	perror("Not enough lines");
	return 0;
      }
      pch = strtok(szBuffer, ",\r\n\t ");
      (*yp)[i].Time = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].I = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Me = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Xram = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].G = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*up)[i].Ic = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*up)[i].Vramc = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Smr = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Scl = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].phe = strtod(pch, NULL);
    }
  fclose(fObs);

  delete [] szBuffer;

  return lIterates;
}