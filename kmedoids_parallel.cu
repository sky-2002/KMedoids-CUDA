#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>   
#include <sstream>    
#include <fstream>    
#include <ctime>      
#include <chrono>     
#include <float.h>   

using namespace std::chrono;
using namespace std;

#define D 2   // Dimension of points
#define TPB 32 // Threads per block


__device__ float distance(float x1, float y1, float x2, float y2)
{
	return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}


// Our custom function
// we also need to update the cluster sizes array after the
// cluster assignment
void _updateClusterSizes(int* clust_sizes, int* clust_assn, int N, int K){
	for(int idx=0; idx<N; idx++){
		clust_sizes[clust_assn[idx]] += 1;
	}
}

void print_centroids(float* centroids, int K){
	cout<<"The centroids are:\n";
    	for(int l=0; l<K; l++){
        	cout<<"centroid: " <<l<<": (" <<centroids[2*l]<<", "<<centroids[2*l+1]<<")"<<endl;
		}
}

//---------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------CLUSTER ASSIGNMENT CODE---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------

_global_ void _SharedMemorykMedoidsClusterAssignmentKernel(float* datapoints, int* clust_assn, float* centroids, int N, int K)
{
    _shared_ float shared_centroids[D * TPB];
    // Copy centroids to shared memory
    int tid = threadIdx.x;
    for (int c = tid; c < D * K; c += TPB) {
        shared_centroids[c] = centroids[c];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float min_dist = FLT_MAX;
        int closest_centroid = -1;

        for (int c = 0; c < K; ++c)
        {
            float dist = distance(shared_centroids[2 * c], shared_centroids[2 * c + 1],
                                  datapoints[2 * idx], datapoints[2 * idx + 1]);
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_centroid = c;
            }
        }
        clust_assn[idx] = closest_centroid;
    }
}

__global__ void kMedoidsClusterAssignmentKernel(float* datapoints, int* clust_assn, float* centroids, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float min_dist = FLT_MAX;
        int closest_centroid = -1;

        // Compute distance for each centroid in parallel
        for (int c = 0; c < K; ++c)
        {
            float dist = sqrtf((centroids[2 * c] - datapoints[2 * idx]) * (centroids[2 * c] - datapoints[2 * idx]) +
                               (centroids[2 * c + 1] - datapoints[2 * idx + 1]) * (centroids[2 * c + 1] - datapoints[2 * idx + 1]));
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_centroid = c;
            }
        }

        clust_assn[idx] = closest_centroid;
    }
}

// This function handles memory allocation on device
// Also handles data transfer from CPU to GPU
// So the input data to this function should be all variables on host
void kMedoidsClusterAssignment(float* datapoints, int* clust_assn, float* centroids, int N, int K, int* clust_sizes)
{
		//print_centroids(centroids, K);
    float* device_datapoints;
    int* device_clust_assn;
    float* device_centroids;

    cudaMalloc((void**)&device_datapoints, sizeof(float) * 2 * N);
    cudaMalloc((void**)&device_clust_assn, sizeof(int) * N);
    cudaMalloc((void**)&device_centroids, sizeof(float) * 2 * K);

    cudaMemcpy(device_datapoints, datapoints, sizeof(float) * 2 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids, sizeof(float) * 2 * K, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    kMedoidsClusterAssignmentKernel<<<numBlocks, blockSize>>>(device_datapoints, device_clust_assn, device_centroids, N, K);
		_updateClusterSizes(clust_sizes, clust_assn, N, K);

    cudaMemcpy(clust_assn, device_clust_assn, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(device_datapoints);
    cudaFree(device_clust_assn);
    cudaFree(device_centroids);
}
//-------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------CENTROID UPDATE CODE------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------

__device__ float _SharedMemorydissimilarities(float* datapoints, int* clust_assn, int idx, int N, int c)
{
    float totalDissimilarity = 0.0;
    __shared__ float sharedData[256];  // Shared memory for caching data
    int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += gridSize)
    {
        if (clust_assn[i] == c && i != idx)
        {
            float dist = distance(datapoints[2 * idx], datapoints[2 * idx + 1],
                                  datapoints[2 * i], datapoints[2 * i + 1]);
            sharedData[tid] += dist;
        }
    }

    __syncthreads();  // Synchronize threads within the block

    // Reduce within block
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(tid < offset)
        {
            sharedData[tid] += sharedData[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        totalDissimilarity = sharedData[0];
    }
    return totalDissimilarity;
}

__device__ float _dissimilarities(float* datapoints, 
            int* clust_assn, int idx, int N, int c)
{
    float totalDissimilarity = 0.0;
    for (int otherIdx = 0; otherIdx < N; ++otherIdx) {
        if (clust_assn[otherIdx] == c && otherIdx != idx) 
        {
            float dist = distance(
                datapoints[2 * idx], 
                datapoints[2 * idx + 1],
                datapoints[2 * otherIdx], 
                datapoints[2 * otherIdx + 1]);
            totalDissimilarity += dist;
        }
    }
    return totalDissimilarity;
}


__global__ void _kMedoidsCentroidUpdateKernel(float* datapoints, 
    int* clust_assn, float* centroids, 
    int* clust_sizes, int N, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K && clust_sizes[c] > 0)
    {
        float minDissimilarity = FLT_MAX;
        int medoidIdx = -1;
        for (int idx = 0; idx < N; ++idx)
        {
            if (clust_assn[idx] == c)
            {
                float totalDissimilarity = _dissimilarities(
                    datapoints, clust_assn, idx, N, c);
                if (totalDissimilarity < minDissimilarity)
                {
                    minDissimilarity = totalDissimilarity;
                    medoidIdx = idx;
                }
            }
        }
        centroids[2 * c] = datapoints[2 * medoidIdx];
        centroids[2 * c + 1] = datapoints[2 * medoidIdx + 1];
    }
}

// This function handles memory allocation on device
// Also handles data transfer from CPU to GPU
// So the input data to this function should be all variables on host
void _kMedoidsCentroidUpdate(float* datapoints, int* clust_assn, float* centroids, int* clust_sizes, int N, int K)
{
    float* device_datapoints;
    int* device_clust_assn;
    float* device_centroids;
    int* device_clust_sizes;

    cudaMalloc((void**)&device_datapoints, sizeof(float) * 2 * N);
    cudaMalloc((void**)&device_clust_assn, sizeof(int) * N);
    cudaMalloc((void**)&device_centroids, sizeof(float) * 2 * K);
    cudaMalloc((void**)&device_clust_sizes, sizeof(int) * K);

    cudaMemcpy(device_datapoints, datapoints, sizeof(float) * 2 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_clust_assn, clust_assn, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids, sizeof(float) * 2 * K, cudaMemcpyHostToDevice);
    cudaMemcpy(device_clust_sizes, clust_sizes, sizeof(int) * K, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (K + blockSize - 1) / blockSize;

    _kMedoidsCentroidUpdateKernel<<<numBlocks, blockSize>>>(device_datapoints, device_clust_assn, device_centroids, device_clust_sizes, N, K);

    cudaMemcpy(centroids, device_centroids, sizeof(float) * 2 * K, cudaMemcpyDeviceToHost);

    cudaFree(device_datapoints);
    cudaFree(device_clust_assn);
    cudaFree(device_centroids);
    cudaFree(device_clust_sizes);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------

bool Read_from_file(float* datapoints, std::string input_file = "points_100.txt"){

	FILE* file = fopen(input_file.c_str(), "r");
	if(file != NULL){
        cout <<"The initial points are: \n";
		int d = 0;
        while ( !feof(file) )
		{
			float x, y;

            // break if you will not find a pair
			if(fscanf(file, "%f %f", &x, &y )!= 2){
				break;
			}
            datapoints[2*d] = x;
			datapoints[2*d+1] = y;
			d = d + 1;
		}
        fclose(file);
		return 0;

	}else{
		cerr<<"Error during opening file \n";
		return -1;
	}
};


void centroid_init(float* datapoints, float* centroids, int N, int K){
	for (int c=0; c<K; c++){
		int temp = (N/K);
		int idx_r = rand()%temp;

		// for each cluster choosing randomly the centroid
		// fixed it by multiplying by 2
		centroids[2*c]= datapoints[(c*temp +idx_r)*2];
		centroids[2*c+1] = datapoints[(c*temp +idx_r)*2+1];
	}
};

// size is the number of points in the chosen array,
void write2csv(float* points, std::string outfile_name, int size)
{
    std::ofstream outfile;
    outfile.open(outfile_name);
    outfile << "x,y\n";  // name of the columns

    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "\n";
    }
}


void write2csv_clust(float* points, int* clust_assn, std::string outfile_name, int size)
{
    std::ofstream outfile;
    outfile.open(outfile_name);
    outfile << "x,y,c\n";  // name of the columns

    // writing of the coordinates (even are x's, odd are y's) and their relative cluster.
    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "," << clust_assn[i] << "\n";
    }
}


int main()
{
	std::string input_file;
	std::string outdir;

	int N, K, MAX_ITER;
	input_file = "4_clus_1000_points.txt";
	outdir = "./";
	K = 4;
	N = K*1000;
	MAX_ITER = 2;

	//allocation of memory on the device
	float *d_datapoints = 0;
	int *d_clust_assn = 0;
	float *d_centroids = 0;
	int *d_clust_sizes = 0;


	// allocation of memory in host
	float *h_centroids = (float*)malloc(D*K*sizeof(float));
	float *h_datapoints = (float*)malloc(D*N*sizeof(float));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));

	srand(5);

	//initialize datapoints
	Read_from_file(h_datapoints, input_file);

	//initialize centroids
	centroid_init(h_datapoints, h_centroids, N, K);
	write2csv(h_centroids, outdir+"initial_cenroids.csv", K);

	printf("Initialization of %d centroids: \n", K);
	for(int c=0; c<K; ++c){
		printf("(%f, %f)\n", h_centroids[2*c], h_centroids[2*c+1]);
	}

	//initialize centroids counter for each clust
   	for(int c = 0; c < K; ++c){
		  h_clust_sizes[c] = 0;
	}

	int cur_iter = 0;

	float time_assignments = 0;         // total time of ROI ASSIGNMENT
	float time_copy= 0;                 // total time of ROI CP
	float time_copy_2= 0;               // total time of ROI CP2

	// ROI WHILE - while cycle (duration of all epochs)
	auto start_while = high_resolution_clock::now();
	while(cur_iter < MAX_ITER)
	{

		// ROI ASSIGNMENT - cluster assignment
		auto start = high_resolution_clock::now();
		//kMedoidsClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints, d_clust_assn, d_centroids, N, K);
		kMedoidsClusterAssignment(h_datapoints, h_clust_assn, h_centroids, N, K, h_clust_sizes);
		auto stop = high_resolution_clock::now();

		// get the time of ROI ASSIGNMENT
		auto duration = duration_cast<microseconds>(stop - start);
		float temp = duration.count();
		time_assignments = time_assignments + temp;

		//call centroid update kernel
		_kMedoidsCentroidUpdate(h_datapoints, h_clust_assn, h_centroids, h_clust_sizes, N, K);

		cur_iter += 1;

		// resetting cluster sizes
		memset(h_clust_sizes, 0, K*sizeof(int));
	}

	auto stop_while = high_resolution_clock::now();

  	// get and print the time of ROI WHILE
	auto duration_while = duration_cast<microseconds>(stop_while - start_while);
	float temp_while = duration_while.count();
	cout << "Time taken by " << MAX_ITER << " iterations is: "<< temp_while << " microseconds" << endl;

  	// print the average time of ROI ASSIGNMENT during each iteration
	time_assignments = time_assignments/MAX_ITER;
	cout << "Time taken by kMedoidsClusterAssignment: "<< time_assignments << " microseconds" << endl;

	cudaMemcpy(h_centroids, d_centroids, D*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_clust_assn, d_clust_assn, N*sizeof(int), cudaMemcpyDeviceToHost);

	print_centroids(h_centroids, K);

	// Naming for the output files
	std::string outfile_points = outdir + "/datapoints.csv";
	std::string outfile_centroids = outdir + "/centroids.csv";
	std::string outfile_clust = outdir + "/clusters.csv";

	// Writing to files
	write2csv(h_datapoints, outfile_points, N);
	write2csv(h_centroids, outfile_centroids, K);
	write2csv_clust(h_datapoints, h_clust_assn, outfile_clust, N);

	// Freeing memory on device
	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_clust_sizes);

	// Freeing memory on host
	free(h_centroids);
	free(h_datapoints);
	free(h_clust_sizes);
	free(h_clust_assn);

	return 0;
}