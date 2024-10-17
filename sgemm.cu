/* sgemm.cu */
#include <iostream>
#include <cstdlib>

/* Function initialization */
float** init_array(int rows, int cols);
void free_array(float** arr, int rows);
bool same_array(float** arr1, float** arr2, int rows, int cols);
float** do_h_mat_mul(float** A, float** B, int m, int k, int n);
float** do_d_mat_mul_element(float** A, float** B, int m, int k, int n);
float** do_d_mat_mul_row(float** A, float** B, int m, int k, int n);
float** do_d_mat_mul_col(float** A, float** B, int m, int k, int n);
void h_mat_mul(float** A, float** B, float** C, int m, int k, int n);
__global__ void d_mat_mul_element(const float* A, const float* B, float* C, int m, int k, int n);
__global__ void d_mat_mul_row(const float* A, const float* B, float* C, int m, int k, int n);
__global__ void d_mat_mul_col(const float* A, const float* B, float* C, int m, int k, int n);

/* Main function */
int main(int argc, char* argv[]){
    // Check if correct number of arguments
    if(argc != 4){
        std::cerr << "Wrong number of inputs\n" << "Try './sgemm <m> <k> <n>'\n";
        return -1;
    }
    
    // Check if dimensions are valid
    int m = std::atoi(argv[1]), k = std::atoi(argv[2]), n = std::atoi(argv[3]);
    if(m < 1 || k < 1 || n < 1){
        std::cerr << "Dimensions of m, k, n have to be at least 1\n";
        return -1;
    }
    
    // Create random matrices A and B
    float** A = init_array(m, k);
    float** B = init_array(k, n);

    // Run the matrix multiplications
    float** CPU_C = do_h_mat_mul(A, B, m, k, n);
    float** GPU_C_ELEMENT = do_d_mat_mul_element(A, B, m, k, n);
    float** GPU_C_ROW = do_d_mat_mul_row(A, B, m, k, n);
    
    // Comparing the outputs
    std::cout << "GPU SINGLE ELEMENT AND CPU COMPARISON: " << same_array(CPU_C, GPU_C_ELEMENT, m, n) << "\n";
    std::cout << "GPU SINGLE ROW AND CPU COMPARISON: " << same_array(CPU_C, GPU_C_ROW, m, n) << "\n";

    // Free memory
    free_array(A, m);
    free_array(B, k);
    free_array(CPU_C, m);
    free_array(GPU_C_ELEMENT, m);
    free_array(GPU_C_ROW, m);

    return 0;
}

/* Helper functions */
// Create random 2D array (dynamic allocation)
float** init_array(int rows, int cols){
    std::srand(1);
    float** ret = new float*[rows];
    for(int i = 0; i < rows; ++i){
        ret[i] = new float[cols];
        for(int j = 0; j < cols; ++j){
            ret[i][j] = std::rand() % 100 / 100.0;
        }
    }
    return ret;
}

// Free allocated memory for 2D array
void free_array(float** arr, int rows){
    for(int i = 0; i < rows; ++i){
        delete[] arr[i];
    }
    delete[] arr;
}

// Compares if output is the same
bool same_array(float** arr1, float** arr2, int rows, int cols) {
    float tolerance = 1e-3;
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            if (fabs(arr1[i][j] - arr2[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

/* Functions to prepare data for matrix multiplication */
// Run matrix multiplication on CPU
float** do_h_mat_mul(float** A, float** B, int m, int k, int n){
    float** C = new float*[m];
    for(int i = 0; i < m; ++i)
        C[i] = new float[n]();
    h_mat_mul(A, B, C, m, k, n);
    return C;
}

// GPU single element at a time matrix multiplication
float** do_d_mat_mul_element(float** A, float** B, int m, int k, int n){
    // Flatten 2D arrays into 1D arrays for device use
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    // Copy data from 2D to 1D
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < k; ++j) {
            h_A[i * k + j] = A[i][j];
        }
    }

    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < n; ++j) {
            h_B[i * n + j] = B[i][j];
        }
    }

    // CUDA memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    d_mat_mul_element<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Convert flattened array back to 2D array
    float** C = new float*[m];
    for(int i = 0; i < m; ++i){
        C[i] = new float[n];
        for(int j = 0; j < n; ++j)
            C[i][j] = h_C[i * n + j];
    }

    // Free arrays
    delete[] h_A;
    delete[] h_B;ELEMENT
}

float** do_d_mat_mul_row(float** A, float** B, int m, int k, int n) {
    // Flatten the arrays
    float* h_A = new float[m*k];
    float* h_B = new float[k*n];
    float* h_C = new float[m*n];

    // Copy from 2D to 1D
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < k; ++j) {
            h_A[i * k + j] = A[i][j]; 
        }
    }

    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < n; ++j) {
            h_B[i * n + j] = B[i][j];
        }
    }

    // Memory allocation on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(10);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    // Launch kernel
    d_mat_mul_row<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Convert flattened array back to 2D array
    float** C = new float*[m];
    for(int i = 0; i < m; ++i){
        C[i] = new float[n];
        for(int j = 0; j < n; ++j)
            C[i][j] = h_C[i * n + j];
    }

    // Free arrays
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return C;
}

/* Matrix Multiplication functions */
// CPU matrix multiplication
void h_mat_mul(float** A, float** B, float** C, int m, int k, int n){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            C[i][j] = 0;
            for(int l = 0; l < k; ++l){
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

// GPU matrix multiplication kernel (single thread single element)
__global__ void d_mat_mul_element(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){
        float val = 0;
        for(int e = 0; e < k; ++e){
            val += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = val;
    }
}

// GPU matric multiplication kernel (single thread single row)
__global__ void d_mat_mul_row(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m) {
        for(int col = 0; col < n; ++col) {
            float sum{0};
            for(int e = 0; e < k; ++e) {
                sum += A[row*k+e] * B[e*n+col];
            }
            C[row * n + col] = sum;
        }
    }
}