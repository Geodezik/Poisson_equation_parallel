#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>

#define SAFE_CUDA(call) do {cudaError_t err__ = (call); if (err__ != cudaSuccess) {fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); MPI_Abort(MPI_COMM_WORLD, 1); }} while(0)

static int threads_per_block = 256;

__global__ void pack_boundaries_kernel(const double* __restrict__ vec, int Nx, int Ny, double* __restrict__ send_left, double* __restrict__ send_right, double* __restrict__ send_down, double* __restrict__ send_up) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Ny) {
		int idxL = tid;
		int idxR = (Nx - 1) * Ny + tid;
		send_left[tid] = vec[idxL];
		send_right[tid] = vec[idxR];
	}
	if (tid < Nx) {
		int idxD = tid * Ny;
		int idxU = (tid + 1) * Ny - 1;
		send_down[tid] = vec[idxD];
		send_up[tid] = vec[idxU];
	}
}

__global__ void build_Aw_kernel(const double* __restrict__ w, double* __restrict__ Aw, const double* __restrict__ a, const double* __restrict__ b,
const double* __restrict__ from_left, const double* __restrict__ from_right, const double* __restrict__ from_bot, const double* __restrict__ from_top, int Nx, int Ny, double h1, double h2) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int total = Nx * Ny;
	if (tid >= total) return;
	int i0 = tid / Ny;
	int j0 = tid % Ny;
	int i = i0 + 1;
	int j = j0 + 1;

	double wij = w[(i - 1) * Ny + (j - 1)];
	double wi1j = (i + 1 <= Nx) ? w[i * Ny + (j - 1)] : from_right[j - 1];
	double wi_1j = (i - 1 >= 1) ? w[(i - 2) * Ny + (j - 1)] : from_left[j - 1];
	double wij1 = (j + 1 <= Ny) ? w[(i - 1) * Ny + j] : from_top[i - 1];
	double wij_1 = (j - 1 >= 1) ? w[(i - 1) * Ny + (j - 2)] : from_bot[i - 1];

	double aij = a[i * (Ny + 1) + j];
	double ai1j = a[(i + 1) * (Ny + 1) + j];
	double bij = b[i * (Ny + 2) + j];
	double bij1 = b[i * (Ny + 2) + (j + 1)];

	double term_x = (aij * (wij - wi_1j) - ai1j * (wi1j - wij)) / (h1 * h1);
	double term_y = (bij * (wij - wij_1) - bij1 * (wij1 - wij)) / (h2 * h2);
	Aw[(i - 1) * Ny + (j - 1)] = term_x + term_y;
}

__global__ void wr_update_kernel(double* __restrict__ w, const double* __restrict__ p, double* __restrict__ r, const double* __restrict__ Ap, double alpha, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) {
		w[tid] += alpha * p[tid];
		r[tid] -= alpha * Ap[tid];
	}
}

__global__ void p_update_kernel(double* __restrict__ p, const double* __restrict__ z, double beta, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) p[tid] = z[tid] + beta * p[tid];
}

__global__ void vec_div_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) c[tid] = a[tid] / b[tid];
}

__global__ void vec_sub_kernel(double* __restrict__ c, const double* __restrict__ a, const double* __restrict__ b, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) c[tid] = a[tid] - b[tid];
}

double dot_device(double* u, double* v, int n) {
	auto U = thrust::device_pointer_cast(u);
	auto V = thrust::device_pointer_cast(v);
	return thrust::inner_product(U, U + n, V, 0.0);
}

struct Boundaries {
	std::vector<double> send_left, send_right, send_down, send_up;
	std::vector<double> from_left, from_right, from_bot, from_top;
	double *d_send_left=nullptr, *d_send_right=nullptr, *d_send_down=nullptr, *d_send_up=nullptr;
	double *d_from_left=nullptr, *d_from_right=nullptr, *d_from_bot=nullptr, *d_from_top=nullptr;
	void init(int Nx, int Ny) {
		send_left.resize(Ny); send_right.resize(Ny);
		send_down.resize(Nx); send_up.resize(Nx);
		from_left.resize(Ny); from_right.resize(Ny);
		from_bot.resize(Nx); from_top.resize(Nx);
		SAFE_CUDA(cudaMalloc(&d_send_left, Ny*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_send_right, Ny*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_send_down, Nx*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_send_up, Nx*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_from_left, Ny*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_from_right, Ny*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_from_bot, Nx*sizeof(double)));
		SAFE_CUDA(cudaMalloc(&d_from_top, Nx*sizeof(double)));
	}
	void destroy() {
		SAFE_CUDA(cudaFree(d_send_left)); SAFE_CUDA(cudaFree(d_send_right));
		SAFE_CUDA(cudaFree(d_send_down)); SAFE_CUDA(cudaFree(d_send_up));
		SAFE_CUDA(cudaFree(d_from_left)); SAFE_CUDA(cudaFree(d_from_right));
		SAFE_CUDA(cudaFree(d_from_bot)); SAFE_CUDA(cudaFree(d_from_top));
	}
};

void device_sendrecv_boundaries(const double* d_vec, int Nx, int Ny, Boundaries& boundaries, int rank_left, int rank_right, int rank_down, int rank_up) {
	int blocks = (std::max(Nx,Ny) + threads_per_block - 1) / threads_per_block;
	pack_boundaries_kernel<<<blocks, threads_per_block>>>(d_vec, Nx, Ny, boundaries.d_send_left, boundaries.d_send_right, boundaries.d_send_down, boundaries.d_send_up);
	SAFE_CUDA(cudaDeviceSynchronize());
	SAFE_CUDA(cudaMemcpy(boundaries.send_left.data(), boundaries.d_send_left, Ny*sizeof(double), cudaMemcpyDeviceToHost));
	SAFE_CUDA(cudaMemcpy(boundaries.send_right.data(), boundaries.d_send_right, Ny*sizeof(double), cudaMemcpyDeviceToHost));
	SAFE_CUDA(cudaMemcpy(boundaries.send_down.data(), boundaries.d_send_down, Nx*sizeof(double), cudaMemcpyDeviceToHost));
	SAFE_CUDA(cudaMemcpy(boundaries.send_up.data(), boundaries.d_send_up, Nx*sizeof(double), cudaMemcpyDeviceToHost));
	MPI_Status st;
	MPI_Sendrecv(boundaries.send_left.data(), Ny, MPI_DOUBLE, rank_left, 101,
	boundaries.from_right.data(), Ny, MPI_DOUBLE, rank_right, 101, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(boundaries.send_right.data(), Ny, MPI_DOUBLE, rank_right, 102,
	boundaries.from_left.data(), Ny, MPI_DOUBLE, rank_left, 102, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(boundaries.send_down.data(), Nx, MPI_DOUBLE, rank_down, 201,
	boundaries.from_top.data(), Nx, MPI_DOUBLE, rank_up, 201, MPI_COMM_WORLD, &st);
	MPI_Sendrecv(boundaries.send_up.data(), Nx, MPI_DOUBLE, rank_up, 202,
	boundaries.from_bot.data(), Nx, MPI_DOUBLE, rank_down, 202, MPI_COMM_WORLD, &st);
	SAFE_CUDA(cudaMemcpy(boundaries.d_from_left, boundaries.from_left.data(), Ny*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(boundaries.d_from_right, boundaries.from_right.data(), Ny*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(boundaries.d_from_bot, boundaries.from_bot.data(), Nx*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(boundaries.d_from_top, boundaries.from_top.data(), Nx*sizeof(double), cudaMemcpyHostToDevice));
}

std::vector<double> solve_linear_system_cuda(const std::vector<double>& B, const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& D, int Nx, int Ny, int M, int N, double h1, double h2, int maxit, double tol, int rank, int rank_left, int rank_right, int rank_down, int rank_up, double& t_loop, double& t_comm) {
	t_comm = 0.0;

	int devCount = 0;
	cudaError_t e = cudaGetDeviceCount(&devCount);
	if (e != cudaSuccess) devCount = 0;
	if (devCount > 0) {
		int dev = rank % devCount;
		SAFE_CUDA(cudaSetDevice(dev));
	}

	const int n = Nx * Ny;
	double *d_B=nullptr, *d_a=nullptr, *d_b=nullptr, *d_D=nullptr, *d_w_good = nullptr;;
	double *d_w=nullptr, *d_r=nullptr, *d_z=nullptr, *d_p=nullptr, *d_Ap=nullptr, *d_Aw_tmp=nullptr;
	SAFE_CUDA(cudaMalloc(&d_B, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_a, (Nx+2)*(Ny+1)*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_b, (Nx+1)*(Ny+2)*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_D, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_w, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_r, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_z, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_p, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_Ap, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_Aw_tmp, n*sizeof(double)));
	SAFE_CUDA(cudaMalloc(&d_w_good, n*sizeof(double)));
	double t_c0 = MPI_Wtime();
	SAFE_CUDA(cudaMemcpy(d_B, B.data(), n*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_a, a.data(), (Nx+2)*(Ny+1)*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_b, b.data(), (Nx+1)*(Ny+2)*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_D, D.data(), n*sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_r, d_B, n*sizeof(double), cudaMemcpyDeviceToDevice));
	t_comm += MPI_Wtime() - t_c0;

	SAFE_CUDA(cudaMemset(d_w, 0, n*sizeof(double)));

	int blocksN = (n + threads_per_block - 1) / threads_per_block;
	vec_div_kernel<<<blocksN, threads_per_block>>>(d_z, d_r, d_D, n);
	t_c0 = MPI_Wtime();
	SAFE_CUDA(cudaMemcpy(d_p, d_z, n*sizeof(double), cudaMemcpyDeviceToDevice));
	t_comm += MPI_Wtime() - t_c0;
	SAFE_CUDA(cudaDeviceSynchronize());

	double rz = dot_device(d_r, d_z, n);
	double rz_global = 0.0;
	t_c0 = MPI_Wtime();
	MPI_Allreduce(&rz, &rz_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	t_comm += MPI_Wtime() - t_c0;

	Boundaries boundaries;
	boundaries.init(Nx, Ny);

	std::vector<double> w_final(n, 0.0);
	double J_prev = 0.0; int have_J_prev = 0; int restarted_prev = 0;

	int it = 0;
	double t_loop0 = MPI_Wtime();
	for (; it < maxit; ++it) {
		double t_c0 = MPI_Wtime();
		device_sendrecv_boundaries(d_p, Nx, Ny, boundaries, rank_left, rank_right, rank_down, rank_up);
		t_comm += MPI_Wtime() - t_c0;
		int blocks = (n + threads_per_block - 1)/threads_per_block;
		build_Aw_kernel<<<blocks, threads_per_block>>>(d_p, d_Ap, d_a, d_b, boundaries.d_from_left, boundaries.d_from_right, boundaries.d_from_bot, boundaries.d_from_top, Nx, Ny, h1, h2);
		SAFE_CUDA(cudaDeviceSynchronize());

		double pAp = dot_device(d_p, d_Ap, n);
		double pAp_global = 0.0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&pAp, &pAp_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		double alpha = rz_global / pAp_global;

		wr_update_kernel<<<blocksN, threads_per_block>>>(d_w, d_p, d_r, d_Ap, alpha, n);
		SAFE_CUDA(cudaDeviceSynchronize());

		double pp = dot_device(d_p, d_p, n);
		double pp_global = 0.0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&pp, &pp_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		double step_norm = std::abs(alpha) * std::sqrt(pp_global * (h1*h2));
		int stop_flag = (step_norm < tol);
		int stop_any = 0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&stop_flag, &stop_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		if (stop_any) { ++it; break; }

		double Bw_local = dot_device(d_B, d_w, n);
		double rw_local = dot_device(d_r, d_w, n);
		double Bw = 0.0, rw = 0.0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&Bw_local, &Bw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&rw_local, &rw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		double Jk = -0.5 * ((Bw + rw) * (h1*h2));

		if (!have_J_prev) {
		    J_prev = Jk;
		    t_c0 = MPI_Wtime();
		    SAFE_CUDA(cudaMemcpy(d_w_good, d_w, n*sizeof(double), cudaMemcpyDeviceToDevice));
		    t_comm += MPI_Wtime() - t_c0;
		    have_J_prev = 1;
		    restarted_prev = 0;
		} else if (Jk > J_prev) {
		    int restart_any = 1, tmp=0;
		    t_c0 = MPI_Wtime();
		    MPI_Allreduce(&restart_any, &tmp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		    SAFE_CUDA(cudaMemcpy(d_w, d_w_good, n*sizeof(double), cudaMemcpyDeviceToDevice));
		    t_comm += MPI_Wtime() - t_c0;

		    t_c0 = MPI_Wtime();
		    device_sendrecv_boundaries(d_w, Nx, Ny, boundaries, rank_left, rank_right, rank_down, rank_up);
		    t_comm += MPI_Wtime() - t_c0;

		    build_Aw_kernel<<<blocks, threads_per_block>>>(d_w, d_Aw_tmp, d_a, d_b,
			boundaries.d_from_left, boundaries.d_from_right,
			boundaries.d_from_bot, boundaries.d_from_top, Nx, Ny, h1, h2);
		    SAFE_CUDA(cudaDeviceSynchronize());

		    vec_sub_kernel<<<blocksN, threads_per_block>>>(d_r, d_B, d_Aw_tmp, n);
		    SAFE_CUDA(cudaDeviceSynchronize());
		    vec_div_kernel<<<blocksN, threads_per_block>>>(d_z, d_r, d_D, n);
		    SAFE_CUDA(cudaMemcpy(d_p, d_z, n*sizeof(double), cudaMemcpyDeviceToDevice));
		    SAFE_CUDA(cudaDeviceSynchronize());
		    double rz_new = dot_device(d_r, d_z, n);
		    MPI_Allreduce(&rz_new, &rz_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		    J_prev = -0.5 * ((dot_device(d_B,d_w,n) + dot_device(d_r,d_w,n)) * (h1*h2));
		    if (restarted_prev) break;
		    restarted_prev = 1;
		    continue;
		} else {
		    J_prev = Jk;
		    SAFE_CUDA(cudaMemcpy(d_w_good, d_w, n*sizeof(double), cudaMemcpyDeviceToDevice));
		    restarted_prev = 0;
		}

		vec_div_kernel<<<blocksN, threads_per_block>>>(d_z, d_r, d_D, n);
		SAFE_CUDA(cudaDeviceSynchronize());
		double rz_new_local = dot_device(d_r, d_z, n);
		double rz_new = 0.0;
		t_c0 = MPI_Wtime();
		MPI_Allreduce(&rz_new_local, &rz_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		t_comm += MPI_Wtime() - t_c0;
		double beta = rz_new / rz_global;
		p_update_kernel<<<blocksN, threads_per_block>>>(d_p, d_z, beta, n);
		SAFE_CUDA(cudaDeviceSynchronize());
		rz_global = rz_new;
	}
	
	t_loop = MPI_Wtime() - t_loop0;
	
	if (rank == 0) std::cout << "Finished in " << it << " iterations" << std::endl;

	t_c0 = MPI_Wtime();
	SAFE_CUDA(cudaMemcpy(w_final.data(), d_w, n*sizeof(double), cudaMemcpyDeviceToHost));
	t_comm += MPI_Wtime() - t_c0;

	boundaries.destroy();
	SAFE_CUDA(cudaFree(d_B)); SAFE_CUDA(cudaFree(d_a)); SAFE_CUDA(cudaFree(d_b)); SAFE_CUDA(cudaFree(d_D));
	SAFE_CUDA(cudaFree(d_w)); SAFE_CUDA(cudaFree(d_r)); SAFE_CUDA(cudaFree(d_z)); SAFE_CUDA(cudaFree(d_p));
	SAFE_CUDA(cudaFree(d_Ap)); SAFE_CUDA(cudaFree(d_Aw_tmp)); SAFE_CUDA(cudaFree(d_w_good));

	return w_final;
}
