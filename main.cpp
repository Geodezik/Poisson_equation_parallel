#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <mpi.h>

std::vector<double> solve_linear_system_cuda(const std::vector<double>& B, const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& D, int Nx, int Ny, int M, int N, double h1, double h2, int maxit, double tol, int rank, int rank_left, int rank_right, int rank_down, int rank_up, double& t_loop, double& t_comm);

struct Params {
	int M = 128;
	int N = 128;
	double A1 = 0.0, B1 = 1.0, A2 = -1.0, B2 = 1.0;
	double tol = 1e-8;
	int maxit = 10000;
	std::string out = "solution.csv";
	int threads = 1;
};

static int rank = 0, world_size = 1;
static int i_0 = 1, i_1 = 1, j_0 = 1, j_1 = 1;
static int Nx_loc = 0, Ny_loc = 0;
static int rank_left = MPI_PROC_NULL, rank_right = MPI_PROC_NULL, rank_down = MPI_PROC_NULL, rank_up = MPI_PROC_NULL;

int idxW(int i, int j, int Ny) { return (i - 1) * Ny + (j - 1); }
int idxAB(int i, int j, int N) { return i * (N + 1) + j; }

void parseArgs(int argc, char** argv, Params& P) {
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--M") && i + 1 < argc) P.M = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--N") && i + 1 < argc) P.N = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tol") && i + 1 < argc) P.tol = std::atof(argv[++i]);
		else if (!strcmp(argv[i], "--maxit") && i + 1 < argc) P.maxit = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--out") && i + 1 < argc) P.out = argv[++i];
		else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {P.threads = std::atoi(argv[++i]); std::cerr << "Warning: --threads is ignored" << std::endl;}
		else throw std::runtime_error("Invalid argument(s) detected");
	}
}

void decompose_2d(int Nx, int Ny, int P, int& Px, int& Py, std::vector<int>& x_off, std::vector<int>& y_off) {
	Px = 1; Py = P;
	double bestA = 1e100;
	for (int d = 1; d * d <= P; ++d) {
		if (P % d) continue;
		int cand_px[2] = { d, P / d };
		int cand_py[2] = { P / d, d };
		for (int k = 0; k < 2; ++k) {
			int px = cand_px[k];
			int py = cand_py[k];
			if (px > Nx || py > Ny) continue;
			int bx = Nx / px, rx = Nx % px;
			int by = Ny / py, ry = Ny % py;
			if (bx == 0 || by == 0) continue;
			double A = std::max((double)(bx + (rx > 0)) / (double)by, (double)(by + (ry > 0)) / (double)bx);
			if ((bestA > 2.0 && A <= 2.0) || (A < bestA) || (A == bestA && std::abs(px - py) < std::abs(Px - Py))) {
				bestA = A;
				Px = px;
				Py = py;
			}
		}
	}
	if (Px * Py != P || Px > Nx || Py > Ny) {
		Px = 1; Py = P;
		for (int d = 1; d <= Nx && d <= P; ++d) {
			if (P % d) continue;
			int py = P / d;
			if (py <= Ny) { Px = d; Py = py; }
		}
	}
	x_off.assign(Px + 1, 0);
	y_off.assign(Py + 1, 0);
	int bx = Nx / Px, rx = Nx % Px;
	for (int i = 0; i < Px; ++i) x_off[i + 1] = x_off[i] + bx + (i < rx);
	int by = Ny / Py, ry = Ny % Py;
	for (int j = 0; j < Py; ++j) y_off[j + 1] = y_off[j] + by + (j < ry);
}

std::vector<double> build_a(int M, int N, double A1, double A2, double h1, double h2, double inv_eps) {
	std::vector<double> a((Nx_loc + 2) * (Ny_loc + 1), 0.0);
	for (int i = 1; i <= Nx_loc + 1; ++i) {
		double s_x_lo = std::sqrt(A1 + (i_0 + i - 1.5) * h1);
		for (int j = 1; j <= Ny_loc; ++j) {
			double y_lo = A2 + (j_0 + j - 1.5) * h2;
			double y_hi = A2 + (j_0 + j - 0.5) * h2;
			double low_point = (y_lo > -s_x_lo) ? y_lo : -s_x_lo;
			double high_point = (y_hi < s_x_lo) ? y_hi : s_x_lo;
			double alpha = std::max(high_point - low_point, 0.0) / h2;
			a[idxAB(i, j, Ny_loc)] = alpha + (1.0 - alpha) * inv_eps;
		}
	}
	return a;
}

std::vector<double> build_b(int M, int N, double A1, double A2, double h1, double h2, double inv_eps) {
	std::vector<double> b((Nx_loc + 1) * (Ny_loc + 2), 0.0);
	for (int i = 1; i <= Nx_loc; ++i) {
		double x_lo = A1 + (i_0 + i - 1.5) * h1;
		double x_hi = A1 + (i_0 + i - 0.5) * h1;
		for (int j = 1; j <= Ny_loc + 1; ++j) {
			double y_lo = A2 + (j_0 + j - 1.5) * h2;
			double low_point = (x_lo > y_lo * y_lo) ? x_lo : (y_lo * y_lo);
			double high_point = (x_hi < 1.0) ? x_hi : 1.0;
			double beta = std::max(high_point - low_point, 0.0) / h1;
			b[idxAB(i, j, Ny_loc + 1)] = beta + (1.0 - beta) * inv_eps;
		}
	}
	return b;
}

std::vector<double> build_F(int Nx, int Ny, int M, int N, double A1, double A2, double h1, double h2, int int_grid_steps = 8) {
	std::vector<double> F(Nx_loc * Ny_loc, 0.0);
	double dy = h2 / int_grid_steps;
	for (int i = 1; i <= Nx_loc; ++i) {
		double x_lo = A1 + (i_0 + i - 1.5) * h1;
		double x_hi = A1 + (i_0 + i - 0.5) * h1;
		for (int j = 1; j <= Ny_loc; ++j) {
			double y_lo = A2 + (j_0 + j - 1.5) * h2;
			double S = 0.0;
			for (int s = 0; s < int_grid_steps; ++s) {
				double y_cur = y_lo + (s + 0.5) * dy;
				double low_point = (x_lo > y_cur * y_cur) ? x_lo : (y_cur * y_cur);
				double high_point = (x_hi < 1.0) ? x_hi : 1.0;
				S += std::max(high_point - low_point, 0.0) * dy;
			}
			F[idxW(i, j, Ny_loc)] = S / (h1 * h2);
		}
	}
	return F;
}

std::vector<double> build_D(int Nx, int Ny, int M, int N, double h1, double h2, std::vector<double>& a, std::vector<double>& b) {
	std::vector<double> D(Nx_loc * Ny_loc, 0.0);
	for (int i = 1; i <= Nx_loc; ++i)
		for (int j = 1; j <= Ny_loc; ++j)
			D[idxW(i, j, Ny_loc)] = (a[idxAB(i, j, Ny_loc)] + a[idxAB(i+1, j, Ny_loc)]) / (h1 * h1) + (b[idxAB(i, j, Ny_loc + 1)] + b[idxAB(i, j+1, Ny_loc + 1)]) / (h2 * h2);
	return D;
}

void save_solution_csv(std::string& filename, std::vector<double>& w_full, int M, int N, double A1, double A2, double h1, double h2) {
	std::ofstream fout(filename.c_str());
	fout << std::setprecision(12);
	fout << "x,y,v\n";
	for (int i = 0; i <= M; ++i) {
		double x = A1 + i * h1;
		for (int j = 0; j <= N; ++j) {
			double y = A2 + j * h2;
			double vij = 0.0;
			if (i >= 1 && i <= M - 1 && j >= 1 && j <= N - 1) vij = w_full[idxW(i, j, N - 1)];
			fout << x << "," << y << "," << vij << "\n";
		}
	}
	fout.close();
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double t_global = MPI_Wtime();

	Params P;
	parseArgs(argc, argv, P);

	const int M = P.M, N = P.N;
	const double A1 = P.A1, B1 = P.B1, A2 = P.A2, B2 = P.B2;
	const double h1 = (B1 - A1) / M;
	const double h2 = (B2 - A2) / N;
	const double h = std::max(h1, h2);
	const double eps = h * h;
	const double inv_eps = 1.0 / eps;
	const int Nx = M - 1;
	const int Ny = N - 1;

	int Px = 1, Py = 1;
	std::vector<int> x_off, y_off;
	decompose_2d(Nx, Ny, world_size, Px, Py, x_off, y_off);
	i_0 = x_off[rank % Px] + 1;
	i_1 = x_off[rank % Px + 1];
	j_0 = y_off[rank / Px] + 1;
	j_1 = y_off[rank / Px + 1];
	Nx_loc = i_1 - i_0 + 1;
	Ny_loc = j_1 - j_0 + 1;

	if (Nx_loc <= 0 || Ny_loc <= 0) throw std::runtime_error("Empty subdomain!");

	rank_left = (rank % Px > 0) ? rank - 1 : MPI_PROC_NULL;
	rank_right = (rank % Px + 1 < Px) ? rank + 1 : MPI_PROC_NULL;
	rank_down = (rank / Px > 0) ? rank - Px : MPI_PROC_NULL;
	rank_up = (rank / Px + 1 < Py) ? rank + Px : MPI_PROC_NULL;

	double t_init = MPI_Wtime();

	std::vector<double> a = build_a(M, N, A1, A2, h1, h2, inv_eps);
	std::vector<double> b = build_b(M, N, A1, A2, h1, h2, inv_eps);
	std::vector<double> F = build_F(Nx_loc, Ny_loc, M, N, A1, A2, h1, h2, 8);
	std::vector<double> D = build_D(Nx_loc, Ny_loc, M, N, h1, h2, a, b);

	t_init = MPI_Wtime() - t_init;

	double t_loop = 0.0, t_comm = 0.0;
	std::vector<double> w = solve_linear_system_cuda(F, a, b, D, Nx_loc, Ny_loc, M, N, h1, h2, P.maxit, P.tol, rank, rank_left, rank_right, rank_down, rank_up, t_loop, t_comm);

	t_global = MPI_Wtime() - t_global;

	auto reduce_max = [&](double v) {
		double r = 0.0;
		MPI_Reduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		return r;
	};

	double t_loop_max = reduce_max(t_loop);
	double t_init_max = reduce_max(t_init);
	double t_comm_max = reduce_max(t_comm);
	double t_global_max = reduce_max(t_global);

	if (rank == 0) {
		std::cout << "Init time: " << t_init_max << " s\n";
		std::cout << "Loop time: " << t_loop_max << " s\n";
		std::cout << "Comm time: " << t_comm_max << " s\n";
		std::cout << "Global time " << t_global_max << " s\n";
	}

	MPI_Finalize();
	return 0;
}

