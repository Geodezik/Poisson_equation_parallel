#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

struct Params {
	int M = 128;
	int N = 128;
	double A1 = 0.0, B1 = 1.0, A2 = -1.0, B2 = 1.0;
	double tol = 1e-8;
	int maxit = 10000;
	std::string out = "solution.csv";
	int threads = 1;
};

int idxW(int i, int j, int Ny) {
	return (i - 1) * Ny + (j - 1);
}

int idxAB(int i, int j, int N) {
	return i * (N + 1) + j;
}

void parseArgs(int argc, char** argv, Params& P) {
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--M") && i + 1 < argc) P.M = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--N") && i + 1 < argc) P.N = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tol") && i + 1 < argc) P.tol = std::atof(argv[++i]);
		else if (!strcmp(argv[i], "--maxit") && i + 1 < argc) P.maxit = std::atoi(argv[++i]);
		else if (!strcmp(argv[i], "--out") && i + 1 < argc) P.out = argv[++i];
		else if (!strcmp(argv[i], "--threads") && i + 1 < argc) P.threads = std::atoi(argv[++i]);
		else throw("Runtime Error: invalid argument(s) detected");
	}
}

std::vector<double> build_a(int M, int N, double A1, double A2, double h1, double h2, double inv_eps)
{
    // Described at the end of (4)
    // a == part of (x_{i-1/2}y_{j-1/2}) -- (x_{i-1/2}y_{j+1/2}) that lays inside of D
    // The idea: "clip" vertical segment parts that are out of parabola
    std::vector<double> a((M + 1) * (N + 1), 0.0);
    for (int i = 1; i <= M; ++i) {
        double s_x_lo = std::sqrt(A1 + (i - 0.5) * h1);
        for (int j = 1; j <= N; ++j) {
            double low_point = std::max(A2 + (j - 0.5) * h2, -s_x_lo);
            double high_point = std::min(A2 + (j + 0.5) * h2, s_x_lo);
            double alpha = std::max(high_point - low_point, 0.0) / h2;
            a[idxAB(i, j, N)] = alpha + (1.0 - alpha) * inv_eps;
        }
    }
    return a;
}

std::vector<double> build_b(int M, int N, double A1, double A2, double h1, double h2, double inv_eps)
{
    // Described at the end of (4)
    // b == part of (x_{i-1/2}y_{j-1/2}) -- (x_{i+1/2}y_{j-1/2}) that lays inside of D
    // The idea is the same but now we're also consdering x == 1 as a border
    std::vector<double> b((M + 1) * (N + 1), 0.0);
    for (int i = 1; i <= M; ++i) {
        double x_lo = A1 + (i - 0.5) * h1;
        double x_hi = A1 + (i + 0.5) * h1;
        for (int j = 1; j <= N; ++j) {
            double y_lo = A2 + (j - 0.5) * h2;
            double low_point = std::max(x_lo, y_lo * y_lo);
            double high_point = std::min(x_hi, 1.0);
            double beta = std::max(high_point - low_point, 0.0) / h1;
            b[idxAB(i, j, N)] = beta + (1.0 - beta) * inv_eps;
        }
    }
    return b;
}

std::vector<double> build_F(int Nx, int Ny, int M, int N, double A1, double A2, double h1, double h2, int int_grid_steps = 8)
{
    // Fij = mes(\Pi_{ij} \intersect D)
    // I use numerical integration and do not approximate borders with a line
    std::vector<double> F(Nx * Ny, 0.0);
    double dy = h2 / int_grid_steps;
    for (int i = 1; i <= Nx; ++i) {
        double x_lo = A1 + (i - 0.5) * h1;
        double x_hi = A1 + (i + 0.5) * h1;
        for (int j = 1; j <= Ny; ++j) {
            double y_lo = A2 + (j - 0.5) * h2;
            double S = 0.0;
            for (int s = 0; s < int_grid_steps; ++s) {
            	// numerical integration (split by y axis into dy slices)
                double y_cur = y_lo + (s + 0.5) * dy;
                double low_point = std::max(x_lo, y_cur * y_cur);
                double high_point = std::min(x_hi, 1.0);
                S += std::max(high_point - low_point, 0.0) * dy;
            }
            F[idxW(i, j, Ny)] = S / (h1 * h2);
        }
    }
    return F;
}

std::vector<double> build_D(int Nx, int Ny, int M, int N, double h1, double h2, std::vector<double>& a, std::vector<double>& b)
{
    std::vector<double> D(Nx * Ny, 0.0);
    for (int i = 1; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            D[idxW(i, j, Ny)] = (a[idxAB(i + 1, j, N)] + a[idxAB(i, j, N)]) / (h1 * h1) + \
            					(b[idxAB(i, j + 1, N)] + b[idxAB(i, j, N)]) / (h2 * h2);
        }
    }
    return D;
}

void build_Aw(std::vector<double>& w, std::vector<double>& Aw, std::vector<double>& a, std::vector<double>& b,
			  int Nx, int Ny, int M, int N, double h1, double h2)
{
	for (int i = 1; i <= Nx; ++i) {
		for (int j = 1; j <= Ny; ++j) {
			// Formula (10) from section (4)
			double wij = w[idxW(i, j, Ny)];
			double wi1j = (i + 1 <= Nx) ? w[idxW(i + 1, j, Ny)] : 0.0;
			double wi_1j = (i - 1 >= 1)  ? w[idxW(i - 1, j, Ny)] : 0.0;
			double wij1  = (j + 1 <= Ny) ? w[idxW(i, j + 1, Ny)] : 0.0;
			double wij_1 = (j - 1 >= 1)  ? w[idxW(i, j - 1, Ny)] : 0.0;
			double aij = a[idxAB(i, j, N)];
			double ai1j = a[idxAB(i + 1, j, N)];
			double bij = b[idxAB(i, j, N)];
			double bij1 = b[idxAB(i, j + 1, N)];
			Aw[idxW(i, j, Ny)] = (aij * (wij - wi_1j) - ai1j * (wi1j - wij)) / (h1 * h1) + \
								 (bij * (wij - wij_1) - bij1 * (wij1 - wij)) / (h2 * h2);
		}
	}
}

double dotE(std::vector<double>& u, std::vector<double>& v, double h1, double h2) {
	double s = 0.0;
    for (int k = 0; k < u.size(); ++k) s += u[k] * v[k];
    return s * (h1 * h2);
}

std::vector<double> solve_linear_system(std::vector<double>& B, std::vector<double>& a, std::vector<double>& b, std::vector<double>& D,
										int Nx, int Ny, int M, int N, double h1, double h2, int maxit, double tol)
{
	std::vector<double> w(Nx * Ny, 0.0);
	std::vector<double> r(w.size(), 0.0), z(w.size(), 0.0), p(w.size(), 0.0), Ap(w.size(), 0.0);

	// Dz_0 = r_0
	// w_1 = w_0 + \alpha_1 * p_1
	r = B;
	for (int i = 0; i < r.size(); ++i) z[i] = r[i] / D[i];
	p = z;
	double rz = dotE(r, z, h1, h2);

	std::vector<double> w_good(w.size(), 0.0);
	std::vector<double> Aw_restart(w.size(), 0.0);
	double J_prev = 0.0;
	bool have_J_prev = false;
	bool restarted = false;

	int it = 0;
	for (; it < maxit; ++it) {
		build_Aw(p, Ap, a, b, Nx, Ny, M, N, h1, h2);
		double alpha = rz / dotE(p, Ap, h1, h2);

		// updating using formulas from (5)
		for (int i = 0; i < w.size(); ++i) {
			w[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
		}

		// ||w_{k+1} - w_k||_E = |alpha| * ||p_k||_E
		if (std::abs(alpha) * std::sqrt(dotE(p, p, h1, h2)) < tol) { ++it; break; }

		// The last formula from section (5)
		double Bw = dotE(B, w, h1, h2);
		double rw = dotE(r, w, h1, h2);
		double Jk = -0.5 * (Bw + rw);

		if (!have_J_prev) {
			J_prev = Jk;
			w_good = w;
			have_J_prev = true;
		} else if (Jk > J_prev) {
			// bad case (monotonicity violated)
		
			if (restarted) {
				std::cerr << "Warning: restart twice in a row" << std::endl;
				break;
			}
			
			std::cerr << "Warning: restart at " << it << std::endl;
			w = w_good;
			build_Aw(w, Aw_restart, a, b, Nx, Ny, M, N, h1, h2);
			for (int k = 0; k < r.size(); ++k) {
				r[k] = B[k] - Aw_restart[k];
				z[k] = r[k] / D[k];
			}
			p = z;
			rz = dotE(r, z, h1, h2);
			J_prev = -0.5 * (dotE(B, w, h1, h2) + dotE(r, w, h1, h2));
			restarted = true;
			continue;
		} else {
			J_prev = Jk;
			w_good = w;
		}

		// r is updated => resolve Dz = r and then update p using (5)
		for (int i = 0; i < r.size(); ++i) z[i] = r[i] / D[i];
		double rz_new = dotE(r, z, h1, h2);
		for (int i = 0; i < w.size(); ++i) p[i] = z[i] + (rz_new / rz) * p[i];
		rz = rz_new;
	}

	std::cout << "Finished in " << it << " iterations" << std::endl;

	return w;
}

void save_solution_csv(std::string& filename, std::vector<double>& w, int M, int N, double A1, double A2, double h1, double h2, int Ny)
{
	std::ofstream fout(filename.c_str());
	fout << std::setprecision(12);
	fout << "x,y,v\n";
	for (int i = 0; i <= M; ++i) {
		double x = A1 + i * h1;
		for (int j = 0; j <= N; ++j) {
			double y = A2 + j * h2;
			double vij = 0.0;
			if (i >= 1 && i <= M - 1 && j >= 1 && j <= N - 1) vij = w[idxW(i, j, Ny)];
			fout << x << "," << y << "," << vij << "\n";
		}
	}
	fout.close();
}

int main(int argc, char** argv) {
	using clock = std::chrono::steady_clock;
	auto t0_global = clock::now();

	Params P;
	parseArgs(argc, argv, P);

	const int M = P.M, N = P.N;
	const double A1 = P.A1, B1 = P.B1, A2 = P.A2, B2 = P.B2;
	const double h1 = (B1 - A1) / M;
	const double h2 = (B2 - A2) / N;
	const double h  = (h1 > h2) ? h1 : h2;
	const double eps = h * h;
	const double inv_eps = 1.0 / eps;
	const int Nx = M - 1;
	const int Ny = N - 1;

	std::vector<double> a = build_a(M, N, A1, A2, h1, h2, inv_eps);
	std::vector<double> b = build_b(M, N, A1, A2, h1, h2, inv_eps);
	std::vector<double> F = build_F(Nx, Ny, M, N, A1, A2, h1, h2);
	std::vector<double> D = build_D(Nx, Ny, M, N, h1, h2, a, b);
	
	auto t_mid = clock::now();
	std::cout << "Init time " << std::chrono::duration<double>(t_mid - t0_global).count() << "s" << std::endl;

	std::vector<double> w = solve_linear_system(F, a, b, D, Nx, Ny, M, N, h1, h2, P.maxit, P.tol);

	auto t1_global = clock::now();
	std::cout << "Loop time " << std::chrono::duration<double>(t_1 - t_mid).count() << "s" << std::endl;
	std::cout << "Global time " << std::chrono::duration<double>(t1_global - t0_global).count() << "s" << std::endl;
	return 0;
}

