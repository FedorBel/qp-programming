#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
// #include <Eigen/Dense>
// #include <Eigen/S>

// using Eigen::MatrixXd;

#include <numeric>

int main()
{
    // MatrixXd m(2, 2);
    // m(0, 0) = 3;
    // m(1, 0) = 2.5;
    // m(0, 1) = -1;
    // m(1, 1) = m(1, 0) + m(0, 1);
    Eigen::Vector2d coeff_vec{1, -1};
    Eigen::Matrix2d coeff = coeff_vec * coeff_vec.transpose();

    Eigen::Matrix2d hess;
    hess(0, 0) = 1;
    hess(0, 1) = 0;
    hess(1, 0) = 0;
    hess(1, 1) = 1;
    hess *= 2;
    std::cout << coeff << std::endl;
    std::cout << hess << std::endl;
    std::cout << coeff * 2 * 10 + hess << std::endl;
    hess += coeff * 2 * 10;

    Eigen::LLT<Eigen::MatrixXd> lltOfA(hess); // compute the Cholesky decomposition of A
    if (lltOfA.info() == Eigen::NumericalIssue)
    {
        std::cout << "Possibly non semi-positive definitie matrix!" << std::endl;
    }

    {
        Eigen::SparseMatrix<double> matrix_h;
        double FLAGS_tension_2_deviation_weight = 0.005;
        double FLAGS_tension_2_curvature_weight = 1.0;
        double FLAGS_tension_2_curvature_rate_weight = 10.0;
        size_t size = 5;
        const size_t x_start_index = 0;
        const size_t y_start_index = x_start_index + size;
        const size_t theta_start_index = y_start_index + size;
        const size_t k_start_index = theta_start_index + size;
        const size_t matrix_size = 4 * size - 1;
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Constant(matrix_size, matrix_size, 0);
        // Deviation and curvature.
        for (int i = 0; i != size; ++i)
        {
            hessian(x_start_index + i, x_start_index + i) = hessian(y_start_index + i, y_start_index + i) = FLAGS_tension_2_deviation_weight * 2;
            if (i != size - 1)
                hessian(k_start_index + i, k_start_index + i) = FLAGS_tension_2_curvature_weight * 2;
        }
        // Curvature change.
        Eigen::Vector2d coeff_vec{1, -1};
        Eigen::Matrix2d coeff = coeff_vec * coeff_vec.transpose();
        for (int i = 0; i != size - 2; ++i)
        {
            hessian.block(k_start_index + i, k_start_index + i, 2, 2) += 2 * FLAGS_tension_2_curvature_rate_weight * coeff;
        }
        matrix_h = hessian.sparseView();
        std::cout << "=======================================" << std::endl;
        std::cout << hessian << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << matrix_h << std::endl;

        Eigen::LLT<Eigen::MatrixXd> lltOfA(matrix_h); // compute the Cholesky decomposition of A
        if (lltOfA.info() == Eigen::NumericalIssue)
        {
            std::cout << "Possibly non semi-positive definitie matrix!" << std::endl;
        }
    }

    std::cout << "constraints" << std::endl;
    {
        int size = 3;
        std::vector<double> angle_list(size, M_PI_4);
        std::vector<double> s_list(size);
        std::iota(s_list.begin(), s_list.end(), 0);
        std::vector<double> k_list(size);
        std::iota(k_list.begin(), k_list.end(), 10);
        const size_t x_start_index = 0;
        const size_t y_start_index = x_start_index + size;
        const size_t theta_start_index = y_start_index + size;
        const size_t k_start_index = theta_start_index + size;
        const size_t cons_x_update_start_index = 0;
        const size_t cons_y_update_start_index = cons_x_update_start_index + size - 1;
        const size_t cons_theta_update_start_index = cons_y_update_start_index + size - 1;
        const size_t cons_x_index = cons_theta_update_start_index + size - 1;
        const size_t cons_y_index = cons_x_index + 1;

        Eigen::MatrixXd cons = Eigen::MatrixXd::Zero(3 * (size - 1) + 2, 4 * size - 1);
        Eigen::VectorXd lower_bound = Eigen::MatrixXd::Zero(3 * (size - 1) + 2, 1);
        Eigen::VectorXd upper_bound = Eigen::MatrixXd::Zero(3 * (size - 1) + 2, 1);
        // Cons.
        for (int i = 0; i != size - 1; ++i)
        {
            const double ds = s_list[i + 1] - s_list[i];
            cons(cons_x_update_start_index + i, x_start_index + i + 1) =
                cons(cons_y_update_start_index + i, y_start_index + i + 1) = cons(cons_theta_update_start_index + i, theta_start_index + i + 1) = 1;
            cons(cons_x_update_start_index + i, x_start_index + i) = cons(cons_y_update_start_index + i, y_start_index + i) = cons(cons_theta_update_start_index + i, theta_start_index + i) = -1;
            cons(cons_x_update_start_index + i, theta_start_index + i) = ds * sin(angle_list[i]);
            cons(cons_y_update_start_index + i, theta_start_index + i) = -ds * cos(angle_list[i]);
            cons(cons_theta_update_start_index + i, k_start_index + i) = -ds;
        }
        cons(cons_x_index, x_start_index) = cons(cons_y_index, y_start_index) = 1;
        Eigen::SparseMatrix<double> matrix_constraints = cons.sparseView();
        // Bounds.
        for (int i = 0; i != size - 1; ++i)
        {
            const double ds = s_list[i + 1] - s_list[i];
            (lower_bound)(cons_x_update_start_index + i) = (upper_bound)(cons_x_update_start_index + i) =
                ds * cos(angle_list[i]);
            (lower_bound)(cons_y_update_start_index + i) = (upper_bound)(cons_y_update_start_index + i) =
                ds * sin(angle_list[i]);
            (lower_bound)(cons_theta_update_start_index + i) = (upper_bound)(cons_theta_update_start_index + i) =
                -ds * k_list[i];
        }
        // (*lower_bound)(cons_x_index) = (*upper_bound)(cons_x_index) = x_list[0];
        // (*lower_bound)(cons_y_index) = (*upper_bound)(cons_y_index) = y_list[0];
        std::cout << "=======================================" << std::endl;
        std::cout << cons << std::endl;
    }
}