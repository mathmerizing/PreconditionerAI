/*
 * Saving system matrices of the Poisson problem with variable coefficients.
 * This file has been created with the help of the deal.II tutorials.
 * Especially: step XX, step XX, ...
 * 
 * Author: Julian Roth, 2020 
 */

// deal.II imports:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

// C++ imports:
#include <iostream>
#include <fstream>

using namespace dealii;

namespace FEMatrixGenerator
{
    template <int dim>
    class FEProblem
    {
    public:
        FEProblem(const unsigned int degree, double first_coeff, double second_coeff);
        void run();

    private:
        void setup_system();
        void assemble_system();
        void invert_system();
        void output_matrices();

        Triangulation<dim>  triangulation;
        FE_Q<dim>           fe;
        DoFHandler<dim>     dof_handler;

        SparsityPattern         sparsity_pattern;
        SparseMatrix<double>    system_matrix;

        AffineConstraints<double> constraints;

        FullMatrix<double>      inverse_matrix;

        const unsigned int degree;
        double first_coeff;
        double second_coeff;
        bool reduce_bandwidth;
    };

    template <int dim>
    FEProblem<dim>::FEProblem(const unsigned int degree, double first_coeff, double second_coeff, bool renumber)
        : fe(degree)
        , dof_handler(triangulation)
        , degree(degree)
        , first_coeff(first_coeff)
        , second_coeff(second_coeff)
        , reduce_bandwidth(renumber)
    {}
} // namespace FEMatrixGenerator

int main()
{
    // TO DO: read coefficients from options.prm with ParameterHandler
    double coeff_1 = 1.0;
    double coeff_2 = 0.1;
    const unsigned int degree = 1;
    bool reduce_bandwidth = true;
    FEMatrixGenerator::FEProblem<2> problem(degree, coeff_1, coeff_2, reduce_bandwidth);
}
