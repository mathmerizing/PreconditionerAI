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
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

// C++ imports:
#include <iostream>
#include <fstream>

using namespace dealii;

// enum for type of DoF renumbering
enum Renumbering { cuthill, king };

namespace FEMatrixGenerator
{
    template <int dim>
    class FEProblem
    {
    public:
        FEProblem(const unsigned int degree, double first_coeff, double second_coeff, bool renumber);
        void run(Renumbering renum);

    private:
        void setup_system();
        void renumbering(Renumbering renum);
        void assemble_system();
        void invert_system();
        void output_matrices();

        Triangulation<dim>  triangulation;
        FE_Q<dim>           fe;
        DoFHandler<dim>     dof_handler;

        SparsityPattern         sparsity_pattern;
        // SparseMatrix<double>    system_matrix;
        FullMatrix<double>      system_matrix;

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

    template <int dim>
    void FEProblem<dim>::setup_system()
    {
        dof_handler.distribute_dofs(fe);

        DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
        std::cout << "Unordered matrix   - bandwidth: " << dynamic_sparsity_pattern.bandwidth() << std::endl;
        
        sparsity_pattern.copy_from(dynamic_sparsity_pattern);
        std::ofstream out("sparsity_pattern_unordered.svg");
        sparsity_pattern.print_svg(out);
    }

    template <int dim>
    void FEProblem<dim>::renumbering(Renumbering renum)
    {
        // chose DoF renumbering
        switch (renum)
        {
        case Renumbering::cuthill:
            DoFRenumbering::Cuthill_McKee(
                dof_handler,    // dof_handler
                false,          // reversed_numbering
                false           // use_constraints
            );
            break;
        case Renumbering::king:
            DoFRenumbering::boost::king_ordering(
                dof_handler,    // dof_handler
                false,          // reversed_numbering
                false           // use_constraints
            );
            break;
        }

        DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
        std::cout << "Renumbered matrix  - bandwidth: " << dynamic_sparsity_pattern.bandwidth() << std::endl;

        sparsity_pattern.copy_from(dynamic_sparsity_pattern);
        std::ofstream out("sparsity_pattern_cuthill.svg");
        sparsity_pattern.print_svg(out);
    }

    template <int dim>
    void FEProblem<dim>::assemble_system()
    {
        // TO DO
    }

    template <int dim>
    void FEProblem<dim>::invert_system()
    {
        // TO DO
    }

    template <int dim>
    void FEProblem<dim>::output_matrices()
    {
        // TO DO
    }

    template <int dim>
    void FEProblem<dim>::run(Renumbering renum)
    {
        // create grid
        int global_refinements = 3;
        GridGenerator::hyper_ball(triangulation);
        triangulation.refine_global(global_refinements);

        setup_system();
        renumbering(renum);
        assemble_system();
        invert_system();
        output_matrices();
    }

} // namespace FEMatrixGenerator

int main()
{
    try
    {
        // TO DO: read coefficients from options.prm with ParameterHandler
        double coeff_1 = 1.0;
        double coeff_2 = 0.1;
        const unsigned int degree = 2;
        bool reduce_bandwidth = true;
        Renumbering renum = Renumbering::king;
        
        FEMatrixGenerator::FEProblem<2> problem(degree, coeff_1, coeff_2, reduce_bandwidth);
        problem.run(renum);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "-----------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!"<< std::endl
                  << "-----------------------------------"
                  << std::endl;
    }
}
    
