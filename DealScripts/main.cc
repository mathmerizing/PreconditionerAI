/*
 * Saving system matrices of the Poisson problem with variable coefficients.
 * This file has been created with the help of the deal.II tutorials.
 * Especially: step-2, step-6, step-16, ...
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

#include <deal.II/numerics/vector_tools.h>

// C++ imports:
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Boost imports:
#include <boost/progress.hpp>

using namespace dealii;

// enum for type of DoF renumbering
enum Renumbering { none, cuthill, king };

std::string to_string(Renumbering renum)
{
    switch (renum)
    {
    case Renumbering::none:
        return "none";
        break;
    case Renumbering::cuthill:
        return "cuthill";
        break;
    case Renumbering::king:
        return "king";
        break;
    default:
        return "unknown";
        break;
    }
}

namespace FEMatrixGenerator
{
    template <int dim>
    class FEProblem
    {
    public:
        FEProblem(const unsigned int degree);
        void run();

    private:
        void setup_system();
        void renumbering(Renumbering renum);
        void assemble_system(double first_coeff, double second_coeff);
        void invert_and_output(Renumbering renum, double first_coeff, double second_coeff);
        void output_matrices(Renumbering renum, double first_coeff, double second_coeff);

        Triangulation<dim>  triangulation;
        FE_Q<dim>           fe;
        DoFHandler<dim>     dof_handler;

        SparsityPattern         sparsity_pattern;
        SparseMatrix<double>    system_matrix;

        Vector<double> system_rhs;

        AffineConstraints<double> constraints;

        SparseMatrix<double>      inverse_matrix;

        const unsigned int degree;
        double first_coeff;
        double second_coeff;
    };

    template <int dim>
    FEProblem<dim>::FEProblem(const unsigned int degree)
        : fe(degree)
        , dof_handler(triangulation)
        , degree(degree)
    {}

    template <int dim>
    void FEProblem<dim>::setup_system()
    {
        dof_handler.distribute_dofs(fe);
        std::cout << "Number of degrees of freedom:   " << dof_handler.n_dofs() << std::endl;

        // set dimension of system_rhs
        system_rhs.reinit(dof_handler.n_dofs());

        // create constraints
        constraints.clear();
        constraints.reinit();
        std::set<types::boundary_id> dirichlet_boundary_ids{0};
        Functions::ConstantFunction<dim> boundary_values(1.0);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                boundary_values,
                                                constraints);
        constraints.close();
    }

    template <int dim>
    void FEProblem<dim>::renumbering(Renumbering renum)
    {
        // chose DoF renumbering
        std::string file_name;
        switch (renum)
        {
        case Renumbering::none:
            file_name = "sparsity_pattern_unordered.svg";
            break;
        case Renumbering::cuthill:
            DoFRenumbering::Cuthill_McKee(
                dof_handler,    // dof_handler
                false,          // reversed_numbering
                false           // use_constraints
            );
            file_name = "sparsity_pattern_cuthill.svg";
            break;
        case Renumbering::king:
            DoFRenumbering::boost::king_ordering(
                dof_handler,    // dof_handler
                false,          // reversed_numbering
                false           // use_constraints
            );
            file_name = "sparsity_pattern_king.svg";
            break;
        }

        // create sparsity pattern
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        std::cout << file_name << " - bandwidth: " << dsp.bandwidth() << std::endl;

        sparsity_pattern.copy_from(dsp);
        std::ofstream out(file_name);
        sparsity_pattern.print_svg(out);
    }

    template <int dim>
    void FEProblem<dim>::assemble_system(double first_coeff, double second_coeff)
    {
        system_matrix.reinit(sparsity_pattern);
        inverse_matrix.reinit(sparsity_pattern);

        const QGauss<dim> quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values 
                                | update_gradients 
                                | update_quadrature_points 
                                | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            cell_matrix = 0;
            cell_rhs    = 0;
            fe_values.reinit(cell);
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
                const double coefficient =
                    (fe_values.get_quadrature_points()[q_index][0] < 0.0) ? first_coeff : second_coeff;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) +=
                            (coefficient *                              // a(x_q)
                             fe_values.shape_grad(i, q_index) *         // grad phi_i(x_q)
                             fe_values.shape_grad(j, q_index) *         // grad phi_j(x_q)
                             fe_values.JxW(q_index));                   // dx
                    cell_rhs(i) = 0.0;
                }
            }
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        }
    }

    template <int dim>
    void FEProblem<dim>::invert_and_output(Renumbering renum, double first_coeff, double second_coeff)
    {
        // convert sparse system to FullMatrix and invert
        FullMatrix<double> tmp_system;
        FullMatrix<double> tmp_inverse(dof_handler.n_dofs());
        tmp_system.copy_from(system_matrix);
        tmp_inverse.invert(tmp_system);

        // store the inverse matrix as a SparseMatrix
        SparsityPattern inverse_sparsity_pattern;
        inverse_sparsity_pattern.copy_from(tmp_inverse);
        inverse_matrix.reinit(inverse_sparsity_pattern);
        inverse_matrix.copy_from(tmp_inverse);

        // output system matrix and inverse matrix
        FEProblem<dim>::output_matrices(renum, first_coeff, second_coeff);
    }

    template <int dim>
    void FEProblem<dim>::output_matrices(Renumbering renum, double first_coeff, double second_coeff)
    {
        const std::string start  = "./dataset/"  + to_string(renum) + "/";
        const std::string ending = std::to_string(dof_handler.n_dofs()) + "_"
                                    + std::to_string(first_coeff) + "_"
                                    + std::to_string(second_coeff) + ".txt";

        // save system matrix as txt file
        const std::string file_name_system = start + "system_" + ending;
        std::ofstream system_matrix_out(file_name_system);
        system_matrix_out.precision(17);
        system_matrix.print(system_matrix_out);

        // save inverse matrix as txt file
        const std::string file_name_inverse = start + "inverse_" + ending;
        std::ofstream inverse_matrix_out(file_name_inverse);
        inverse_matrix_out.precision(17);
        inverse_matrix.print(inverse_matrix_out);
    }

    template <int dim>
    void FEProblem<dim>::run()
    {
        // create grid
        int global_refinements = 3;
        GridGenerator::hyper_ball(triangulation);
        triangulation.refine_global(global_refinements);

        setup_system();

        // create a vector in which all possible coefficient values are saved
        std::vector<double> coefficient_vec;
        for (int i = 1; i <= 10; i++)
            coefficient_vec.push_back(i * 0.1);
        const auto n_coeffcients = coefficient_vec.size(); 

        // for each Renumbering type
        for (int renumInt = Renumbering::none; renumInt <= Renumbering::king; renumInt++)
        {
            Renumbering renum = static_cast<Renumbering>(renumInt);
            renumbering(renum);

            int iter = 0;
            
            for(auto const& first_coeff: coefficient_vec) 
            {   
                std::cout << "" << std::endl;
                std::cout << "LOOP " + std::to_string(iter+1) + "/"
                            + std::to_string(n_coeffcients) + ": ";
                boost::progress_display progress(10) ;
                for(auto const& second_coeff: coefficient_vec)
                {
                    assemble_system(first_coeff,second_coeff);
                    invert_and_output(renum, first_coeff,second_coeff);
                    ++progress;
                }
                ++iter;
            }
            std::cout << "" << std::endl; 
        }
    }

} // namespace FEMatrixGenerator

int main()
{
    try
    {
        const unsigned int degree = 2;
        
        FEMatrixGenerator::FEProblem<2> problem(degree);
        problem.run();
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
    
