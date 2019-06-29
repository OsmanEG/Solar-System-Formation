//*********************************************************************//
//                       Osman El-Ghotmi                               //
//                     University of Ottawa                            //
//                    Solar System Formation                           //
//*********************************************************************//

#include "eigen/Dense"
#include "eigen/Eigen"
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////

//Class representing particulate model solver
class Particulate_Model_Solver {
public:

    //Default constructors and assignment
    Particulate_Model_Solver()                                              = default;
    Particulate_Model_Solver(const Particulate_Model_Solver&)               = default;
    Particulate_Model_Solver(Particulate_Model_Solver&&)                    = default;
    Particulate_Model_Solver& operator=(const Particulate_Model_Solver&)    = default;
    Particulate_Model_Solver& operator=(Particulate_Model_Solver&&)         = default;

    //Constructors with real information
    Particulate_Model_Solver(double x_min_in, double x_max_in, int nx_in,
                             double y_min_in, double y_max_in, int ny_in,
                             const std::function<Eigen::Vector3d(double,double)>& initial_condition) :
            SolutionVector(3*nx_in*ny_in), //Solution vector composed of nx and ny
            ResidualVector(3*nx_in*ny_in), //Residual vector composed of nx and ny
            x_min(x_min_in), //Minimum X value from input
            x_max(x_max_in), //Maximum X value from input
            y_min(y_min_in), //Minimum Y value from input
            y_max(y_max_in), //Maximum Y value from input
            nx(nx_in), //nx from input
            ny(ny_in), //ny from input
            delta_x((x_max-x_min)/static_cast<double>(nx+1)), //Grid value of delta x
            delta_y((y_max-y_min)/static_cast<double>(ny+1)), //Grid value of delta y
            current_time(0.0) //Current time
    {
        //Checking for valid inputs
        if(x_min > x_max || nx <= 0 || y_min > y_max || ny <= 0) {
            throw std::runtime_error("Invalid inputs to solver.");
        }
        //Setting the initial value conditions
        set_initial_conditions(initial_condition);
    }

    //Returning values for nodes and grid dimensions
    auto num_x() const {return nx;} //Number of nodes in the x direction
    auto num_y() const {return ny;} //Number of nodes in the y direction
    auto dx()    const {return delta_x;} //delta x
    auto dy()    const {return delta_y;} //delta y

    //Positions of i-j nodes
    Eigen::Vector2d node(int i, int j) const {
        while   (i<0) i    += nx;
        auto    index_i   = i%nx;
        while   (j<0) j    += ny;
        auto    index_j   = j%ny;
        return {x_min + static_cast<double>(index_i)*delta_x + (i/nx)*(x_max-x_min),
                y_min + static_cast<double>(index_j)*delta_y + (j/ny)*(y_max-y_min)};
    }

    auto& U() {return SolutionVector;} //Full solution vector
    const auto& U() const {return SolutionVector;} //Full solution vector

    //Solution at i-j node. Returns eigen segment. Desired subvector reference
    auto U(int i, int j) {
        auto index = global_solution_index(i,j);
        return SolutionVector.segment<3>(index);
    }

    //Solution at i-j node. Returns eigen segment. Desired subvector reference
    auto U(int i, int j) const {
        auto index = global_solution_index(i,j);
        return SolutionVector.segment<3>(index);
    }

    auto& dUdt() {return ResidualVector;} //Full residual vector
    const auto& dUdt() const {return ResidualVector;} //Full residual vector

    //Solution at i-j node. Returns eigen segment. Desired subvector reference
    auto dUdt(int i, int j) {
        auto index = global_solution_index(i,j);
        return ResidualVector.segment<3>(index);
    }

    //Solution at i-j node. Returns eigen segment. Desired subvector reference
    auto dUdt(int i, int j) const {
        auto index = global_solution_index(i,j);
        return ResidualVector.segment<3>(index);
    }

    auto time() const {return current_time;} //Current time

    //Force x
    Eigen::Vector3d Fx(const Eigen::Vector3d& U) {
        return {U[1], U[1]*U[1]/U[0], U[1]*U[2]/U[0]};
    }

    //Force y
    Eigen::Vector3d Fy(const Eigen::Vector3d& U) {
        return {U[2], U[2]*U[1]/U[0], U[2]*U[2]/U[0]};
    }

    //Max wave speeds in x and y directions
    Eigen::Vector2d max_wavespeeds(const Eigen::Vector3d& U) {
        double a  = 0;
        double ux = U[1]/U[0];
        double uy = U[2]/U[0];
        return {fabs(ux)+a, fabs(uy)+a};
    }

    void time_march(double final_time, double CFL); //Time march to time

private:

    //Index in solution vector. Storing location for node i-j entries
    int global_solution_index(int i, int j) const {
        while(i<0) i += nx;
        auto index_i = i%nx;
        while(j<0) j += ny;
        auto index_j = j%ny;
        return 3 * (index_i + nx*index_j);
    }

    //Set initial conditions
    void set_initial_conditions(const std::function<Eigen::Vector3d(double, double)>& initial_condition) {
        for(int i = 0; i < nx; ++i) {
            for(int j = 0; j < ny; ++j) {
                auto n = node(i,j);
                U(i,j) = initial_condition(n.x(), n.y());
            }
        }
    }

    Eigen::VectorXd SolutionVector; //Solution vector (U)
    Eigen::VectorXd ResidualVector; // Residual vector (dUdt)
    const double x_min; //Minimum value of x
    const double x_max; //Maximum value of x
    const int nx; //Number of nodes in the x direction
    const double delta_x; //Spacing of nodes in the x direction
    const double y_min; //Minimum value of y
    const double y_max; //Maximum value of y
    const int ny; //Number of nodes in the y direction
    const double delta_y; //Spacing of nodes in the y direction
    double current_time; //Solution time
    constexpr static double g_const = 6.67408e-11; //Gravitational constant
};

////////////////////////////////////////////////////////////////////////

//Local lax friedrichs time march
void Particulate_Model_Solver::time_march(double final_time, double CFL) {
    std::cout << "Time marching from t = " << current_time
              << " to t = " << final_time
              << ".  Total difference = " << final_time-current_time << ".\n";
    const double time_tolerance = 1.0e-12; //Tolerance
    const double min_length = std::min(dx(), dy()); //Minimum length
    double A = delta_x*delta_y; //Area
    double mass_one = 0; //Mass of evaluated cell
    double mass_two = 0; //Mass of compared cell
    double distance_X = 0; //X direction distance
    double distance_Y = 0; //Y direction distance
    double calculated_Force_X = 0; //Calculated force in the x direction
    double calculated_Force_Y = 0; //Calculated force in the y direction
    double acceleration_X = 0; //Acceleration in the x direction
    double acceleration_Y = 0; //Acceleration in the y direction
    double x_position = 0.0;
    double y_position = 0.0;
    double radius = 0.0;
    double velocity = 0.0;
    double theta = 0.0;
    double rho = 0.0;
    int counter = 0;

    while(current_time < final_time - time_tolerance) {
        auto dt = std::numeric_limits<double>::max();
        dUdt().fill(0.0);
        for(int i = 0; i < num_x(); ++i) {
            for(int j = 0; j < num_y(); ++j) {
                Eigen::Vector3d Um = U(i  , j  );
                Eigen::Vector3d Ul = U(i-1, j  );
                Eigen::Vector3d Ur = U(i+1, j  );
                Eigen::Vector3d Ub = U(i  , j-1);
                Eigen::Vector3d Ut = U(i  , j+1);
                Eigen::Vector2d lambda = max_wavespeeds(Um);
                auto max_lambda = std::max(lambda[0], lambda[1]);
                dt = std::min(dt, CFL*min_length/max_lambda);
                dUdt(i,j) =  (Fx(Ul) - Fx(Ur) + lambda[0]*(Ul - 2.0*Um + Ur)) / (2.0*delta_x)
                             +(Fy(Ub) - Fy(Ut) + lambda[1]*(Ub - 2.0*Um + Ut)) / (2.0*delta_y);
            }
        }

        //Initial Condition for Orbital Velocity
            //Non of this really makes sense or works so I don't know
        if(counter == 0) {
            counter = 1;
            for(int i = x_min; i < x_max; ++i) {
                for(int j = y_min; j < y_max; ++j) {
                    x_position = i*delta_x;
                    y_position = j*delta_y;
                    radius = sqrt(fabs(x_position*x_position) + fabs(y_position*y_position));
                    theta = atan2(y_position, x_position);
                    rho = U(i,j)[0];
                    if(radius > 0.0){
                        velocity = sqrt(fabs((g_const*rho*A)/radius));
                    }
                    else{
                        velocity = 0.0;
                    }
                    U(i,j)[1] += -rho*velocity*cos(theta+3.14159/2);
                    U(i,j)[2] += -rho*velocity*sin(theta+3.14159/2);
                }
            }
        }

        //Force of gravity calculation
        for(int i1 = 0; i1 < num_x(); ++i1){
            //New cell to be evaluated: x coordinate
            for(int j1 = 0; j1 < num_y(); ++j1){
                //New cell to be evaluated: y coordinate
                mass_one = U(i1,j1)[0]*A; //Calculating the mass of the cell being evaluated
                for(int i2 = 0; i2 < num_x(); ++i2){
                    //Comparing cell: x coordinate
                    for(int j2 = 0; j2 < num_y(); ++j2){
                        //Comparing cell: y coordinate
                        mass_two = U(i2,j2)[0]*A; //Calculating the mass of the comparing cell
                        distance_X = (i2 - i1)*delta_x; //Distance in the x direction
                        distance_Y = (j2 - j1)*delta_y; //Distance in the y direction

                        //Calculating the force between each cell
                        if(fabs(distance_X) > 0.0 && fabs(distance_Y) > 0.0) {
                            calculated_Force_X = mass_one*((g_const*mass_two)/((distance_X)*(fabs(distance_X))));
                            calculated_Force_Y = mass_one*((g_const*mass_two)/((distance_Y)*(fabs(distance_Y))));
                        }
                        else if(fabs(distance_Y) > 0.0 && fabs(distance_X) == 0.0) {
                            calculated_Force_X = 0.0;
                            calculated_Force_Y = mass_one*((g_const*mass_two)/((distance_Y)*(fabs(distance_Y))));
                        }

                        else if(fabs(distance_X) > 0.0 && fabs(distance_Y) == 0.0) {
                            calculated_Force_X = mass_one*((g_const*mass_two)/((distance_X)*(fabs(distance_X))));
                            calculated_Force_Y = 0.0;
                        }

                        //Calculating the acceleration resulting between each of the cells
                        acceleration_X = calculated_Force_X/mass_one;
                        acceleration_Y = calculated_Force_Y/mass_one;

                        //Summing the resulting accelerations of each cell
                        U(i2,j2)[1] += -acceleration_X*dt*U(i1,j1)[0];
                        U(i2,j2)[2] += -acceleration_Y*dt*U(i1,j1)[0];

                        //Output comments to analyse the resulting output from each of the cells
                        //std::cout << "mass one: " << mass_one << ".\n";
                        //std::cout << "mass two: " << mass_two << ".\n";
                        //std::cout << "distance X: " << distance_X << ".\n";
                        //std::cout << "distance Y: " << distance_Y << ".\n";
                        //std::cout << "Force X: " << calculated_Force_X << ".\n";
                        //std::cout << "Force Y: " << calculated_Force_Y << ".\n";
                    }
                }
            }
        }

        //Checking the position of the current time
        if(current_time + dt > final_time) {
            dt = final_time - current_time;
        }

        U()  += dt*dUdt();
        current_time += dt;
        //std::cout << "Time = " << std::setw(10) << current_time << '\n';
    }

    std::cout << "Time marching done.\n";
}

///////////////////////////////////////////////////////////////////////////

//Writing out to VTK
void write_to_VTK(const Particulate_Model_Solver& solver,
                  const std::string& filename) {
    std::ofstream fout(filename);
    if(!fout) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::cout << "Writing output to file: " << filename << '\n';
    const auto num_nodes = (solver.num_x()+1)*(solver.num_y()+1);
    fout << "# vtk DataFile Version 2.0\n"
         << "Particulate Flow Equations Solution\n"
         << "ASCII\n"
         << "DATASET STRUCTURED_GRID\n"
         << "DIMENSIONS " << solver.num_x()+1 << " " << solver.num_y()+1 << " 1\n"
         << "POINTS " << num_nodes << " double\n";
    for(int j = 0; j <= solver.num_y(); ++j) {
        for(int i = 0; i <= solver.num_x(); ++i) {
            auto n = solver.node(i,j);
            auto U = solver.U(i,j);
            fout << n.x() << " " << n.y() << " " << 0.0 << '\n';
        }
    }

    fout << "\nPOINT_DATA " << num_nodes
         << "\nSCALARS h double 1\nLOOkUP_TABLE default\n";
    for(int j = 0; j <= solver.num_y(); ++j) {
        for(int i = 0; i <= solver.num_x(); ++i) {
            fout << solver.U(i,j)[0] << '\n';
        }
    }

    fout << "\nVECTORS u double\n";
    for(int j = 0; j <= solver.num_y(); ++j) {
        for(int i = 0; i <= solver.num_x(); ++i) {
            auto ux = solver.U(i,j)[1]/solver.U(i,j)[0];
            auto uy = solver.U(i,j)[2]/solver.U(i,j)[0];
            fout << ux << " " << uy << " 0.0\n";
        }
    }
}

///////////////////////////////////////////////////////////////////////

//Making movie
void make_movie(Particulate_Model_Solver& solver,
                const double final_time,
                const double CFL,
                const int number_of_frames,
                const std::string& filename_base) {
    if(number_of_frames < 2) {
        throw std::runtime_error("make_movie requires at least two frames.");
    }

    const auto frame_time = (final_time-solver.time())/static_cast<double>(number_of_frames-1);
    const auto build_filename = [](const std::string& filename_base, int index) {
        std::stringstream filename_ss;
        filename_ss <<  filename_base << "_" << std::setfill('0') << std::setw(5) << index << ".vtk";
        return filename_ss.str();
    };

    write_to_VTK(solver, build_filename(filename_base, 0));
    for(int i = 1; i<number_of_frames; ++i) {
        double target_time = static_cast<double>(i)*frame_time;
        solver.time_march(target_time, CFL);
        write_to_VTK(solver, build_filename(filename_base, i));
    }
}

/////////////////////////////////////////////////////////////////////////

//Main
int main() {
    std::cout << "|------------------------------------|\n"
              << "|          Osman El-Ghotmi           |\n"
              << "|        University of Ottawa        |\n"
              << "|       Solar System Formation       |\n"
              << "|------------------------------------|\n";
    double x_min = -10.0; //Initial condition for minimum x value
    double x_max =  10.0; //Initial condition for maximum x value
    double y_min = -10.0; //Initial condition for minimum y value
    double y_max =  10.0; //Initial condition for maximum y value
    int       nx =  100; //Initial condition for number of x cells
    int       ny =  100; //Initial condition for number of y cells

    //Initial conditions for density values
    auto initial_condition = [] (double x, double y) {
        Eigen::Vector3d U;
        U[0] = 1.0e-4;
        U[1] = 1.0e-4;
        U[2] = 1.0e-4;
        if(fabs(x*x+y*y) < 1.0) {
            U[0] += 3.0;
        }
        if(fabs(x*x+y*y) < 2.5 && fabs(x*x+y*y) >= 1.0) {
            U[0] += 0.75;
        }
        if(fabs(x*x+y*y) < 40.0 && fabs(x*x+y*y) >= 30.0) {
            U[0] += 1.0;
        }
        return U;
    };

    //Calling on the particulate model solver with initial conditions
    auto solver = Particulate_Model_Solver(x_min, x_max, nx, y_min, y_max, ny, initial_condition);

    //Creating a movie for paraview
    make_movie(solver, 2.5, 0.5, 750, "movie");
    return 0;
}