/*
  Branch and bound algorithm to find the minimum of continuous binary
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <mpi.h>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"


using namespace std;


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(),xm);
  xr = interval(xm,x.right());
  yl = interval(y.left(),ym);
  yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
  interval fxy = f(x,y);

  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is
    // greater than the new minimum upper bound
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) {
    // We have potentially a new minimizer
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);

  minimize(f,xl,yl,threshold,min_ub,ml);
  minimize(f,xl,yr,threshold,min_ub,ml);
  minimize(f,xr,yl,threshold,min_ub,ml);
  minimize(f,xr,yr,threshold,min_ub,ml);
}


int main(int argc, char** argv)
{
  if (argc != 3) {
    cout << "Usage : " << argv[0] << " function precision" << endl;
    cout << "Which function to optimize?" << endl;
    cout << "Possible choices: " << endl;
    for (auto fname : functions) {
      cout << argv[0] << " " << fname.first << " 0.001"<< endl;
    }

    return 1;
  }

  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();
  double local_min_ub = numeric_limits<double>::infinity();
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision = std::stod(argv[2]);

  // Name of the function to optimize
  string choice_fun = argv[1];

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;

  try {
    fun = functions.at(choice_fun);
  } catch (out_of_range) {
    cerr << "Bad choice" << endl;
    return 1;
  }

  int rank,size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // split work on 4 worker
  interval xl, xr, yl, yr;
  split_box(fun.x,fun.y,xl,xr,yl,yr);

  if ( 0 % size == rank) {
    minimize(fun.f,xl,yl,precision,local_min_ub,minimums);
  }
  if ( 1 % size == rank) {
    minimize(fun.f,xl,yr,precision,local_min_ub,minimums);
  }
  if ( 2 % size == rank) {
    minimize(fun.f,xr,yl,precision,local_min_ub,minimums);
  }
  if ( 3 % size == rank) {
    minimize(fun.f,xr,yr,precision,local_min_ub,minimums);
  }

  // Displaying all potential minimizers
  copy(minimums.begin(),minimums.end(),
       ostream_iterator<minimizer>(cout,"\n"));
  cout << "Number of minimizers: " << minimums.size() << endl;
  cout << "Upper bound for minimum: " << local_min_ub << endl;

  MPI_Reduce(&local_min_ub, &min_ub, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    cout << "-- Result ------------------------------" << endl;
    cout << "Upper bound for minimum: " << min_ub << endl;
  }

  MPI_Finalize();
}

