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
#include <thread>
#include <mutex>
#include <unistd.h>
#include <vector>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"


using namespace std;

mutex local_min_mutex;

struct CheckMin {
  double *min;
  int rank;
  CheckMin(double *m, int r) : min(m), rank(r) {}
  void operator()(void) {
    MPI_Request req[4];
    MPI_Status status[4];
    double woldMin[4] = {*min, *min, *min, *min};
    int flag[4] = {0, 0, 0, 0};
    flag[rank] = 1;
    for (int i = 0; i < 4; ++i) {
      MPI_Irecv(&(woldMin[i]), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(req[i]));
    }

    do {
      for (int i = 0; i < 4; ++i) {
	if (i != rank) {
	  MPI_Test(&(req[i]), &(flag[i]), &(status[i]));

	  if (flag[i]) {
	    {
	      lock_guard<mutex> guard(local_min_mutex);
	      if (woldMin[i] < *min) {
		*min = woldMin[i];
	      }
	    }
	  }
	}
      }
      usleep(50000);

    } while(!(flag[0] && flag[1] && flag[2] && flag[3]));
  }
};


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
	      double* min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
  interval fxy = f(x,y);

  if (fxy.left() > *min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < *min_ub) { // Current box contains a new minimum?
    {
      lock_guard<mutex> guard(local_min_mutex);
      if (fxy.right() < *min_ub) {
	*min_ub = fxy.right();
      }
    }
    // Discarding all saved boxes whose minimum lower bound is
    // greater than the new minimum upper bound
#pragma omp critical
    {
      auto discard_begin = ml.lower_bound(minimizer{0,0,*min_ub,0});
      ml.erase(discard_begin,ml.end());
    }
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) {
    // We have potentially a new minimizer
#pragma omp critical
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);

#pragma omp parallel
#pragma omp single nowait
  {
#pragma omp task shared(min_ub)
    minimize(f,xl,yl,threshold,min_ub,ml);
#pragma omp task shared(min_ub)
    minimize(f,xl,yr,threshold,min_ub,ml);
#pragma omp task shared(min_ub)
    minimize(f,xr,yl,threshold,min_ub,ml);
#pragma omp task shared(min_ub)
    minimize(f,xr,yr,threshold,min_ub,ml);
#pragma omp taskwait
  }
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

  vector<thread> poolThread;

  if ( 0 % size == rank) {
    poolThread.push_back(thread(CheckMin(&min_ub, rank)));
    minimize(fun.f,xl,yl,precision,&min_ub,minimums);

    for (int i = 0; i < 4; i++) {
      if (i != rank) {
	MPI_Request req;
	MPI_Isend(&min_ub, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req);
      }
    }
  }
  if ( 1 % size == rank) {
    poolThread.push_back(thread(CheckMin(&min_ub, rank)));
    minimize(fun.f,xl,yr,precision,&min_ub,minimums);

    for (int i = 0; i < 4; i++) {
      if (i != rank) {
	MPI_Request req;
	MPI_Isend(&min_ub, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req);
      }
    }
  }
  if ( 2 % size == rank) {
    poolThread.push_back(thread(CheckMin(&min_ub, rank)));
    minimize(fun.f,xr,yl,precision,&min_ub,minimums);

    for (int i = 0; i < 4; i++) {
      if (i != rank) {
	MPI_Request req;
	MPI_Isend(&min_ub, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req);
      }
    }
  }
  if ( 3 % size == rank) {
    poolThread.push_back(thread(CheckMin(&min_ub, rank)));
    minimize(fun.f,xr,yr,precision,&min_ub,minimums);

    for (int i = 0; i < 4; i++) {
      if (i != rank) {
	MPI_Request req;
	MPI_Isend(&min_ub, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req);
      }
    }
  }

  // Displaying all potential minimizers
//  copy(minimums.begin(),minimums.end(),
//       ostream_iterator<minimizer>(cout,"\n"));
//  cout << "Number of minimizers: " << minimums.size() << endl;
//  cout << "Upper bound for minimum: " << min_ub << endl;

  for(auto& t: poolThread) {
    t.join();
  }

  if (rank == 0) {
    cout << "-- Result ------------------------------" << endl;
    cout << "Upper bound for minimum: " << min_ub << endl;
  }


  MPI_Finalize();
}

