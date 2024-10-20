from eigency.core cimport *
from libcpp.string cimport string

ctypedef unsigned int uint
#cdef extern from "instantAlg.cpp":
cdef extern from "incremental_update.cpp":
#cdef extern from "instantAlg_arxiv.cpp":
	pass

cdef extern from "instantAlg.h" namespace "propagation":
	cdef cppclass Instantgnn:
		Instantgnn() except+
		double initial_operation(string,string,uint,uint,double,double, double, double,double,Map[MatrixXd], string &) except +
		void split_batch(string, double, Map[MatrixXd], double) except +
		void snapshot_lazy(string, double,double,double, double, Map[MatrixXd], Map[MatrixXd], string &) except +
		void snapshot_operation(string, double, double, Map[MatrixXd], string &)
		
