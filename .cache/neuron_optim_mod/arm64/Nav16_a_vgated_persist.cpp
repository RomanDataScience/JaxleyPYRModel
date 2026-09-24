/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
// clang-format off
#include "md1redef.h"
#include "section_fwd.hpp"
#include "nrniv_mf.h"
#include "md2redef.h"
#include "nrnconf.h"
// clang-format on
#include "neuron/cache/mechanism_range.hpp"
static constexpr auto number_of_datum_variables = 3;
static constexpr auto number_of_floating_point_variables = 33;
namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
  neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#if NRN_ENABLE_ARCH_INDEP_EXP_POW
#undef pow
#define pow hoc_pow
#endif
#endif
 
#define nrn_init _nrn_init__na16a_vgp
#define _nrn_initial _nrn_initial__na16a_vgp
#define nrn_cur _nrn_cur__na16a_vgp
#define _nrn_current _nrn_current__na16a_vgp
#define nrn_jacob _nrn_jacob__na16a_vgp
#define nrn_state _nrn_state__na16a_vgp
#define _net_receive _net_receive__na16a_vgp 
#define kin kin__na16a_vgp 
#define rates rates__na16a_vgp 
 
#define _threadargscomma_ _ml, _iml, _ppvar, _thread, _globals, _nt,
#define _threadargsprotocomma_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _internalthreadargsprotocomma_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _threadargs_ _ml, _iml, _ppvar, _thread, _globals, _nt
#define _threadargsproto_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
#define _internalthreadargsproto_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 
#define t _nt->_t
#define dt _nt->_dt
#define gbar _ml->template fpfield<0>(_iml)
#define gbar_columnindex 0
#define I2init _ml->template fpfield<1>(_iml)
#define I2init_columnindex 1
#define C1O1v2 _ml->template fpfield<2>(_iml)
#define C1O1v2_columnindex 2
#define dist _ml->template fpfield<3>(_iml)
#define dist_columnindex 3
#define slowdown _ml->template fpfield<4>(_iml)
#define slowdown_columnindex 4
#define persist _ml->template fpfield<5>(_iml)
#define persist_columnindex 5
#define p_vhalf _ml->template fpfield<6>(_iml)
#define p_vhalf_columnindex 6
#define p_k _ml->template fpfield<7>(_iml)
#define p_k_columnindex 7
#define fast_inactivation_tau_scale _ml->template fpfield<8>(_iml)
#define fast_inactivation_tau_scale_columnindex 8
#define slow_recovery_tau_scale _ml->template fpfield<9>(_iml)
#define slow_recovery_tau_scale_columnindex 9
#define ina _ml->template fpfield<10>(_iml)
#define ina_columnindex 10
#define g _ml->template fpfield<11>(_iml)
#define g_columnindex 11
#define C1 _ml->template fpfield<12>(_iml)
#define C1_columnindex 12
#define O1 _ml->template fpfield<13>(_iml)
#define O1_columnindex 13
#define I1 _ml->template fpfield<14>(_iml)
#define I1_columnindex 14
#define I2 _ml->template fpfield<15>(_iml)
#define I2_columnindex 15
#define ena _ml->template fpfield<16>(_iml)
#define ena_columnindex 16
#define C1O1_a _ml->template fpfield<17>(_iml)
#define C1O1_a_columnindex 17
#define O1C1_a _ml->template fpfield<18>(_iml)
#define O1C1_a_columnindex 18
#define O1I1_a _ml->template fpfield<19>(_iml)
#define O1I1_a_columnindex 19
#define I1O1_a _ml->template fpfield<20>(_iml)
#define I1O1_a_columnindex 20
#define I1I2_a _ml->template fpfield<21>(_iml)
#define I1I2_a_columnindex 21
#define I2I1_a _ml->template fpfield<22>(_iml)
#define I2I1_a_columnindex 22
#define I1C1_a _ml->template fpfield<23>(_iml)
#define I1C1_a_columnindex 23
#define C1I1_a _ml->template fpfield<24>(_iml)
#define C1I1_a_columnindex 24
#define Q10 _ml->template fpfield<25>(_iml)
#define Q10_columnindex 25
#define p_gate _ml->template fpfield<26>(_iml)
#define p_gate_columnindex 26
#define DC1 _ml->template fpfield<27>(_iml)
#define DC1_columnindex 27
#define DO1 _ml->template fpfield<28>(_iml)
#define DO1_columnindex 28
#define DI1 _ml->template fpfield<29>(_iml)
#define DI1_columnindex 29
#define DI2 _ml->template fpfield<30>(_iml)
#define DI2_columnindex 30
#define v _ml->template fpfield<31>(_iml)
#define v_columnindex 31
#define _g _ml->template fpfield<32>(_iml)
#define _g_columnindex 32
#define _ion_ena *(_ml->dptr_field<0>(_iml))
#define _p_ion_ena static_cast<neuron::container::data_handle<double>>(_ppvar[0])
#define _ion_ina *(_ml->dptr_field<1>(_iml))
#define _p_ion_ina static_cast<neuron::container::data_handle<double>>(_ppvar[1])
#define _ion_dinadv *(_ml->dptr_field<2>(_iml))
 /* Thread safe. No static _ml, _iml or _ppvar. */
 static int hoc_nrnpointerindex =  -1;
 static _nrn_mechanism_std_vector<Datum> _extcall_thread;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_persist_gate(void);
 static void _hoc_rates2(void);
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 static void _hoc_setdata();
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {"setdata_na16a_vgp", _hoc_setdata},
 {"persist_gate_na16a_vgp", _hoc_persist_gate},
 {"rates2_na16a_vgp", _hoc_rates2},
 {"rates_na16a_vgp", _hoc_rates},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_persist_gate(Prop*);
 static double _npy_rates2(Prop*);
 static double _npy_rates(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"persist_gate", _npy_persist_gate},
 {"rates2", _npy_rates2},
 {"rates", _npy_rates},
 {0, 0}
};
#define persist_gate persist_gate_na16a_vgp
#define rates2 rates2_na16a_vgp
 extern double persist_gate( _internalthreadargsprotocomma_ double );
 extern double rates2( _internalthreadargsprotocomma_ double , double , double , double );
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
#define C1I1k2 C1I1k2_na16a_vgp
 double C1I1k2 = -11;
#define C1I1v2 C1I1v2_na16a_vgp
 double C1I1v2 = -65;
#define C1I1b2 C1I1b2_na16a_vgp
 double C1I1b2 = 0.2;
#define C1O1k2 C1O1k2_na16a_vgp
 double C1O1k2 = -6;
#define C1O1b2 C1O1b2_na16a_vgp
 double C1O1b2 = 14;
#define I2I1k1 I2I1k1_na16a_vgp
 double I2I1k1 = 12;
#define I2I1v1 I2I1v1_na16a_vgp
 double I2I1v1 = -50;
#define I2I1b1 I2I1b1_na16a_vgp
 double I2I1b1 = 0.0018;
#define I1I2k2 I1I2k2_na16a_vgp
 double I1I2k2 = -5;
#define I1I2v2 I1I2v2_na16a_vgp
 double I1I2v2 = -25;
#define I1I2b2 I1I2b2_na16a_vgp
 double I1I2b2 = 0.022;
#define I1C1k1 I1C1k1_na16a_vgp
 double I1C1k1 = 10;
#define I1C1v1 I1C1v1_na16a_vgp
 double I1C1v1 = -65;
#define I1C1b1 I1C1b1_na16a_vgp
 double I1C1b1 = 0.2;
#define O1I1k2 O1I1k2_na16a_vgp
 double O1I1k2 = -12;
#define O1I1v2 O1I1v2_na16a_vgp
 double O1I1v2 = 10;
#define O1I1b2 O1I1b2_na16a_vgp
 double O1I1b2 = 5;
#define O1I1k1 O1I1k1_na16a_vgp
 double O1I1k1 = 12;
#define O1I1v1 O1I1v1_na16a_vgp
 double O1I1v1 = -42;
#define O1I1b1 O1I1b1_na16a_vgp
 double O1I1b1 = 1;
#define O1C1k2 O1C1k2_na16a_vgp
 double O1C1k2 = -5.1;
#define O1C1v2 O1C1v2_na16a_vgp
 double O1C1v2 = 0;
#define O1C1b2 O1C1b2_na16a_vgp
 double O1C1b2 = 0;
#define O1C1k1 O1C1k1_na16a_vgp
 double O1C1k1 = 9;
#define O1C1v1 O1C1v1_na16a_vgp
 double O1C1v1 = -48;
#define O1C1b1 O1C1b1_na16a_vgp
 double O1C1b1 = 4;
#define p_k2 p_k2_na16a_vgp
 double p_k2 = 15;
#define p_vhalf2 p_vhalf2_na16a_vgp
 double p_vhalf2 = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"p_vhalf2_na16a_vgp", "mV"},
 {"p_k2_na16a_vgp", "mV"},
 {"gbar_na16a_vgp", "mho/cm2"},
 {"p_vhalf_na16a_vgp", "mV"},
 {"p_k_na16a_vgp", "mV"},
 {"ina_na16a_vgp", "mA/cm2"},
 {"g_na16a_vgp", "mho/cm2"},
 {0, 0}
};
 static double C10 = 0;
 static double I20 = 0;
 static double I10 = 0;
 static double O10 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"C1O1b2_na16a_vgp", &C1O1b2_na16a_vgp},
 {"C1O1k2_na16a_vgp", &C1O1k2_na16a_vgp},
 {"O1C1b1_na16a_vgp", &O1C1b1_na16a_vgp},
 {"O1C1v1_na16a_vgp", &O1C1v1_na16a_vgp},
 {"O1C1k1_na16a_vgp", &O1C1k1_na16a_vgp},
 {"O1C1b2_na16a_vgp", &O1C1b2_na16a_vgp},
 {"O1C1v2_na16a_vgp", &O1C1v2_na16a_vgp},
 {"O1C1k2_na16a_vgp", &O1C1k2_na16a_vgp},
 {"O1I1b1_na16a_vgp", &O1I1b1_na16a_vgp},
 {"O1I1v1_na16a_vgp", &O1I1v1_na16a_vgp},
 {"O1I1k1_na16a_vgp", &O1I1k1_na16a_vgp},
 {"O1I1b2_na16a_vgp", &O1I1b2_na16a_vgp},
 {"O1I1v2_na16a_vgp", &O1I1v2_na16a_vgp},
 {"O1I1k2_na16a_vgp", &O1I1k2_na16a_vgp},
 {"I1C1b1_na16a_vgp", &I1C1b1_na16a_vgp},
 {"I1C1v1_na16a_vgp", &I1C1v1_na16a_vgp},
 {"I1C1k1_na16a_vgp", &I1C1k1_na16a_vgp},
 {"C1I1b2_na16a_vgp", &C1I1b2_na16a_vgp},
 {"C1I1v2_na16a_vgp", &C1I1v2_na16a_vgp},
 {"C1I1k2_na16a_vgp", &C1I1k2_na16a_vgp},
 {"I1I2b2_na16a_vgp", &I1I2b2_na16a_vgp},
 {"I1I2v2_na16a_vgp", &I1I2v2_na16a_vgp},
 {"I1I2k2_na16a_vgp", &I1I2k2_na16a_vgp},
 {"I2I1b1_na16a_vgp", &I2I1b1_na16a_vgp},
 {"I2I1v1_na16a_vgp", &I2I1v1_na16a_vgp},
 {"I2I1k1_na16a_vgp", &I2I1k1_na16a_vgp},
 {"p_vhalf2_na16a_vgp", &p_vhalf2_na16a_vgp},
 {"p_k2_na16a_vgp", &p_k2_na16a_vgp},
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 _prop_id = _nrn_get_prop_id(_prop);
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 static void nrn_alloc(Prop*);
static void nrn_init(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_state(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 static void nrn_cur(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_jacob(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(Prop*, int, neuron::container::data_handle<double>*, neuron::container::data_handle<double>*, double*, int);
static void _ode_spec(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void _ode_matsol(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[3].literal_value<int>()
 static void _ode_matsol_instance1(_internalthreadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"na16a_vgp",
 "gbar_na16a_vgp",
 "I2init_na16a_vgp",
 "C1O1v2_na16a_vgp",
 "dist_na16a_vgp",
 "slowdown_na16a_vgp",
 "persist_na16a_vgp",
 "p_vhalf_na16a_vgp",
 "p_k_na16a_vgp",
 "fast_inactivation_tau_scale_na16a_vgp",
 "slow_recovery_tau_scale_na16a_vgp",
 0,
 "ina_na16a_vgp",
 "g_na16a_vgp",
 0,
 "C1_na16a_vgp",
 "O1_na16a_vgp",
 "I1_na16a_vgp",
 "I2_na16a_vgp",
 0,
 0};
 static Symbol* _na_sym;
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     0.1, /* gbar */
     0, /* I2init */
     0, /* C1O1v2 */
     0, /* dist */
     0.2, /* slowdown */
     0, /* persist */
     -50, /* p_vhalf */
     5, /* p_k */
     1, /* fast_inactivation_tau_scale */
     1, /* slow_recovery_tau_scale */
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
   _ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 33);
 	/*initialize range parameters*/
 	gbar = _parm_default[0]; /* 0.1 */
 	I2init = _parm_default[1]; /* 0 */
 	C1O1v2 = _parm_default[2]; /* 0 */
 	dist = _parm_default[3]; /* 0 */
 	slowdown = _parm_default[4]; /* 0.2 */
 	persist = _parm_default[5]; /* 0 */
 	p_vhalf = _parm_default[6]; /* -50 */
 	p_k = _parm_default[7]; /* 5 */
 	fast_inactivation_tau_scale = _parm_default[8]; /* 1 */
 	slow_recovery_tau_scale = _parm_default[9]; /* 1 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 33);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = _nrn_mechanism_get_param_handle(prop_ion, 0); /* ena */
 	_ppvar[1] = _nrn_mechanism_get_param_handle(prop_ion, 3); /* ina */
 	_ppvar[2] = _nrn_mechanism_get_param_handle(prop_ion, 4); /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _Nav16_a_vgated_persist_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 3);
  _extcall_thread.resize(2);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"gbar"} /* 0 */,
                                       _nrn_mechanism_field<double>{"I2init"} /* 1 */,
                                       _nrn_mechanism_field<double>{"C1O1v2"} /* 2 */,
                                       _nrn_mechanism_field<double>{"dist"} /* 3 */,
                                       _nrn_mechanism_field<double>{"slowdown"} /* 4 */,
                                       _nrn_mechanism_field<double>{"persist"} /* 5 */,
                                       _nrn_mechanism_field<double>{"p_vhalf"} /* 6 */,
                                       _nrn_mechanism_field<double>{"p_k"} /* 7 */,
                                       _nrn_mechanism_field<double>{"fast_inactivation_tau_scale"} /* 8 */,
                                       _nrn_mechanism_field<double>{"slow_recovery_tau_scale"} /* 9 */,
                                       _nrn_mechanism_field<double>{"ina"} /* 10 */,
                                       _nrn_mechanism_field<double>{"g"} /* 11 */,
                                       _nrn_mechanism_field<double>{"C1"} /* 12 */,
                                       _nrn_mechanism_field<double>{"O1"} /* 13 */,
                                       _nrn_mechanism_field<double>{"I1"} /* 14 */,
                                       _nrn_mechanism_field<double>{"I2"} /* 15 */,
                                       _nrn_mechanism_field<double>{"ena"} /* 16 */,
                                       _nrn_mechanism_field<double>{"C1O1_a"} /* 17 */,
                                       _nrn_mechanism_field<double>{"O1C1_a"} /* 18 */,
                                       _nrn_mechanism_field<double>{"O1I1_a"} /* 19 */,
                                       _nrn_mechanism_field<double>{"I1O1_a"} /* 20 */,
                                       _nrn_mechanism_field<double>{"I1I2_a"} /* 21 */,
                                       _nrn_mechanism_field<double>{"I2I1_a"} /* 22 */,
                                       _nrn_mechanism_field<double>{"I1C1_a"} /* 23 */,
                                       _nrn_mechanism_field<double>{"C1I1_a"} /* 24 */,
                                       _nrn_mechanism_field<double>{"Q10"} /* 25 */,
                                       _nrn_mechanism_field<double>{"p_gate"} /* 26 */,
                                       _nrn_mechanism_field<double>{"DC1"} /* 27 */,
                                       _nrn_mechanism_field<double>{"DO1"} /* 28 */,
                                       _nrn_mechanism_field<double>{"DI1"} /* 29 */,
                                       _nrn_mechanism_field<double>{"DI2"} /* 30 */,
                                       _nrn_mechanism_field<double>{"v"} /* 31 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 32 */,
                                       _nrn_mechanism_field<double*>{"_ion_ena", "na_ion"} /* 0 */,
                                       _nrn_mechanism_field<double*>{"_ion_ina", "na_ion"} /* 1 */,
                                       _nrn_mechanism_field<double*>{"_ion_dinadv", "na_ion"} /* 2 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 3 */);
  hoc_register_prop_size(_mechtype, 33, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 na16a_vgp /Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/Nav16_a_vgated_persist.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "Nav1.6 ionic voltage-gated channel with kinetic scheme (voltage-gated persistent variant)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_internalthreadargsprotocomma_ double);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(static_cast<SparseObj*>(_so), _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
 static int _cvspth1 = 1;
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(static_cast<SparseObj*>(_so), _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 0;
 static neuron::container::field_index _slist1[4], _dlist1[4]; static double *_temp1;
 static int kin (void* _so, double* _rhs, _internalthreadargsproto_);
 
static int kin (void* _so, double* _rhs, _internalthreadargsproto_)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<4;_i++){
  	_RHS1(_i) = -_dt1*(_ml->data(_iml, _slist1[_i]) - _ml->data(_iml, _dlist1[_i]));
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v ) ;
   /* ~ C1 <-> O1 ( C1O1_a , O1C1_a )*/
 f_flux =  C1O1_a * C1 ;
 b_flux =  O1C1_a * O1 ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  C1O1_a ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 3 ,1)  -= _term;
 _term =  O1C1_a ;
 _MATELM1( 1 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ O1 <-> I1 ( O1I1_a , I1O1_a )*/
 f_flux =  O1I1_a * O1 ;
 b_flux =  I1O1_a * I1 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  O1I1_a ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  I1O1_a ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ I1 <-> C1 ( I1C1_a , C1I1_a )*/
 f_flux =  I1C1_a * I1 ;
 b_flux =  C1I1_a * C1 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  I1C1_a ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  C1I1_a ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ I1 <-> I2 ( I1I2_a , I2I1_a )*/
 f_flux =  I1I2_a * I1 ;
 b_flux =  I2I1_a * I2 ;
 _RHS1( 2) -= (f_flux - b_flux);
 
 _term =  I1I2_a ;
 _MATELM1( 2 ,2)  += _term;
 _term =  I2I1_a ;
 _MATELM1( 2 ,0)  -= _term;
 /*REACTION*/
   /* O1 + C1 + I1 + I2 = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= I2 ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= I1 ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= C1 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= O1 ;
 /*CONSERVATION*/
   } return _reset;
 }
 
double rates2 ( _internalthreadargsprotocomma_ double _lv , double _lb , double _lvv , double _lk ) {
   double _lrates2;
 double _lArg ;
 _lArg = ( _lv - _lvv ) / _lk ;
   if ( _lArg < - 50.0 ) {
     _lrates2 = _lb ;
     }
   else if ( _lArg > 50.0 ) {
     _lrates2 = 0.0 ;
     }
   else {
     _lrates2 = ( _lb / ( 1.0 + exp ( _lArg ) ) ) ;
     }
   
return _lrates2;
 }
 
static void _hoc_rates2(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  rates2 ( _threadargscomma_ *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
static double _npy_rates2(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  rates2 ( _threadargscomma_ *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 return(_r);
}
 
double persist_gate ( _internalthreadargsprotocomma_ double _lv ) {
   double _lpersist_gate;
 double _lrising , _lfalling ;
 _lrising = rates2 ( _threadargscomma_ _lv , 1.0 , p_vhalf , - p_k ) ;
   _lfalling = rates2 ( _threadargscomma_ _lv , 1.0 , p_vhalf2 , p_k2 ) ;
   _lpersist_gate = _lrising * _lfalling ;
   
return _lpersist_gate;
 }
 
static void _hoc_persist_gate(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  if(!_prop_id) {
    hoc_execerror("No data for persist_gate_na16a_vgp. Requires prior call to setdata_na16a_vgp and that the specified mechanism instance still be in existence.", NULL);
  }
  Prop* _local_prop = _extcall_prop;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  persist_gate ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_persist_gate(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  persist_gate ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
static int  rates ( _internalthreadargsprotocomma_ double _lv ) {
    C1O1_a = Q10 * ( rates2 ( _threadargscomma_ _lv , C1O1b2 , C1O1v2 , C1O1k2 ) ) ;
   O1C1_a = Q10 * ( rates2 ( _threadargscomma_ _lv , O1C1b1 , O1C1v1 , O1C1k1 ) + rates2 ( _threadargscomma_ _lv , O1C1b2 , O1C1v2 , O1C1k2 ) ) ;
   O1I1_a = 0.5 * Q10 * ( rates2 ( _threadargscomma_ _lv , O1I1b1 , O1I1v1 , O1I1k1 ) + rates2 ( _threadargscomma_ _lv , O1I1b2 , O1I1v2 , O1I1k2 ) ) / fast_inactivation_tau_scale ;
   p_gate = persist_gate ( _threadargscomma_ _lv ) ;
   I1O1_a = persist * p_gate * O1I1_a ;
   I1C1_a = Q10 * ( rates2 ( _threadargscomma_ _lv , I1C1b1 , I1C1v1 , I1C1k1 ) ) / fast_inactivation_tau_scale ;
   C1I1_a = Q10 * ( rates2 ( _threadargscomma_ _lv , C1I1b2 , C1I1v2 , C1I1k2 ) ) / fast_inactivation_tau_scale ;
   I1I2_a = slowdown * dist * Q10 * ( rates2 ( _threadargscomma_ _lv , I1I2b2 , I1I2v2 , I1I2k2 ) ) / slow_recovery_tau_scale ;
   I2I1_a = slowdown * Q10 * ( rates2 ( _threadargscomma_ _lv , I2I1b1 , I2I1v1 , I2I1k1 ) ) / slow_recovery_tau_scale ;
     return 0; }
 
static void _hoc_rates(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  if(!_prop_id) {
    hoc_execerror("No data for rates_na16a_vgp. Requires prior call to setdata_na16a_vgp and that the specified mechanism instance still be in existence.", NULL);
  }
  Prop* _local_prop = _extcall_prop;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r = 1.;
 rates ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_rates(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r = 1.;
 rates ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(_internalthreadargsproto_) {
  int _reset=0;
  {
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<4;_i++) _ml->data(_iml, _dlist1[_i]) = 0.0;}
 rates ( _threadargscomma_ v ) ;
 /* ~ C1 <-> O1 ( C1O1_a , O1C1_a )*/
 f_flux =  C1O1_a * C1 ;
 b_flux =  O1C1_a * O1 ;
 DC1 -= (f_flux - b_flux);
 DO1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O1 <-> I1 ( O1I1_a , I1O1_a )*/
 f_flux =  O1I1_a * O1 ;
 b_flux =  I1O1_a * I1 ;
 DO1 -= (f_flux - b_flux);
 DI1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I1 <-> C1 ( I1C1_a , C1I1_a )*/
 f_flux =  I1C1_a * I1 ;
 b_flux =  C1I1_a * C1 ;
 DI1 -= (f_flux - b_flux);
 DC1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I1 <-> I2 ( I1I2_a , I2I1_a )*/
 f_flux =  I1I2_a * I1 ;
 b_flux =  I2I1_a * I2 ;
 DI1 -= (f_flux - b_flux);
 DI2 += (f_flux - b_flux);
 
 /*REACTION*/
   /* O1 + C1 + I1 + I2 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, _internalthreadargsproto_) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<4;_i++){
  	_RHS1(_i) = _dt1*(_ml->data(_iml, _dlist1[_i]));
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v ) ;
 /* ~ C1 <-> O1 ( C1O1_a , O1C1_a )*/
 _term =  C1O1_a ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 3 ,1)  -= _term;
 _term =  O1C1_a ;
 _MATELM1( 1 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ O1 <-> I1 ( O1I1_a , I1O1_a )*/
 _term =  O1I1_a ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  I1O1_a ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ I1 <-> C1 ( I1C1_a , C1I1_a )*/
 _term =  I1C1_a ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  C1I1_a ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ I1 <-> I2 ( I1I2_a , I2I1_a )*/
 _term =  I1I2_a ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 0 ,2)  -= _term;
 _term =  I2I1_a ;
 _MATELM1( 2 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
   /* O1 + C1 + I1 + I2 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 4;}
 
static void _ode_spec(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 (_threadargs_);
  }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  Datum* _ppvar;
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 4; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _cvode_sparse_thread(&(_thread[_cvspth1].literal_value<void*>()), 4, _dlist1, neuron::scopmath::row_view{_ml, _iml}, _ode_matsol1, _threadargs_);
 }
 
static void _ode_matsol(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(static_cast<SparseObj*>(_thread[_spth1].get<void*>()));
   _nrn_destroy_sparseobj_thread(static_cast<SparseObj*>(_thread[_cvspth1].get<void*>()));
 }

static void initmodel(_internalthreadargsproto_) {
  int _i; double _save;{
  C1 = C10;
  I2 = I20;
  I1 = I10;
  O1 = O10;
 {
   Q10 = pow( 3.0 , ( ( celsius - 20.0 ) / 10.0 ) ) ;
    _ss_sparse_thread(&(_thread[_spth1].literal_value<void*>()), 4, _slist1, _dlist1, neuron::scopmath::row_view{_ml, _iml}, &t, dt, kin, _linmat1, _threadargs_);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 4; ++_i) {
      _ml->data(_iml, _slist1[_i]) += dt*_ml->data(_iml, _dlist1[_i]);
    }}
 }
 
}
}

static void nrn_init(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
 v = _v;
  ena = _ion_ena;
 initmodel(_threadargs_);
 }
}

static double _nrn_current(_internalthreadargsprotocomma_ double _v) {
double _current=0.; v=_v;
{ {
   g = gbar * ( O1 ) ;
   ina = g * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_rhs = _nt->node_rhs_storage();
auto const _vec_sav_rhs = _nt->node_sav_rhs_storage();
auto const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
  ena = _ion_ena;
 auto const _g_local = _nrn_current(_threadargscomma_ _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_threadargscomma_ _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g_local - _rhs)/.001;
  _ion_ina += ina ;
	 _vec_rhs[_ni[_iml]] -= _rhs;
 
}
 
}

static void nrn_jacob(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_d = _nt->node_d_storage();
auto const _vec_sav_d = _nt->node_sav_d_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
  _vec_d[_ni[_iml]] += _g;
 
}
 
}

static void nrn_state(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
_ni = _ml_arg->_nodeindices;
size_t _cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (size_t _iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _nd = _ml_arg->_nodelist[_iml];
   _v = _vec_v[_ni[_iml]];
 v=_v;
{
  ena = _ion_ena;
 {  sparse_thread(&(_thread[_spth1].literal_value<void*>()), 4, _slist1, _dlist1, neuron::scopmath::row_view{_ml, _iml}, &t, dt, kin, _linmat1, _threadargs_);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 4; ++_i) {
      _ml->data(_iml, _slist1[_i]) += dt*_ml->data(_iml, _dlist1[_i]);
    }}
 } }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = {I2_columnindex, 0};  _dlist1[0] = {DI2_columnindex, 0};
 _slist1[1] = {C1_columnindex, 0};  _dlist1[1] = {DC1_columnindex, 0};
 _slist1[2] = {I1_columnindex, 0};  _dlist1[2] = {DI1_columnindex, 0};
 _slist1[3] = {O1_columnindex, 0};  _dlist1[3] = {DO1_columnindex, 0};
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/Nav16_a_vgated_persist.mod";
    const char* nmodl_file_text = 
  "TITLE Nav1.6 ionic voltage-gated channel with kinetic scheme (voltage-gated persistent variant)\n"
  "\n"
  "COMMENT\n"
  "A six-state markovian kinetic model of ionic channel.\n"
  "Part of a study on kinetic models.\n"
  "Author: Piero Balbi, August 2016\n"
  "With Changes by Dr. Christopher Knowlton and Carol Upchurch\n"
  "\n"
  "MODIFICATION (this file):\n"
  "The original 'persist' parameter scaled the I1->O1 leak-back rate\n"
  "(I1O1_a = persist*O1I1_a) uniformly at all voltages. This let a\n"
  "fixed fraction of inactivated channels recycle into the open state\n"
  "even at hyperpolarized/resting potentials, where no persistent Na\n"
  "current should exist physiologically. That produced depolarization\n"
  "block and spontaneous (stimulus-independent) oscillations when\n"
  "persist was tuned to fit the depolarized plateau.\n"
  "\n"
  "Here the leak-back rate is additionally multiplied by a sigmoidal\n"
  "voltage gate, persist_gate(v), parameterized by p_vhalf and p_k.\n"
  "The gate is close to 0 at rest/hyperpolarized potentials and close\n"
  "to 1 near and above spike threshold, so the persistent component\n"
  "only engages in the voltage range where it is meant to contribute\n"
  "(e.g. during a depolarizing step or the interspike plateau), and\n"
  "switches off automatically as the membrane repolarizes -- without\n"
  "requiring any explicit stimulus-timing information.\n"
  "\n"
  "See accompanying references for the biophysical motivation:\n"
  "persistent Na current in real neurons is itself voltage-gated,\n"
  "typically activating over roughly -65 to -40 mV, so this brings the\n"
  "mechanism in line with experimentally characterized I_NaP rather\n"
  "than treating 'persist' as a constant offset.\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX na16a_vgp\n"
  "    USEION na READ ena WRITE ina\n"
  "    RANGE gbar, ina, g, dist, persist, slowdown, C1O1v2, I2init\n"
  "    RANGE p_vhalf, p_k, fast_inactivation_tau_scale, slow_recovery_tau_scale\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    v (mV)\n"
  "    ena (mV)\n"
  "    celsius (degC)\n"
  "    gbar  = 0.1  (mho/cm2)\n"
  "    I2init=0\n"
  "\n"
  "    C1O1b2    = 14\n"
  "    C1O1v2    = 0       :0-18\n"
  "    C1O1k2    = -6      :-10\n"
  "    O1C1b1    = 4\n"
  "    O1C1v1    = -48\n"
  "    O1C1k1    = 9\n"
  "    O1C1b2    = 0        :14\n"
  "    O1C1v2    = 0         :-18\n"
  "    O1C1k2    = -5.1      :-10\n"
  "    O1I1b1    = 1         :6\n"
  "    O1I1v1    = -42        :-40\n"
  "    O1I1k1    = 12         :13\n"
  "    O1I1b2    = 5      :10\n"
  "    O1I1v2    = 10      :15\n"
  "    O1I1k2    = -12     :-18\n"
  "    I1C1b1    = 0.2     :0.1\n"
  "    I1C1v1    = -65     :-86\n"
  "    I1C1k1    = 10      :9\n"
  "\n"
  "    C1I1b2    = 0.2     :0.08\n"
  "    C1I1v2    = -65     :-55\n"
  "    C1I1k2    = -11     :-12\n"
  "    I1I2b2    = 0.022   :0.00022\n"
  "    I1I2v2    = -25     :-25\n"
  "    I1I2k2    = -5      :-5 -2\n"
  "\n"
  "    I2I1b1    = 0.0018\n"
  "    I2I1v1    = -50         :-50    -40\n"
  "    I2I1k1    = 12          :12 1\n"
  "    dist = 0\n"
  "    slowdown = 0.2\n"
  "\n"
  "    persist = 0          : max fraction of O1I1_a leaked back to O1\n"
  "    p_vhalf = -50  (mV)  : half-activation voltage of the persistent gate\n"
  "    p_k     = 5    (mV)  : slope of the persistent gate (mV); larger = softer onset\n"
  "    p_vhalf2 = 0   (mV)\n"
  "    p_k2     = 15  (mV)\n"
  "    fast_inactivation_tau_scale = 1\n"
  "    slow_recovery_tau_scale = 1\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    ina  (mA/cm2)\n"
  "    g   (mho/cm2)\n"
  "\n"
  "    C1O1_a (/ms)\n"
  "    O1C1_a (/ms)\n"
  "    O1I1_a (/ms)\n"
  "    I1O1_a (/ms)\n"
  "    I1I2_a (/ms)\n"
  "    I2I1_a (/ms)\n"
  "    I1C1_a (/ms)\n"
  "    C1I1_a (/ms)\n"
  "\n"
  "    Q10 (1)\n"
  "    p_gate (1)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    C1\n"
  "    O1\n"
  "    I1\n"
  "    I2\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "    Q10 = 3^((celsius-20(degC))/10 (degC))\n"
  "    SOLVE kin\n"
  "    STEADYSTATE sparse\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE kin METHOD sparse\n"
  "    g = gbar * (O1) : (mho/cm2)\n"
  "    ina = g * (v - ena)     : (mA/cm2)\n"
  "}\n"
  "\n"
  "KINETIC kin {\n"
  "    rates(v)\n"
  "\n"
  "    ~ C1 <->  O1 (C1O1_a, O1C1_a)\n"
  "    ~ O1 <->  I1 (O1I1_a, I1O1_a)\n"
  "    ~ I1 <->  C1 (I1C1_a, C1I1_a)\n"
  "    ~ I1 <->  I2 (I1I2_a, I2I1_a)\n"
  "\n"
  "    CONSERVE O1 + C1 + I1 + I2= 1\n"
  "}\n"
  "\n"
  "FUNCTION rates2(v, b, vv, k) {\n"
  "    LOCAL Arg\n"
  "    Arg=(v-vv)/k\n"
  "    if (Arg<-50) {rates2=b}\n"
  "    else if (Arg>50) {rates2=0}\n"
  "    else {rates2 = (b/(1+exp(Arg)))}\n"
  "}\n"
  "\n"
  "FUNCTION persist_gate(v(mV)) {\n"
  "    LOCAL rising, falling\n"
  "    rising  = rates2(v, 1, p_vhalf,  -p_k)   : ~0 below p_vhalf, ~1 above\n"
  "    falling = rates2(v, 1, p_vhalf2, p_k2)   : ~1 below p_vhalf2, ~0 above\n"
  "    persist_gate = rising*falling\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "UNITSOFF\n"
  "\n"
  "    C1O1_a = Q10*(rates2(v, C1O1b2, C1O1v2, C1O1k2))\n"
  "    O1C1_a = Q10*(rates2(v, O1C1b1, O1C1v1, O1C1k1) + rates2(v, O1C1b2, O1C1v2, O1C1k2))\n"
  "    O1I1_a = 0.5*Q10*(rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2)) / fast_inactivation_tau_scale\n"
  "\n"
  "    p_gate = persist_gate(v)\n"
  "    I1O1_a = persist*p_gate*O1I1_a\n"
  "\n"
  "    I1C1_a = Q10*(rates2(v, I1C1b1, I1C1v1, I1C1k1)) / fast_inactivation_tau_scale\n"
  "    C1I1_a = Q10*(rates2(v, C1I1b2, C1I1v2, C1I1k2)) / fast_inactivation_tau_scale\n"
  "    I1I2_a = slowdown*dist*Q10*(rates2(v, I1I2b2, I1I2v2, I1I2k2)) / slow_recovery_tau_scale\n"
  "    I2I1_a = slowdown*Q10*(rates2(v, I2I1b1, I2I1v1, I2I1k1)) / slow_recovery_tau_scale\n"
  "UNITSON\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
