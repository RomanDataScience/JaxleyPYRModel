/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
#include <vector>
using std::size_t;
static auto& std_cerr_stream = std::cerr;
static constexpr auto number_of_datum_variables = 3;
static constexpr auto number_of_floating_point_variables = 12;
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
 
#define nrn_init _nrn_init__kad
#define _nrn_initial _nrn_initial__kad
#define nrn_cur _nrn_cur__kad
#define _nrn_current _nrn_current__kad
#define nrn_jacob _nrn_jacob__kad
#define nrn_state _nrn_state__kad
#define _net_receive _net_receive__kad 
#define rates rates__kad 
#define states states__kad 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _internalthreadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
#define _internalthreadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gkabar _ml->template fpfield<0>(_iml)
#define gkabar_columnindex 0
#define vhalfn _ml->template fpfield<1>(_iml)
#define vhalfn_columnindex 1
#define vhalfl _ml->template fpfield<2>(_iml)
#define vhalfl_columnindex 2
#define i _ml->template fpfield<3>(_iml)
#define i_columnindex 3
#define gka _ml->template fpfield<4>(_iml)
#define gka_columnindex 4
#define n _ml->template fpfield<5>(_iml)
#define n_columnindex 5
#define l _ml->template fpfield<6>(_iml)
#define l_columnindex 6
#define ek _ml->template fpfield<7>(_iml)
#define ek_columnindex 7
#define Dn _ml->template fpfield<8>(_iml)
#define Dn_columnindex 8
#define Dl _ml->template fpfield<9>(_iml)
#define Dl_columnindex 9
#define ik _ml->template fpfield<10>(_iml)
#define ik_columnindex 10
#define _g _ml->template fpfield<11>(_iml)
#define _g_columnindex 11
#define _ion_ek *(_ml->dptr_field<0>(_iml))
#define _p_ion_ek static_cast<neuron::container::data_handle<double>>(_ppvar[0])
#define _ion_ik *(_ml->dptr_field<1>(_iml))
#define _p_ion_ik static_cast<neuron::container::data_handle<double>>(_ppvar[1])
#define _ion_dikdv *(_ml->dptr_field<2>(_iml))
 static _nrn_mechanism_cache_instance _ml_real{nullptr};
static _nrn_mechanism_cache_range *_ml{&_ml_real};
static size_t _iml{0};
static Datum *_ppvar;
 static int hoc_nrnpointerindex =  -1;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_alpl(void);
 static void _hoc_alpn(void);
 static void _hoc_betl(void);
 static void _hoc_betn(void);
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
 {"setdata_kad", _hoc_setdata},
 {"alpl_kad", _hoc_alpl},
 {"alpn_kad", _hoc_alpn},
 {"betl_kad", _hoc_betl},
 {"betn_kad", _hoc_betn},
 {"rates_kad", _hoc_rates},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_alpl(Prop*);
 static double _npy_alpn(Prop*);
 static double _npy_betl(Prop*);
 static double _npy_betn(Prop*);
 static double _npy_rates(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"alpl", _npy_alpl},
 {"alpn", _npy_alpn},
 {"betl", _npy_betl},
 {"betn", _npy_betn},
 {"rates", _npy_rates},
 {0, 0}
};
#define alpl alpl_kad
#define alpn alpn_kad
#define betl betl_kad
#define betn betn_kad
 extern double alpl( double );
 extern double alpn( double );
 extern double betl( double );
 extern double betn( double );
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
#define a0n a0n_kad
 double a0n = 0.1;
#define a0l a0l_kad
 double a0l = 0.05;
#define gml gml_kad
 double gml = 1;
#define gmn gmn_kad
 double gmn = 0.39;
#define linf linf_kad
 double linf = 0;
#define lmin lmin_kad
 double lmin = 2;
#define ninf ninf_kad
 double ninf = 0;
#define nmin nmin_kad
 double nmin = 0.2;
#define pw pw_kad
 double pw = -1;
#define qtl qtl_kad
 double qtl = 1;
#define q10 q10_kad
 double q10 = 5;
#define qq qq_kad
 double qq = 5;
#define taun taun_kad
 double taun = 0;
#define taul taul_kad
 double taul = 0;
#define tq tq_kad
 double tq = -40;
#define zetal zetal_kad
 double zetal = 3;
#define zetan zetan_kad
 double zetan = -1.8;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"a0l_kad", "/ms"},
 {"a0n_kad", "/ms"},
 {"zetan_kad", "1"},
 {"zetal_kad", "1"},
 {"gmn_kad", "1"},
 {"gml_kad", "1"},
 {"lmin_kad", "mS"},
 {"nmin_kad", "mS"},
 {"pw_kad", "1"},
 {"gkabar_kad", "mho/cm2"},
 {"vhalfn_kad", "mV"},
 {"vhalfl_kad", "mV"},
 {"i_kad", "mA/cm2"},
 {0, 0}
};
 static double delta_t = 0.01;
 static double l0 = 0;
 static double n0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"a0l_kad", &a0l_kad},
 {"a0n_kad", &a0n_kad},
 {"zetan_kad", &zetan_kad},
 {"zetal_kad", &zetal_kad},
 {"gmn_kad", &gmn_kad},
 {"gml_kad", &gml_kad},
 {"lmin_kad", &lmin_kad},
 {"nmin_kad", &nmin_kad},
 {"pw_kad", &pw_kad},
 {"tq_kad", &tq_kad},
 {"qq_kad", &qq_kad},
 {"q10_kad", &q10_kad},
 {"qtl_kad", &qtl_kad},
 {"ninf_kad", &ninf_kad},
 {"linf_kad", &linf_kad},
 {"taul_kad", &taul_kad},
 {"taun_kad", &taun_kad},
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
 neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
_ppvar = _nrn_mechanism_access_dparam(_prop);
 Node * _node = _nrn_mechanism_access_node(_prop);
v = _nrn_mechanism_access_voltage(_node);
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
"kad",
 "gkabar_kad",
 "vhalfn_kad",
 "vhalfl_kad",
 0,
 "i_kad",
 "gka_kad",
 0,
 "n_kad",
 "l_kad",
 0,
 0};
 static Symbol* _k_sym;
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     0.008, /* gkabar */
     -1, /* vhalfn */
     -56, /* vhalfl */
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
    assert(_nrn_mechanism_get_num_vars(_prop) == 12);
 	/*initialize range parameters*/
 	gkabar = _parm_default[0]; /* 0.008 */
 	vhalfn = _parm_default[1]; /* -1 */
 	vhalfl = _parm_default[2]; /* -56 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 12);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0] = _nrn_mechanism_get_param_handle(prop_ion, 0); /* ek */
 	_ppvar[1] = _nrn_mechanism_get_param_handle(prop_ion, 3); /* ik */
 	_ppvar[2] = _nrn_mechanism_get_param_handle(prop_ion, 4); /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _kadist_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"gkabar"} /* 0 */,
                                       _nrn_mechanism_field<double>{"vhalfn"} /* 1 */,
                                       _nrn_mechanism_field<double>{"vhalfl"} /* 2 */,
                                       _nrn_mechanism_field<double>{"i"} /* 3 */,
                                       _nrn_mechanism_field<double>{"gka"} /* 4 */,
                                       _nrn_mechanism_field<double>{"n"} /* 5 */,
                                       _nrn_mechanism_field<double>{"l"} /* 6 */,
                                       _nrn_mechanism_field<double>{"ek"} /* 7 */,
                                       _nrn_mechanism_field<double>{"Dn"} /* 8 */,
                                       _nrn_mechanism_field<double>{"Dl"} /* 9 */,
                                       _nrn_mechanism_field<double>{"ik"} /* 10 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 11 */,
                                       _nrn_mechanism_field<double*>{"_ion_ek", "k_ion"} /* 0 */,
                                       _nrn_mechanism_field<double*>{"_ion_ik", "k_ion"} /* 1 */,
                                       _nrn_mechanism_field<double*>{"_ion_dikdv", "k_ion"} /* 2 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 3 */);
  hoc_register_prop_size(_mechtype, 12, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kad /Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/kadist.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "K-A channel from Klee Ficker and Heinemann";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 static neuron::container::field_index _slist1[2], _dlist1[2];
 static int states(_internalthreadargsproto_);
 
double alpn (  double _lv ) {
   double _lalpn;
 double _lzeta , _lArg ;
 _lArg = ( _lv - tq ) / qq ;
   if ( _lArg < - 50.0 ) {
     _lzeta = zetan + pw ;
     }
   else if ( _lArg > 50.0 ) {
     _lzeta = zetan ;
     }
   else {
     _lzeta = zetan + pw / ( 1.0 + exp ( _lArg ) ) ;
     }
   _lArg = 1.e-3 * _lzeta * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ;
   if ( _lArg < - 50.0 ) {
     _lalpn = 0.0 ;
     }
   else if ( _lArg > 50.0 ) {
     _lalpn = exp ( 50.0 ) ;
     }
   else {
     _lalpn = exp ( _lArg ) ;
     }
   
return _lalpn;
 }
 
static void _hoc_alpn(void) {
  double _r;
  
  if(!_prop_id) {
    hoc_execerror("No data for alpn_kad. Requires prior call to setdata_kad and that the specified mechanism instance still be in existence.", NULL);
  } else {
    _setdata(_extcall_prop);
  }
   _r =  alpn (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_alpn(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r =  alpn (  *getarg(1) );
 return(_r);
}
 
double betn (  double _lv ) {
   double _lbetn;
 double _lzeta , _lArg ;
 _lArg = ( _lv - tq ) / qq ;
   if ( _lArg < - 50.0 ) {
     _lzeta = zetan + pw ;
     }
   else if ( _lArg > 50.0 ) {
     _lzeta = zetan ;
     }
   else {
     _lzeta = zetan + pw / ( 1.0 + exp ( _lArg ) ) ;
     }
   _lArg = 1.e-3 * _lzeta * gmn * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ;
   if ( _lArg < - 50.0 ) {
     _lbetn = 0.0 ;
     }
   else if ( _lArg > 50.0 ) {
     _lbetn = exp ( 50.0 ) ;
     }
   else {
     _lbetn = exp ( _lArg ) ;
     }
   
return _lbetn;
 }
 
static void _hoc_betn(void) {
  double _r;
  
  if(!_prop_id) {
    hoc_execerror("No data for betn_kad. Requires prior call to setdata_kad and that the specified mechanism instance still be in existence.", NULL);
  } else {
    _setdata(_extcall_prop);
  }
   _r =  betn (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_betn(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r =  betn (  *getarg(1) );
 return(_r);
}
 
double alpl (  double _lv ) {
   double _lalpl;
 double _lArg ;
 _lArg = 1.e-3 * zetal * ( _lv - vhalfl ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ;
   if ( _lArg < - 50.0 ) {
     _lalpl = 0.0 ;
     }
   else if ( _lArg > 50.0 ) {
     _lalpl = exp ( 50.0 ) ;
     }
   else {
     _lalpl = exp ( _lArg ) ;
     }
   
return _lalpl;
 }
 
static void _hoc_alpl(void) {
  double _r;
  
  if(!_prop_id) {
    hoc_execerror("No data for alpl_kad. Requires prior call to setdata_kad and that the specified mechanism instance still be in existence.", NULL);
  } else {
    _setdata(_extcall_prop);
  }
   _r =  alpl (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_alpl(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r =  alpl (  *getarg(1) );
 return(_r);
}
 
double betl (  double _lv ) {
   double _lbetl;
 double _lArg ;
 _lArg = 1.e-3 * zetal * gml * ( _lv - vhalfl ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ;
   if ( _lArg < - 50.0 ) {
     _lbetl = 0.0 ;
     }
   else if ( _lArg > 50.0 ) {
     _lbetl = exp ( 50.0 ) ;
     }
   else {
     _lbetl = exp ( _lArg ) ;
     }
   
return _lbetl;
 }
 
static void _hoc_betl(void) {
  double _r;
  
  if(!_prop_id) {
    hoc_execerror("No data for betl_kad. Requires prior call to setdata_kad and that the specified mechanism instance still be in existence.", NULL);
  } else {
    _setdata(_extcall_prop);
  }
   _r =  betl (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_betl(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r =  betl (  *getarg(1) );
 return(_r);
}
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / taun ;
   Dl = ( linf - l ) / taul ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taun )) ;
 Dl = Dl  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taul )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taun)))*(- ( ( ( ninf ) ) / taun ) / ( ( ( ( - 1.0 ) ) ) / taun ) - n) ;
    l = l + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taul)))*(- ( ( ( linf ) ) / taul ) / ( ( ( ( - 1.0 ) ) ) / taul ) - l) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   double _la , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 24.0 ) / 10.0 ) ) ;
   _la = alpn ( _threadargscomma_ _lv ) ;
   ninf = 1.0 / ( 1.0 + _la ) ;
   taun = betn ( _threadargscomma_ _lv ) / ( _lqt * a0n * ( 1.0 + _la ) ) ;
   if ( taun < nmin ) {
     taun = nmin ;
     }
   _la = alpl ( _threadargscomma_ _lv ) ;
   linf = 1.0 / ( 1.0 + _la ) ;
   taul = 0.26 * ( _lv + 50.0 ) / qtl ;
   if ( taul < lmin / qtl ) {
     taul = lmin / qtl ;
     }
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
  
  if(!_prop_id) {
    hoc_execerror("No data for rates_kad. Requires prior call to setdata_kad and that the specified mechanism instance still be in existence.", NULL);
  } else {
    _setdata(_extcall_prop);
  }
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_rates(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r = 1.;
 rates (  *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
      Node* _nd{};
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
  ek = _ion_ek;
     _ode_spec1 ();
  }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 2; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
      Node* _nd{};
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
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  l = l0;
  n = n0;
 {
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
   l = linf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
Node *_nd; double _v; int* _ni; int _cntml;
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
_ml = &_lmr;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
 v = _v;
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gka = gkabar * n * l ;
   i = gka * ( v - ek ) ;
   ik = i ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_rhs = _nt->node_rhs_storage();
auto const _vec_sav_rhs = _nt->node_sav_rhs_storage();
auto const _vec_v = _nt->node_voltage_storage();
Node *_nd; int* _ni; double _rhs, _v; int _cntml;
_ml = &_lmr;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
  ek = _ion_ek;
 auto const _g_local = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g_local - _rhs)/.001;
  _ion_ik += ik ;
	 _vec_rhs[_ni[_iml]] -= _rhs;
 
}}

static void nrn_jacob(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_d = _nt->node_d_storage();
auto const _vec_sav_d = _nt->node_sav_d_storage();
auto* const _ml = &_lmr;
Node *_nd; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
  _vec_d[_ni[_iml]] += _g;
 
}}

static void nrn_state(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _cntml;
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
_ml = &_lmr;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _nd = _ml_arg->_nodelist[_iml];
   _v = _vec_v[_ni[_iml]];
 v=_v;
{
  ek = _ion_ek;
 { error =  states();
 if(error){
  std_cerr_stream << "at line 55 in file kadist.mod:\nBREAKPOINT {\n";
  std_cerr_stream << _ml << ' ' << _iml << '\n';
  abort_run(error);
}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = {n_columnindex, 0};  _dlist1[0] = {Dn_columnindex, 0};
 _slist1[1] = {l_columnindex, 0};  _dlist1[1] = {Dl_columnindex, 0};
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/kadist.mod";
    const char* nmodl_file_text = 
  "TITLE K-A channel from Klee Ficker and Heinemann\n"
  ": modified to account for Dax A Current ----------\n"
  ": M.Migliore Jun 1997\n"
  "\n"
  "UNITS {\n"
  "        (mA) = (milliamp)\n"
  "        (mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	celsius\n"
  "        v (mV)\n"
  "        gkabar=.008 (mho/cm2)\n"
  "        vhalfn=-1   (mV)\n"
  "        vhalfl=-56   (mV)\n"
  "        a0l=0.05      (/ms)\n"
  "        a0n=.1    (/ms)\n"
  "        zetan=-1.8    (1)\n"
  "        zetal=3    (1)\n"
  "        gmn=0.39   (1)\n"
  "        gml=1   (1)\n"
  "        lmin=2  (mS)\n"
  "        nmin=0.2  (mS)\n"
  "        pw=-1    (1)\n"
  "        tq=-40\n"
  "        qq=5\n"
  "        q10=5\n"
  "        qtl=1\n"
  "	ek\n"
  "}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "        SUFFIX kad\n"
  "        USEION k READ ek WRITE ik\n"
  "        RANGE gkabar,gka,vhalfn,vhalfl,i\n"
  "        GLOBAL ninf,linf,taul,taun,lmin\n"
  "}\n"
  "\n"
  "STATE {\n"
  "        n\n"
  "        l\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "        ik (mA/cm2)\n"
  "        i (mA/cm2)\n"
  "        ninf\n"
  "        linf      \n"
  "        taul\n"
  "        taun\n"
  "        gka\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD cnexp\n"
  "        gka = gkabar*n*l\n"
  "        i = gka*(v-ek)\n"
  "        ik = i\n"
  "\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	n=ninf\n"
  "	l=linf\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION alpn(v(mV)) {\n"
  "    LOCAL zeta,Arg\n"
  "    Arg=(v-tq)/qq\n"
  "    \n"
  "    if (Arg<-50) {zeta=zetan+pw}\n"
  "    else if (Arg>50) {zeta=zetan}\n"
  "    else {zeta=zetan+pw/(1+exp(Arg))}\n"
  "    \n"
  "    Arg=1.e-3*zeta*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))\n"
  "    \n"
  "    if (Arg<-50) {alpn=0}\n"
  "    else if (Arg>50) {alpn=exp(50)}\n"
  "    else {alpn=exp(Arg)}\n"
  "}\n"
  "\n"
  "FUNCTION betn(v(mV)) {\n"
  "    LOCAL zeta,Arg\n"
  "    Arg=(v-tq)/qq\n"
  "    \n"
  "    if (Arg<-50) {zeta=zetan+pw}\n"
  "    else if (Arg>50) {zeta=zetan}\n"
  "    else {zeta=zetan+pw/(1+exp(Arg))}\n"
  "    \n"
  "    Arg=1.e-3*zeta*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))\n"
  "    \n"
  "    if (Arg<-50) {betn=0}\n"
  "    else if (Arg>50) {betn=exp(50)}\n"
  "    else {betn=exp(Arg)}\n"
  "}\n"
  "\n"
  "FUNCTION alpl(v(mV)) {\n"
  "    LOCAL Arg\n"
  "    Arg=1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))\n"
  "    \n"
  "    if (Arg<-50) {alpl=0}\n"
  "    else if (Arg>50) {alpl=exp(50)}\n"
  "    else {alpl=exp(Arg)}\n"
  "}\n"
  "\n"
  "FUNCTION betl(v(mV)) {\n"
  "    LOCAL Arg\n"
  "    Arg=1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))\n"
  "    \n"
  "    if (Arg<-50) {betl=0}\n"
  "    else if (Arg>50) {betl=exp(50)}\n"
  "    else {betl=exp(Arg)}\n"
  "}\n"
  "\n"
  "DERIVATIVE states {  \n"
  "        rates(v)\n"
  "        n' = (ninf - n)/taun\n"
  "        l' = (linf - l)/taul\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) { :callable from hoc\n"
  "        LOCAL a,qt\n"
  "        qt=q10^((celsius-24)/10)\n"
  "        a = alpn(v)\n"
  "        ninf = 1/(1 + a)\n"
  "        taun = betn(v)/(qt*a0n*(1+a))\n"
  "        if (taun<nmin) {taun=nmin}\n"
  "        a = alpl(v)\n"
  "        linf = 1/(1+ a)\n"
  "        taul = 0.26*(v+50)/qtl\n"
  "        if (taul<lmin/qtl) {taul=lmin/qtl}\n"
  "}\n"
  "\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
