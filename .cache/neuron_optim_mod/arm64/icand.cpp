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
static constexpr auto number_of_datum_variables = 1;
static constexpr auto number_of_floating_point_variables = 17;
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
 
#define nrn_init _nrn_init__icand
#define _nrn_initial _nrn_initial__icand
#define nrn_cur _nrn_cur__icand
#define _nrn_current _nrn_current__icand
#define nrn_jacob _nrn_jacob__icand
#define nrn_state _nrn_state__icand
#define _net_receive _net_receive__icand 
#define evaluate_fct evaluate_fct__icand 
#define states states__icand 
 
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
#define depth _ml->template fpfield<0>(_iml)
#define depth_columnindex 0
#define taur _ml->template fpfield<1>(_iml)
#define taur_columnindex 1
#define erev _ml->template fpfield<2>(_iml)
#define erev_columnindex 2
#define can _ml->template fpfield<3>(_iml)
#define can_columnindex 3
#define gbar _ml->template fpfield<4>(_iml)
#define gbar_columnindex 4
#define taumin _ml->template fpfield<5>(_iml)
#define taumin_columnindex 5
#define concrelease _ml->template fpfield<6>(_iml)
#define concrelease_columnindex 6
#define itrpm4 _ml->template fpfield<7>(_iml)
#define itrpm4_columnindex 7
#define Po _ml->template fpfield<8>(_iml)
#define Po_columnindex 8
#define DPo _ml->template fpfield<9>(_iml)
#define DPo_columnindex 9
#define jip3p _ml->template fpfield<10>(_iml)
#define jip3p_columnindex 10
#define ican _ml->template fpfield<11>(_iml)
#define ican_columnindex 11
#define drive_channel _ml->template fpfield<12>(_iml)
#define drive_channel_columnindex 12
#define Po_inf _ml->template fpfield<13>(_iml)
#define Po_inf_columnindex 13
#define Tau _ml->template fpfield<14>(_iml)
#define Tau_columnindex 14
#define v _ml->template fpfield<15>(_iml)
#define v_columnindex 15
#define _g _ml->template fpfield<16>(_iml)
#define _g_columnindex 16
#define diam	(*(_ml->dptr_field<0>(_iml)))
 /* Thread safe. No static _ml, _iml or _ppvar. */
 static int hoc_nrnpointerindex =  -1;
 static _nrn_mechanism_std_vector<Datum> _extcall_thread;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_MyExp(void);
 static void _hoc_evaluate_fct(void);
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
 {"setdata_icand", _hoc_setdata},
 {"MyExp_icand", _hoc_MyExp},
 {"evaluate_fct_icand", _hoc_evaluate_fct},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_MyExp(Prop*);
 static double _npy_evaluate_fct(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"MyExp", _npy_MyExp},
 {"evaluate_fct", _npy_evaluate_fct},
 {0, 0}
};
#define MyExp MyExp_icand
 extern double MyExp( _internalthreadargsprotocomma_ double );
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
#define Kd Kd_icand
 double Kd = 0.087;
#define cai cai_icand
 double cai = 0;
#define ica ica_icand
 double ica = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"cai_icand", "mM"},
 {"ica_icand", "mA/cm2"},
 {"Kd_icand", "mM"},
 {"depth_icand", "um"},
 {"taur_icand", "ms"},
 {"erev_icand", "mV"},
 {"can_icand", "mM"},
 {"gbar_icand", "mho/cm2"},
 {"taumin_icand", "ms"},
 {"itrpm4_icand", "mA/cm2"},
 {0, 0}
};
 static double Po0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"cai_icand", &cai_icand},
 {"ica_icand", &ica_icand},
 {"Kd_icand", &Kd_icand},
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
 
#define _cvode_ieq _ppvar[1].literal_value<int>()
 static void _ode_matsol_instance1(_internalthreadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"icand",
 "depth_icand",
 "taur_icand",
 "erev_icand",
 "can_icand",
 "gbar_icand",
 "taumin_icand",
 "concrelease_icand",
 0,
 "itrpm4_icand",
 0,
 "Po_icand",
 0,
 0};
 static Symbol* _morphology_sym;
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     0.0125, /* depth */
     80, /* taur */
     0, /* erev */
     0, /* can */
     0.0001, /* gbar */
     0.1, /* taumin */
     500, /* concrelease */
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
   _ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 17);
 	/*initialize range parameters*/
 	depth = _parm_default[0]; /* 0.0125 */
 	taur = _parm_default[1]; /* 80 */
 	erev = _parm_default[2]; /* 0 */
 	can = _parm_default[3]; /* 0 */
 	gbar = _parm_default[4]; /* 0.0001 */
 	taumin = _parm_default[5]; /* 0.1 */
 	concrelease = _parm_default[6]; /* 500 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 17);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_morphology_sym);
 	_ppvar[0] = _nrn_mechanism_get_param_handle(prop_ion, 0); /* diam */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _icand_reg() {
	int _vectorized = 1;
  _initlists();
 	_morphology_sym = hoc_lookup("morphology");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 5);
  _extcall_thread.resize(4);
  _thread_mem_init(_extcall_thread.data());
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"depth"} /* 0 */,
                                       _nrn_mechanism_field<double>{"taur"} /* 1 */,
                                       _nrn_mechanism_field<double>{"erev"} /* 2 */,
                                       _nrn_mechanism_field<double>{"can"} /* 3 */,
                                       _nrn_mechanism_field<double>{"gbar"} /* 4 */,
                                       _nrn_mechanism_field<double>{"taumin"} /* 5 */,
                                       _nrn_mechanism_field<double>{"concrelease"} /* 6 */,
                                       _nrn_mechanism_field<double>{"itrpm4"} /* 7 */,
                                       _nrn_mechanism_field<double>{"Po"} /* 8 */,
                                       _nrn_mechanism_field<double>{"DPo"} /* 9 */,
                                       _nrn_mechanism_field<double>{"jip3p"} /* 10 */,
                                       _nrn_mechanism_field<double>{"ican"} /* 11 */,
                                       _nrn_mechanism_field<double>{"drive_channel"} /* 12 */,
                                       _nrn_mechanism_field<double>{"Po_inf"} /* 13 */,
                                       _nrn_mechanism_field<double>{"Tau"} /* 14 */,
                                       _nrn_mechanism_field<double>{"v"} /* 15 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 16 */,
                                       _nrn_mechanism_field<double*>{"diam", "diam"} /* 0 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 1 */);
  hoc_register_prop_size(_mechtype, 17, 2);
  hoc_register_dparam_semantics(_mechtype, 1, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 0, "diam");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 icand /Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/icand.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 0x1.78e555060882cp+16;
 static double PI = 0x1.921fb54442d18p+1;
static int _reset;
static const char *modelname = "Slow Ca-dependent cation current";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct(_internalthreadargsprotocomma_ double, double);
 
#define _deriv1_advance _thread[0].literal_value<int>()
#define _dith1 1
#define _recurse _thread[2].literal_value<int>()
#define _newtonspace1 _thread[3].literal_value<NewtonSpace*>()
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 static neuron::container::field_index _slist2[1];
  static neuron::container::field_index _slist1[1], _dlist1[1];
 static int states(_internalthreadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (_internalthreadargsproto_) {int _reset = 0; {
   evaluate_fct ( _threadargscomma_ v , can ) ;
   DPo = ( Po_inf - Po ) / ( Tau ) ;
   }
 return _reset;
}
 static int _ode_matsol1 (_internalthreadargsproto_) {
 evaluate_fct ( _threadargscomma_ v , can ) ;
 DPo = DPo  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ( Tau ) )) ;
  return 0;
}
 /*END CVODE*/
 
static int states (_internalthreadargsproto_) {
  int _reset=0;
  int error = 0;
 {
  auto* _savstate1 =_thread[_dith1].get<double*>();
  auto* _dlist2 = _thread[_dith1].get<double*>() + 1;
  int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 for(int _id=0; _id < 1; _id++) {
  _savstate1[_id] = _ml->data(_iml, _slist1[_id]);
}
 error = nrn_newton_thread(_newtonspace1, 1, _slist2, neuron::scopmath::row_view{_ml, _iml}, states, _dlist2, _ml, _iml, _ppvar, _thread, _globals, _nt);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   evaluate_fct ( _threadargscomma_ v , can ) ;
   DPo = ( Po_inf - Po ) / ( Tau ) ;
   {int _id; for(_id=0; _id < 1; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _ml->data(_iml, _dlist1[_id]) - (_ml->data(_iml, _slist1[_id]) - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _ml->data(_iml, _slist1[_id]) - _savstate1[_id];}}}
 } }
 return _reset;}
 
double MyExp ( _internalthreadargsprotocomma_ double _lx ) {
   double _lMyExp;
 if ( _lx < - 50.0 ) {
     _lMyExp = 0.0 ;
     }
   else if ( _lx > 50.0 ) {
     _lMyExp = exp ( 50.0 ) ;
     }
   else {
     _lMyExp = exp ( _lx ) ;
     }
   
return _lMyExp;
 }
 
static void _hoc_MyExp(void) {
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
 _r =  MyExp ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_MyExp(Prop* _prop) {
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
 _r =  MyExp ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
static int  evaluate_fct ( _internalthreadargsprotocomma_ double _lv , double _lcai ) {
   double _lalpha , _lalpha2 , _lbeta ;
 _lalpha = 0.0057 * MyExp ( _threadargscomma_ 0.0060 * - 60.0 ) ;
   _lbeta = 0.033 * MyExp ( _threadargscomma_ - 0.019 * - 60.0 ) ;
   _lalpha2 = _lalpha / ( 1.0 + ( Kd / _lcai ) ) ;
   Po_inf = _lalpha2 / ( _lalpha2 + _lbeta ) ;
   Tau = 1.0 / ( _lalpha2 + _lbeta ) ;
   if ( Tau < taumin ) {
     Tau = taumin ;
     }
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  if(!_prop_id) {
    hoc_execerror("No data for evaluate_fct_icand. Requires prior call to setdata_icand and that the specified mechanism instance still be in existence.", NULL);
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
 evaluate_fct ( _threadargscomma_ *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static double _npy_evaluate_fct(Prop* _prop) {
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
 evaluate_fct ( _threadargscomma_ *getarg(1) , *getarg(2) );
 return(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
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
     _ode_spec1 (_threadargs_);
 }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  Datum* _ppvar;
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 1; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _ode_matsol1 (_threadargs_);
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
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[_dith1] = new double[2]{};
   _newtonspace1 = nrn_cons_newtonspace(1);
 }
 
static void _thread_cleanup(Datum* _thread) {
   delete[] _thread[_dith1].get<double*>();
   nrn_destroy_newtonspace(_newtonspace1);
 }

static void initmodel(_internalthreadargsproto_) {
  int _i; double _save;{
  Po = Po0;
 {
   can = cai ;
   evaluate_fct ( _threadargscomma_ v , can ) ;
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
 initmodel(_threadargs_);
}
}

static double _nrn_current(_internalthreadargsprotocomma_ double _v) {
double _current=0.; v=_v;
{ {
   itrpm4 = gbar * Po * ( v - erev ) ;
   }
 _current += itrpm4;

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
 auto const _g_local = _nrn_current(_threadargscomma_ _v + .001);
 	{ _rhs = _nrn_current(_threadargscomma_ _v);
 	}
 _g = (_g_local - _rhs)/.001;
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
 {  _deriv1_advance = 1;
 derivimplicit_thread(1, _slist1, _dlist1, neuron::scopmath::row_view{_ml, _iml}, states, _ml, _iml, _ppvar, _thread, _globals, _nt);
_deriv1_advance = 0;
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 1; ++_i) {
      _ml->data(_iml, _slist1[_i]) += dt*_ml->data(_iml, _dlist1[_i]);
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = {Po_columnindex, 0};  _dlist1[0] = {DPo_columnindex, 0};
 _slist2[0] = {Po_columnindex, 0};
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/icand.mod";
    const char* nmodl_file_text = 
  "TITLE Slow Ca-dependent cation current\n"
  ":\n"
  ":   We've moved to the Model described by Nillus in 2004, while keeping the description of the nanodomain\n"
  ": modified by Canavier too include separate pool for ICan calcium microdomain\n"
  ": at this point ican doesn't activate other pools of calcium need to declare a new ion species\n"
  ": because at this point the code requires pools for SK+BK+T inactivation and ICAN that decay at different rates\n"
  "\n"
  "INDEPENDENT {\n"
  "    t FROM 0 TO 1 WITH 1 (ms)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX icand\n"
  "    :USEION ca READ cai, ica \n"
  "    RANGE depth,taur,erev, can\n"
  "    RANGE gbar,itrpm4, concrelease\n"
  "    RANGE beta,taumin\n"
  "    NONSPECIFIC_CURRENT itrpm4\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (mA)=(milliamp)\n"
  "    (mV)=(millivolt)\n"
  "    (molar)=(1/liter)\n"
  "    (mM)=(millimolar)\n"
  "    (um)=(micron)\n"
  "    (msM)=(ms mM)\n"
  "    FARADAY=(faraday) (coulomb)\n"
  "	PI      = (pi)       (1)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    v (mV)\n"
  "    depth=0.0125 (um)         : depth of shell 0.0125 for pc1a\n"
  "    taur=80 (ms)           : rate of calcium removal	100\n"
  "    erev=0 (mV)             : reversal potential\n"
  "    cai (mM)                : will now decay to bulk cai\n"
  "    can (mM) \n"
  "	ica       (mA/cm2)\n"
  "    gbar=0.0001 (mho/cm2)\n"
  "    : middle point of activation fct, for ip3 as somacar, for current injection\n"
  "    taumin=0.1 (ms)         : minimal value of time constant\n"
  "    concrelease=500\n"
  "    Kd = 87e-3 (mM)		:87e-3\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    Po\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	jip3p (mM/ms)\n"
  "    ican (mA/cm2)\n"
  "    drive_channel (mM/ms)\n"
  "    itrpm4 (mA/cm2)\n"
  "    Po_inf\n"
  "    Tau (ms)\n"
  "	diam      (um)\n"
  "    :cai (mM) \n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD derivimplicit\n"
  "    itrpm4=gbar*Po*(v-erev)\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    evaluate_fct(v,can)\n"
  "    Po'=(Po_inf-Po)/(Tau)\n"
  "}\n"
  "\n"
  "FUNCTION MyExp(x) {\n"
  "    if (x<-50) {MyExp=0}\n"
  "    else if (x>50) {MyExp=exp(50)}\n"
  "    else {MyExp=exp(x)}\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "INITIAL {\n"
  "    : activation kinetics are assumed to be at 22 deg. C\n"
  "    : Q10 is assumed to be 3\n"
  "    can=cai\n"
  "    evaluate_fct(v,can)\n"
  "}\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV),cai(mM)) {\n"
  "    LOCAL alpha, alpha2, beta\n"
  "    :alpha=0.0057*MyExp(0.0060*v)\n"
  "    :beta=0.033*MyExp(-0.019*v)\n"
  "    \n"
  "    alpha=0.0057*MyExp(0.0060*-60)\n"
  "    beta=0.033*MyExp(-0.019*-60)\n"
  "\n"
  "    alpha2=alpha/(1+(Kd/cai))\n"
  "    Po_inf=alpha2/(alpha2+beta)\n"
  "    Tau=1/(alpha2+beta)\n"
  "    if (Tau<taumin) {Tau=taumin}                        : min value of time cst\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
