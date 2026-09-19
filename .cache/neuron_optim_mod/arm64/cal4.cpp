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
 static void _difusfunc(ldifusfunc2_t, neuron::model_sorted_token const&, NrnThread&);
static constexpr auto number_of_datum_variables = 7;
static constexpr auto number_of_floating_point_variables = 47;
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
 
#define nrn_init _nrn_init__cal4
#define _nrn_initial _nrn_initial__cal4
#define nrn_cur _nrn_cur__cal4
#define _nrn_current _nrn_current__cal4
#define nrn_jacob _nrn_jacob__cal4
#define nrn_state _nrn_state__cal4
#define _net_receive _net_receive__cal4 
#define factors factors__cal4 
#define state state__cal4 
 
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
#define gamma _ml->template fpfield<0>(_iml)
#define gamma_columnindex 0
#define sites _ml->template fpfield<1>(_iml)
#define sites_columnindex 1
#define alpha _ml->template fpfield<2>(_iml)
#define alpha_columnindex 2
#define beta _ml->template fpfield<3>(_iml)
#define beta_columnindex 3
#define DCa _ml->template fpfield<4>(_iml)
#define DCa_columnindex 4
#define cip3 _ml->template fpfield<5>(_iml)
#define cip3_columnindex 5
#define jip3 _ml->template fpfield<6>(_iml)
#define jip3_columnindex 6
#define ca1 _ml->template fpfield<7>(_iml)
#define ca1_columnindex 7
#define ca2 _ml->template fpfield<8>(_iml)
#define ca2_columnindex 8
#define ca3 _ml->template fpfield<9>(_iml)
#define ca3_columnindex 9
#define ica_pmp _ml->template fpfield<10>(_iml)
#define ica_pmp_columnindex 10
#define ip3ca _ml->template fpfield<11>(_iml)
#define ip3ca_columnindex 11
#define ca _ml->template data_array<12, 4>(_iml)
#define ca_columnindex 12
#define hc _ml->template data_array<13, 4>(_iml)
#define hc_columnindex 13
#define ho _ml->template data_array<14, 4>(_iml)
#define ho_columnindex 14
#define bufs _ml->template data_array<15, 4>(_iml)
#define bufs_columnindex 15
#define cabufs _ml->template data_array<16, 4>(_iml)
#define cabufs_columnindex 16
#define bufm _ml->template data_array<17, 4>(_iml)
#define bufm_columnindex 17
#define cabufm _ml->template data_array<18, 4>(_iml)
#define cabufm_columnindex 18
#define bapta _ml->template data_array<19, 4>(_iml)
#define bapta_columnindex 19
#define cabapta _ml->template data_array<20, 4>(_iml)
#define cabapta_columnindex 20
#define ip3cas _ml->template data_array<21, 4>(_iml)
#define ip3cas_columnindex 21
#define ica _ml->template fpfield<22>(_iml)
#define ica_columnindex 22
#define cai _ml->template fpfield<23>(_iml)
#define cai_columnindex 23
#define ica_pmp_last _ml->template fpfield<24>(_iml)
#define ica_pmp_last_columnindex 24
#define parea _ml->template fpfield<25>(_iml)
#define parea_columnindex 25
#define sump _ml->template fpfield<26>(_iml)
#define sump_columnindex 26
#define cao _ml->template fpfield<27>(_iml)
#define cao_columnindex 27
#define jchnl _ml->template fpfield<28>(_iml)
#define jchnl_columnindex 28
#define L _ml->template data_array<29, 4>(_iml)
#define L_columnindex 29
#define adjusted _ml->template fpfield<30>(_iml)
#define adjusted_columnindex 30
#define so _ml->template fpfield<31>(_iml)
#define so_columnindex 31
#define that _ml->template fpfield<32>(_iml)
#define that_columnindex 32
#define bufs_0 _ml->template fpfield<33>(_iml)
#define bufs_0_columnindex 33
#define bufm_0 _ml->template fpfield<34>(_iml)
#define bufm_0_columnindex 34
#define bapta_0 _ml->template fpfield<35>(_iml)
#define bapta_0_columnindex 35
#define Dca _ml->template data_array<36, 4>(_iml)
#define Dca_columnindex 36
#define Dhc _ml->template data_array<37, 4>(_iml)
#define Dhc_columnindex 37
#define Dho _ml->template data_array<38, 4>(_iml)
#define Dho_columnindex 38
#define Dbufs _ml->template data_array<39, 4>(_iml)
#define Dbufs_columnindex 39
#define Dcabufs _ml->template data_array<40, 4>(_iml)
#define Dcabufs_columnindex 40
#define Dbufm _ml->template data_array<41, 4>(_iml)
#define Dbufm_columnindex 41
#define Dcabufm _ml->template data_array<42, 4>(_iml)
#define Dcabufm_columnindex 42
#define Dbapta _ml->template data_array<43, 4>(_iml)
#define Dbapta_columnindex 43
#define Dcabapta _ml->template data_array<44, 4>(_iml)
#define Dcabapta_columnindex 44
#define Dip3cas _ml->template data_array<45, 4>(_iml)
#define Dip3cas_columnindex 45
#define _g _ml->template fpfield<46>(_iml)
#define _g_columnindex 46
#define _ion_cao *(_ml->dptr_field<0>(_iml))
#define _p_ion_cao static_cast<neuron::container::data_handle<double>>(_ppvar[0])
#define _ion_ica *(_ml->dptr_field<1>(_iml))
#define _p_ion_ica static_cast<neuron::container::data_handle<double>>(_ppvar[1])
#define _ion_cai *(_ml->dptr_field<2>(_iml))
#define _p_ion_cai static_cast<neuron::container::data_handle<double>>(_ppvar[2])
#define _ion_dicadv *(_ml->dptr_field<3>(_iml))
#define _ion_ca_erev *_ml->dptr_field<4>(_iml)
#define _style_ca	*_ppvar[5].get<int*>()
#define diam	(*(_ml->dptr_field<6>(_iml)))
 static _nrn_mechanism_cache_instance _ml_real{nullptr};
static _nrn_mechanism_cache_range *_ml{&_ml_real};
static size_t _iml{0};
static Datum *_ppvar;
 static int hoc_nrnpointerindex =  -1;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_factors(void);
 static void _hoc_u(void);
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
 {"setdata_cal4", _hoc_setdata},
 {"factors_cal4", _hoc_factors},
 {"u_cal4", _hoc_u},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_factors(Prop*);
 static double _npy_u(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"factors", _npy_factors},
 {"u", _npy_u},
 {0, 0}
};
#define u u_cal4
 extern double u( double , double );
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
#define DBufm DBufm_cal4
 double DBufm = 0.05;
#define Kp Kp_cal4
 double Kp = 0.00027;
#define Kinh Kinh_cal4
 double Kinh = 0.0006;
#define Kact Kact_cal4
 double Kact = 0.0007;
#define Kip3 Kip3_cal4
 double Kip3 = 0.0008;
#define KDBAPTA KDBAPTA_cal4
 double KDBAPTA = 0.2;
#define KDm KDm_cal4
 double KDm = 0.24;
#define KDs KDs_cal4
 double KDs = 10;
#define TBufBAPTA TBufBAPTA_cal4
 double TBufBAPTA = 0;
#define TBufm TBufm_cal4
 double TBufm = 0.075;
#define TBufs TBufs_cal4
 double TBufs = 0.45;
#define caer caer_cal4
 double caer = 0.4;
#define cath cath_cal4
 double cath = 0.0002;
#define cai0 cai0_cal4
 double cai0 = 5e-05;
#define ip3i ip3i_cal4
 double ip3i = 0.01;
#define jmax jmax_cal4
 double jmax = 0.0035;
#define kfm kfm_cal4
 double kfm = 1000;
#define kfBAPTA kfBAPTA_cal4
 double kfBAPTA = 500;
#define kfs kfs_cal4
 double kfs = 1000;
#define kon kon_cal4
 double kon = 2.7;
#define vmax vmax_cal4
 double vmax = 0.0001;
#define vrat vrat_cal4
 double vrat[4];
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"ip3i_cal4", "mM"},
 {"cai0_cal4", "mM"},
 {"cath_cal4", "mM"},
 {"jmax_cal4", "mM/ms"},
 {"caer_cal4", "mM"},
 {"Kip3_cal4", "mM"},
 {"Kact_cal4", "mM"},
 {"kon_cal4", "/mM-ms"},
 {"Kinh_cal4", "mM"},
 {"vmax_cal4", "mM/ms"},
 {"Kp_cal4", "mM"},
 {"TBufs_cal4", "mM"},
 {"kfs_cal4", "/mM-ms"},
 {"KDs_cal4", "uM"},
 {"TBufBAPTA_cal4", "mM"},
 {"kfBAPTA_cal4", "/mM-ms"},
 {"KDBAPTA_cal4", "uM"},
 {"TBufm_cal4", "mM"},
 {"kfm_cal4", "/mM-ms"},
 {"KDm_cal4", "uM"},
 {"DBufm_cal4", "um2/ms"},
 {"vrat_cal4", "1"},
 {"gamma_cal4", "um/s"},
 {"alpha_cal4", "1"},
 {"beta_cal4", "1"},
 {"DCa_cal4", "um2/ms"},
 {"ca_cal4", "mM"},
 {"bufs_cal4", "mM"},
 {"cabufs_cal4", "mM"},
 {"bufm_cal4", "mM"},
 {"cabufm_cal4", "mM"},
 {"bapta_cal4", "mM"},
 {"cabapta_cal4", "mM"},
 {"ip3cas_cal4", "mM"},
 {"cip3_cal4", "mA/cm2"},
 {"jip3_cal4", "mM/ms"},
 {"ca1_cal4", "mM"},
 {"ca2_cal4", "mM"},
 {"ca3_cal4", "mM"},
 {"ica_pmp_cal4", "mA/cm2"},
 {"ip3ca_cal4", "mM"},
 {0, 0}
};
 static double bapta0 = 0;
 static double bufm0 = 0;
 static double bufs0 = 0;
 static double cabapta0 = 0;
 static double cabufm0 = 0;
 static double cabufs0 = 0;
 static double ca0 = 0;
 static double delta_t = 0.01;
 static double ho0 = 0;
 static double hc0 = 0;
 static double ip3cas0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"ip3i_cal4", &ip3i_cal4},
 {"cai0_cal4", &cai0_cal4},
 {"cath_cal4", &cath_cal4},
 {"jmax_cal4", &jmax_cal4},
 {"caer_cal4", &caer_cal4},
 {"Kip3_cal4", &Kip3_cal4},
 {"Kact_cal4", &Kact_cal4},
 {"kon_cal4", &kon_cal4},
 {"Kinh_cal4", &Kinh_cal4},
 {"vmax_cal4", &vmax_cal4},
 {"Kp_cal4", &Kp_cal4},
 {"TBufs_cal4", &TBufs_cal4},
 {"kfs_cal4", &kfs_cal4},
 {"KDs_cal4", &KDs_cal4},
 {"TBufBAPTA_cal4", &TBufBAPTA_cal4},
 {"kfBAPTA_cal4", &kfBAPTA_cal4},
 {"KDBAPTA_cal4", &KDBAPTA_cal4},
 {"TBufm_cal4", &TBufm_cal4},
 {"kfm_cal4", &kfm_cal4},
 {"KDm_cal4", &KDm_cal4},
 {"DBufm_cal4", &DBufm_cal4},
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {"vrat_cal4", vrat_cal4, 4},
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
 
#define _cvode_ieq _ppvar[7].literal_value<int>()
 static void _ode_synonym(_nrn_model_sorted_token const&, NrnThread&, Memb_list&, int);
 static void _ode_matsol_instance1(_internalthreadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"cal4",
 "gamma_cal4",
 "sites_cal4",
 "alpha_cal4",
 "beta_cal4",
 "DCa_cal4",
 0,
 "cip3_cal4",
 "jip3_cal4",
 "ca1_cal4",
 "ca2_cal4",
 "ca3_cal4",
 "ica_pmp_cal4",
 "ip3ca_cal4",
 0,
 "ca_cal4[4]",
 "hc_cal4[4]",
 "ho_cal4[4]",
 "bufs_cal4[4]",
 "cabufs_cal4[4]",
 "bufm_cal4[4]",
 "cabufm_cal4[4]",
 "bapta_cal4[4]",
 "cabapta_cal4[4]",
 "ip3cas_cal4[4]",
 0,
 0};
 static Symbol* _morphology_sym;
 static Symbol* _ca_sym;
 static int _type_ica;
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     8, /* gamma */
     3, /* sites */
     1, /* alpha */
     1, /* beta */
     0.22, /* DCa */
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
   _ppvar = nrn_prop_datum_alloc(_mechtype, 8, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 47);
 	/*initialize range parameters*/
 	gamma = _parm_default[0]; /* 8 */
 	sites = _parm_default[1]; /* 3 */
 	alpha = _parm_default[2]; /* 1 */
 	beta = _parm_default[3]; /* 1 */
 	DCa = _parm_default[4]; /* 0.22 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 47);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_morphology_sym);
 	_ppvar[6] = _nrn_mechanism_get_param_handle(prop_ion, 0); /* diam */
 prop_ion = need_memb(_ca_sym);
  _type_ica = _nrn_mechanism_get_type(prop_ion);
 nrn_check_conc_write(_prop, prop_ion, 1);
 nrn_promote(prop_ion, 3, 0);
 	_ppvar[0] = _nrn_mechanism_get_param_handle(prop_ion, 2); /* cao */
 	_ppvar[1] = _nrn_mechanism_get_param_handle(prop_ion, 3); /* ica */
 	_ppvar[2] = _nrn_mechanism_get_param_handle(prop_ion, 1); /* cai */
 	_ppvar[3] = _nrn_mechanism_get_param_handle(prop_ion, 4); /* _ion_dicadv */
 	_ppvar[4] = _nrn_mechanism_get_param_handle(prop_ion, 0); // erev ca
 	_ppvar[5] = {neuron::container::do_not_search, &(_nrn_mechanism_access_dparam(prop_ion)[0].literal_value<int>())}; /* iontype for ca */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {"ca_cal4", 1e-07},
 {"bufs_cal4", 0.001},
 {"cabufs_cal4", 1e-07},
 {"bufm_cal4", 0.0001},
 {"cabufm_cal4", 1e-08},
 {"bapta_cal4", 0.001},
 {"cabapta_cal4", 1e-07},
 {0, 0}
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _cal4_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("ca", -10000.);
 	_morphology_sym = hoc_lookup("morphology");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"gamma"} /* 0 */,
                                       _nrn_mechanism_field<double>{"sites"} /* 1 */,
                                       _nrn_mechanism_field<double>{"alpha"} /* 2 */,
                                       _nrn_mechanism_field<double>{"beta"} /* 3 */,
                                       _nrn_mechanism_field<double>{"DCa"} /* 4 */,
                                       _nrn_mechanism_field<double>{"cip3"} /* 5 */,
                                       _nrn_mechanism_field<double>{"jip3"} /* 6 */,
                                       _nrn_mechanism_field<double>{"ca1"} /* 7 */,
                                       _nrn_mechanism_field<double>{"ca2"} /* 8 */,
                                       _nrn_mechanism_field<double>{"ca3"} /* 9 */,
                                       _nrn_mechanism_field<double>{"ica_pmp"} /* 10 */,
                                       _nrn_mechanism_field<double>{"ip3ca"} /* 11 */,
                                       _nrn_mechanism_field<double>{"ca", 4} /* 12 */,
                                       _nrn_mechanism_field<double>{"hc", 4} /* 13 */,
                                       _nrn_mechanism_field<double>{"ho", 4} /* 14 */,
                                       _nrn_mechanism_field<double>{"bufs", 4} /* 15 */,
                                       _nrn_mechanism_field<double>{"cabufs", 4} /* 16 */,
                                       _nrn_mechanism_field<double>{"bufm", 4} /* 17 */,
                                       _nrn_mechanism_field<double>{"cabufm", 4} /* 18 */,
                                       _nrn_mechanism_field<double>{"bapta", 4} /* 19 */,
                                       _nrn_mechanism_field<double>{"cabapta", 4} /* 20 */,
                                       _nrn_mechanism_field<double>{"ip3cas", 4} /* 21 */,
                                       _nrn_mechanism_field<double>{"ica"} /* 22 */,
                                       _nrn_mechanism_field<double>{"cai"} /* 23 */,
                                       _nrn_mechanism_field<double>{"ica_pmp_last"} /* 24 */,
                                       _nrn_mechanism_field<double>{"parea"} /* 25 */,
                                       _nrn_mechanism_field<double>{"sump"} /* 26 */,
                                       _nrn_mechanism_field<double>{"cao"} /* 27 */,
                                       _nrn_mechanism_field<double>{"jchnl"} /* 28 */,
                                       _nrn_mechanism_field<double>{"L", 4} /* 29 */,
                                       _nrn_mechanism_field<double>{"adjusted"} /* 30 */,
                                       _nrn_mechanism_field<double>{"so"} /* 31 */,
                                       _nrn_mechanism_field<double>{"that"} /* 32 */,
                                       _nrn_mechanism_field<double>{"bufs_0"} /* 33 */,
                                       _nrn_mechanism_field<double>{"bufm_0"} /* 34 */,
                                       _nrn_mechanism_field<double>{"bapta_0"} /* 35 */,
                                       _nrn_mechanism_field<double>{"Dca", 4} /* 36 */,
                                       _nrn_mechanism_field<double>{"Dhc", 4} /* 37 */,
                                       _nrn_mechanism_field<double>{"Dho", 4} /* 38 */,
                                       _nrn_mechanism_field<double>{"Dbufs", 4} /* 39 */,
                                       _nrn_mechanism_field<double>{"Dcabufs", 4} /* 40 */,
                                       _nrn_mechanism_field<double>{"Dbufm", 4} /* 41 */,
                                       _nrn_mechanism_field<double>{"Dcabufm", 4} /* 42 */,
                                       _nrn_mechanism_field<double>{"Dbapta", 4} /* 43 */,
                                       _nrn_mechanism_field<double>{"Dcabapta", 4} /* 44 */,
                                       _nrn_mechanism_field<double>{"Dip3cas", 4} /* 45 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 46 */,
                                       _nrn_mechanism_field<double*>{"_ion_cao", "ca_ion"} /* 0 */,
                                       _nrn_mechanism_field<double*>{"_ion_ica", "ca_ion"} /* 1 */,
                                       _nrn_mechanism_field<double*>{"_ion_cai", "ca_ion"} /* 2 */,
                                       _nrn_mechanism_field<double*>{"_ion_dicadv", "ca_ion"} /* 3 */,
                                       _nrn_mechanism_field<double*>{"_ion_ca_erev", "ca_ion"} /* 4 */,
                                       _nrn_mechanism_field<int*>{"_style_ca", "#ca_ion"} /* 5 */,
                                       _nrn_mechanism_field<double*>{"diam", "diam"} /* 6 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 7 */);
  hoc_register_prop_size(_mechtype, 110, 8);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "#ca_ion");
  hoc_register_dparam_semantics(_mechtype, 7, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 6, "diam");
 	nrn_writes_conc(_mechtype, 0);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_synonym(_mechtype, _ode_synonym);
 	hoc_register_ldifus1(_difusfunc);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cal4 /Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/cal4.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 0x1.34c0c8b92a9b7p+3;
 static double PI = 0x1.921fb54442d18p+1;
 static double volo = 1e10;
 static double _zfactors_done , _zjx ;
 static double _zfrat [ 4 ] ;
 static double _zdsq , _zdsqvol ;
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int factors();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  0
 static void* _sparseobj1;
 static void* _cvsparseobj1;
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 static neuron::container::field_index _slist1[32], _dlist1[32]; static double *_temp1;
 static int state ();
 
static int  factors (  ) {
   double _lr , _ldr2 ;
 _lr = 1.0 / 2.0 ;
   _ldr2 = _lr / ( 4.0 - 1.0 ) / 2.0 ;
   vrat [ 0 ] = 0.0 ;
   _zfrat [ 0 ] = 2.0 * _lr ;
   {int  _li ;for ( _li = 0 ; _li <= 4 - 2 ; _li ++ ) {
     vrat [ _li ] = vrat [ _li ] + PI * ( _lr - _ldr2 / 2.0 ) * 2.0 * _ldr2 ;
     _lr = _lr - _ldr2 ;
     _zfrat [ _li + 1 ] = 2.0 * PI * _lr / ( 2.0 * _ldr2 ) ;
     _lr = _lr - _ldr2 ;
     vrat [ _li + 1 ] = PI * ( _lr + _ldr2 / 2.0 ) * 2.0 * _ldr2 ;
     } }
    return 0; }
 
static void _hoc_factors(void) {
  double _r;
    _r = 1.;
 factors (  );
 hoc_retpushx(_r);
}
 
static double _npy_factors(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r = 1.;
 factors (  );
 return(_r);
}
 
static int state ()
 {_reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<32;_i++){
  	_RHS1(_i) = -_dt1*(_ml->data(_iml, _slist1[_i]) - _ml->data(_iml, _dlist1[_i]));
	_MATELM1(_i, _i) = _dt1;
      
} 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 4) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 4, _i + 4) *= ( diam * diam * vrat [ ((int) _i ) ]);  } 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 12) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 12, _i + 12) *= ( diam * diam * vrat [ ((int) _i ) ]);  } 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 16) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 16, _i + 16) *= ( diam * diam * vrat [ ((int) _i ) ]);  } }
 /* COMPARTMENT _li , diam * diam * vrat [ ((int) _i ) ] {
     ca bufs cabufs bufm cabufm }
   */
 /* COMPARTMENT volo {
     }
   */
 /* LONGITUDINAL_DIFFUSION _li , DCa * diam * diam * vrat [ ((int) _i ) ] {
     ca }
   */
 /* LONGITUDINAL_DIFFUSION _li , DBufm * diam * diam * vrat [ ((int) _i ) ] {
     bufm cabufm }
   */
 /* ~ ca [ 0 ] <-> sump ( ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) , ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) )*/
 f_flux =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) * ca [ 0] ;
 b_flux =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) * sump ;
 _RHS1( 16 +  0) -= (f_flux - b_flux);
 
 _term =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) ;
 _MATELM1( 16 +  0 ,16 +  0)  += _term;
 /*REACTION*/
  ica_pmp = 2.0 * FARADAY * ( f_flux - b_flux ) / parea ;
   /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 16 +  0) += (b_flux =   ( - ( ica - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 4 - 2 ; _li ++ ) {
     /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DCa * _zfrat [ _li + 1 ] , DCa * _zfrat [ _li + 1 ] )*/
 f_flux =  DCa * _zfrat [ _li + 1 ] * ca [ _li] ;
 b_flux =  DCa * _zfrat [ _li + 1 ] * ca [ _li + 1] ;
 _RHS1( 16 +  _li) -= (f_flux - b_flux);
 _RHS1( 16 +  _li + 1) += (f_flux - b_flux);
 
 _term =  DCa * _zfrat [ _li + 1 ] ;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li + 1 ,16 +  _li)  -= _term;
 _term =  DCa * _zfrat [ _li + 1 ] ;
 _MATELM1( 16 +  _li ,16 +  _li + 1)  -= _term;
 _MATELM1( 16 +  _li + 1 ,16 +  _li + 1)  += _term;
 /*REACTION*/
  } }
   _zdsq = diam * diam ;
   {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
     _zdsqvol = _zdsq * vrat [ _li ] ;
     /* ~ ca [ _li ] + bufs [ _li ] <-> cabufs [ _li ] ( kfs * _zdsqvol , ( 0.001 ) * KDs * kfs * _zdsqvol )*/
 f_flux =  kfs * _zdsqvol * bufs [ _li] * ca [ _li] ;
 b_flux =  ( 0.001 ) * KDs * kfs * _zdsqvol * cabufs [ _li] ;
 _RHS1( 4 +  _li) -= (f_flux - b_flux);
 _RHS1( 16 +  _li) -= (f_flux - b_flux);
 _RHS1( 12 +  _li) += (f_flux - b_flux);
 
 _term =  kfs * _zdsqvol * ca [ _li] ;
 _MATELM1( 4 +  _li ,4 +  _li)  += _term;
 _MATELM1( 16 +  _li ,4 +  _li)  += _term;
 _MATELM1( 12 +  _li ,4 +  _li)  -= _term;
 _term =  kfs * _zdsqvol * bufs [ _li] ;
 _MATELM1( 4 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 12 +  _li ,16 +  _li)  -= _term;
 _term =  ( 0.001 ) * KDs * kfs * _zdsqvol ;
 _MATELM1( 4 +  _li ,12 +  _li)  -= _term;
 _MATELM1( 16 +  _li ,12 +  _li)  -= _term;
 _MATELM1( 12 +  _li ,12 +  _li)  += _term;
 /*REACTION*/
  /* ~ ca [ _li ] + bapta [ _li ] <-> cabapta [ _li ] ( kfBAPTA * _zdsqvol , ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol )*/
 f_flux =  kfBAPTA * _zdsqvol * bapta [ _li] * ca [ _li] ;
 b_flux =  ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol * cabapta [ _li] ;
 _RHS1( 0 +  _li) -= (f_flux - b_flux);
 _RHS1( 16 +  _li) -= (f_flux - b_flux);
 _RHS1( 8 +  _li) += (f_flux - b_flux);
 
 _term =  kfBAPTA * _zdsqvol * ca [ _li] ;
 _MATELM1( 0 +  _li ,0 +  _li)  += _term;
 _MATELM1( 16 +  _li ,0 +  _li)  += _term;
 _MATELM1( 8 +  _li ,0 +  _li)  -= _term;
 _term =  kfBAPTA * _zdsqvol * bapta [ _li] ;
 _MATELM1( 0 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 8 +  _li ,16 +  _li)  -= _term;
 _term =  ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol ;
 _MATELM1( 0 +  _li ,8 +  _li)  -= _term;
 _MATELM1( 16 +  _li ,8 +  _li)  -= _term;
 _MATELM1( 8 +  _li ,8 +  _li)  += _term;
 /*REACTION*/
  } }
   {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
     _zdsqvol = _zdsq * vrat [ _li ] ;
     /* ~ ca [ _li ] < < ( - _zdsqvol * beta * vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 16 +  _li) += (b_flux =   ( - _zdsqvol * beta * vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) ) );
 /*FLUX*/
  /* ~ hc [ _li ] <-> ho [ _li ] ( kon * Kinh , kon * ca [ _li ] )*/
 f_flux =  kon * Kinh * hc [ _li] ;
 b_flux =  kon * ca [ _li ] * ho [ _li] ;
 _RHS1( 24 +  _li) -= (f_flux - b_flux);
 _RHS1( 20 +  _li) += (f_flux - b_flux);
 
 _term =  kon * Kinh ;
 _MATELM1( 24 +  _li ,24 +  _li)  += _term;
 _MATELM1( 20 +  _li ,24 +  _li)  -= _term;
 _term =  kon * ca [ _li ] ;
 _MATELM1( 24 +  _li ,20 +  _li)  -= _term;
 _MATELM1( 20 +  _li ,20 +  _li)  += _term;
 /*REACTION*/
  /* ~ ca [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 16 +  _li) += (b_flux =   ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) ) );
 /*FLUX*/
  /* ~ ca [ _li ] < < ( _zdsqvol * beta * L [ _li ] * ( 1.0 - ( ca [ _li ] / caer ) ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 16 +  _li) += (b_flux =   ( _zdsqvol * beta * L [ _li ] * ( 1.0 - ( ca [ _li ] / caer ) ) ) );
 /*FLUX*/
  /* ~ ip3cas [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 28 +  _li) += (b_flux =   ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) ) );
 /*FLUX*/
  } }
   jip3 = ( jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) ;
   cip3 = ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) * ( 2.0 * FARADAY ) / ( PI * ( diam ) ) ;
   ip3ca = ip3cas [ 0 ] ;
   cai = ca [ 0 ] ;
   ca1 = ca [ 1 ] ;
   ca2 = ca [ 2 ] ;
   ca3 = ca [ 3 ] ;
     } return _reset;
 }
 
double u (  double _lx , double _lth ) {
   double _lu;
 if ( _lx > _lth ) {
     _lu = 1.0 ;
     }
   else {
     _lu = 0.0 ;
     }
   
return _lu;
 }
 
static void _hoc_u(void) {
  double _r;
    _r =  u (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static double _npy_u(Prop* _prop) {
    double _r{0.0};
    neuron::legacy::set_globals_from_prop(_prop, _ml_real, _ml, _iml);
  _ppvar = _nrn_mechanism_access_dparam(_prop);
 _r =  u (  *getarg(1) , *getarg(2) );
 return(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<32;_i++) _ml->data(_iml, _dlist1[_i]) = 0.0;}
 /* COMPARTMENT _li , diam * diam * vrat [ ((int) _i ) ] {
   ca bufs cabufs bufm cabufm }
 */
 /* COMPARTMENT volo {
   }
 */
 /* LONGITUDINAL_DIFFUSION _li , DCa * diam * diam * vrat [ ((int) _i ) ] {
   ca }
 */
 /* LONGITUDINAL_DIFFUSION _li , DBufm * diam * diam * vrat [ ((int) _i ) ] {
   bufm cabufm }
 */
 /* ~ ca [ 0 ] <-> sump ( ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) , ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) )*/
 f_flux =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) * ca [ 0] ;
 b_flux =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) * sump ;
 Dca [ 0] -= (f_flux - b_flux);
 
 /*REACTION*/
  ica_pmp = 2.0 * FARADAY * ( f_flux - b_flux ) / parea ;
 /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 Dca [ 0] += (b_flux =   ( - ( ica - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 4 - 2 ; _li ++ ) {
   /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DCa * _zfrat [ _li + 1 ] , DCa * _zfrat [ _li + 1 ] )*/
 f_flux =  DCa * _zfrat [ _li + 1 ] * ca [ _li] ;
 b_flux =  DCa * _zfrat [ _li + 1 ] * ca [ _li + 1] ;
 Dca [ _li] -= (f_flux - b_flux);
 Dca [ _li + 1] += (f_flux - b_flux);
 
 /*REACTION*/
  } }
 _zdsq = diam * diam ;
 {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
   _zdsqvol = _zdsq * vrat [ _li ] ;
   /* ~ ca [ _li ] + bufs [ _li ] <-> cabufs [ _li ] ( kfs * _zdsqvol , ( 0.001 ) * KDs * kfs * _zdsqvol )*/
 f_flux =  kfs * _zdsqvol * bufs [ _li] * ca [ _li] ;
 b_flux =  ( 0.001 ) * KDs * kfs * _zdsqvol * cabufs [ _li] ;
 Dbufs [ _li] -= (f_flux - b_flux);
 Dca [ _li] -= (f_flux - b_flux);
 Dcabufs [ _li] += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca [ _li ] + bapta [ _li ] <-> cabapta [ _li ] ( kfBAPTA * _zdsqvol , ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol )*/
 f_flux =  kfBAPTA * _zdsqvol * bapta [ _li] * ca [ _li] ;
 b_flux =  ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol * cabapta [ _li] ;
 Dbapta [ _li] -= (f_flux - b_flux);
 Dca [ _li] -= (f_flux - b_flux);
 Dcabapta [ _li] += (f_flux - b_flux);
 
 /*REACTION*/
  } }
 {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
   _zdsqvol = _zdsq * vrat [ _li ] ;
   /* ~ ca [ _li ] < < ( - _zdsqvol * beta * vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) )*/
 f_flux = b_flux = 0.;
 Dca [ _li] += (b_flux =   ( - _zdsqvol * beta * vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) ) );
 /*FLUX*/
  /* ~ hc [ _li ] <-> ho [ _li ] ( kon * Kinh , kon * ca [ _li ] )*/
 f_flux =  kon * Kinh * hc [ _li] ;
 b_flux =  kon * ca [ _li ] * ho [ _li] ;
 Dhc [ _li] -= (f_flux - b_flux);
 Dho [ _li] += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 f_flux = b_flux = 0.;
 Dca [ _li] += (b_flux =   ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) ) );
 /*FLUX*/
  /* ~ ca [ _li ] < < ( _zdsqvol * beta * L [ _li ] * ( 1.0 - ( ca [ _li ] / caer ) ) )*/
 f_flux = b_flux = 0.;
 Dca [ _li] += (b_flux =   ( _zdsqvol * beta * L [ _li ] * ( 1.0 - ( ca [ _li ] / caer ) ) ) );
 /*FLUX*/
  /* ~ ip3cas [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 f_flux = b_flux = 0.;
 Dip3cas [ _li] += (b_flux =   ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) ) );
 /*FLUX*/
  } }
 jip3 = ( jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) ;
 cip3 = ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) * ( 2.0 * FARADAY ) / ( PI * ( diam ) ) ;
 ip3ca = ip3cas [ 0 ] ;
 cai = ca [ 0 ] ;
 ca1 = ca [ 1 ] ;
 ca2 = ca [ 2 ] ;
 ca3 = ca [ 3 ] ;
 for (_i=0; _i < 4; _i++) { _ml->data(_iml, _dlist1[_i + 4]) /= ( diam * diam * vrat [ ((int) _i ) ]);}
 for (_i=0; _i < 4; _i++) { _ml->data(_iml, _dlist1[_i + 12]) /= ( diam * diam * vrat [ ((int) _i ) ]);}
 for (_i=0; _i < 4; _i++) { _ml->data(_iml, _dlist1[_i + 16]) /= ( diam * diam * vrat [ ((int) _i ) ]);}
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<32;_i++){
  	_RHS1(_i) = _dt1*(_ml->data(_iml, _dlist1[_i]));
	_MATELM1(_i, _i) = _dt1;
      
} 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 4) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 4, _i + 4) *= ( diam * diam * vrat [ ((int) _i ) ]);  } 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 12) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 12, _i + 12) *= ( diam * diam * vrat [ ((int) _i ) ]);  } 
for (_i=0; _i < 4; _i++) {
  	_RHS1(_i + 16) *= ( diam * diam * vrat [ ((int) _i ) ]) ;
_MATELM1(_i + 16, _i + 16) *= ( diam * diam * vrat [ ((int) _i ) ]);  } }
 /* COMPARTMENT _li , diam * diam * vrat [ ((int) _i ) ] {
 ca bufs cabufs bufm cabufm }
 */
 /* COMPARTMENT volo {
 }
 */
 /* LONGITUDINAL_DIFFUSION _li , DCa * diam * diam * vrat [ ((int) _i ) ] {
 ca }
 */
 /* LONGITUDINAL_DIFFUSION _li , DBufm * diam * diam * vrat [ ((int) _i ) ] {
 bufm cabufm }
 */
 /* ~ ca [ 0 ] <-> sump ( ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) , ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) )*/
 _term =  ( 0.001 ) * parea * gamma * u ( _threadargscomma_ ca [ 0 ] / ( 1.0 ) , cath / ( 1.0 ) ) ;
 _MATELM1( 16 +  0 ,16 +  0)  += _term;
 /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) )*/
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 4 - 2 ; _li ++ ) {
 /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DCa * _zfrat [ _li + 1 ] , DCa * _zfrat [ _li + 1 ] )*/
 _term =  DCa * _zfrat [ _li + 1 ] ;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li + 1 ,16 +  _li)  -= _term;
 _term =  DCa * _zfrat [ _li + 1 ] ;
 _MATELM1( 16 +  _li ,16 +  _li + 1)  -= _term;
 _MATELM1( 16 +  _li + 1 ,16 +  _li + 1)  += _term;
 /*REACTION*/
  } }
 _zdsq = diam * diam ;
 {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
 _zdsqvol = _zdsq * vrat [ _li ] ;
 /* ~ ca [ _li ] + bufs [ _li ] <-> cabufs [ _li ] ( kfs * _zdsqvol , ( 0.001 ) * KDs * kfs * _zdsqvol )*/
 _term =  kfs * _zdsqvol * ca [ _li] ;
 _MATELM1( 4 +  _li ,4 +  _li)  += _term;
 _MATELM1( 16 +  _li ,4 +  _li)  += _term;
 _MATELM1( 12 +  _li ,4 +  _li)  -= _term;
 _term =  kfs * _zdsqvol * bufs [ _li] ;
 _MATELM1( 4 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 12 +  _li ,16 +  _li)  -= _term;
 _term =  ( 0.001 ) * KDs * kfs * _zdsqvol ;
 _MATELM1( 4 +  _li ,12 +  _li)  -= _term;
 _MATELM1( 16 +  _li ,12 +  _li)  -= _term;
 _MATELM1( 12 +  _li ,12 +  _li)  += _term;
 /*REACTION*/
  /* ~ ca [ _li ] + bapta [ _li ] <-> cabapta [ _li ] ( kfBAPTA * _zdsqvol , ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol )*/
 _term =  kfBAPTA * _zdsqvol * ca [ _li] ;
 _MATELM1( 0 +  _li ,0 +  _li)  += _term;
 _MATELM1( 16 +  _li ,0 +  _li)  += _term;
 _MATELM1( 8 +  _li ,0 +  _li)  -= _term;
 _term =  kfBAPTA * _zdsqvol * bapta [ _li] ;
 _MATELM1( 0 +  _li ,16 +  _li)  += _term;
 _MATELM1( 16 +  _li ,16 +  _li)  += _term;
 _MATELM1( 8 +  _li ,16 +  _li)  -= _term;
 _term =  ( 0.001 ) * KDBAPTA * kfBAPTA * _zdsqvol ;
 _MATELM1( 0 +  _li ,8 +  _li)  -= _term;
 _MATELM1( 16 +  _li ,8 +  _li)  -= _term;
 _MATELM1( 8 +  _li ,8 +  _li)  += _term;
 /*REACTION*/
  } }
 {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
 _zdsqvol = _zdsq * vrat [ _li ] ;
 /* ~ ca [ _li ] < < ( - _zdsqvol * beta * vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) )*/
 /*FLUX*/
  /* ~ hc [ _li ] <-> ho [ _li ] ( kon * Kinh , kon * ca [ _li ] )*/
 _term =  kon * Kinh ;
 _MATELM1( 24 +  _li ,24 +  _li)  += _term;
 _MATELM1( 20 +  _li ,24 +  _li)  -= _term;
 _term =  kon * ca [ _li ] ;
 _MATELM1( 24 +  _li ,20 +  _li)  -= _term;
 _MATELM1( 20 +  _li ,20 +  _li)  += _term;
 /*REACTION*/
  /* ~ ca [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 /*FLUX*/
  /* ~ ca [ _li ] < < ( _zdsqvol * beta * L [ _li ] * ( 1.0 - ( ca [ _li ] / caer ) ) )*/
 /*FLUX*/
  /* ~ ip3cas [ _li ] < < ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) )*/
 /*FLUX*/
  } }
 jip3 = ( jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) ;
 cip3 = ( _zdsqvol * alpha * jmax * ( 1.0 - ( ca [ 0 ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ 0 ] / ( ca [ 0 ] + Kact ) ) * ho [ 0 ] ) , sites ) ) * ( 2.0 * FARADAY ) / ( PI * ( diam ) ) ;
 ip3ca = ip3cas [ 0 ] ;
 cai = ca [ 0 ] ;
 ca1 = ca [ 1 ] ;
 ca2 = ca [ 2 ] ;
 ca3 = ca [ 3 ] ;
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 32;}
 
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
     _ode_spec1 ();
  _ion_cai = cai;
  }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 32; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 static void _ode_synonym(_nrn_model_sorted_token const& _sorted_token, NrnThread& _nt, Memb_list& _ml_arg, int _type) {
 _nrn_mechanism_cache_range _lmr{_sorted_token, _nt, _ml_arg, _type};
auto* const _ml = &_lmr;
auto const _cnt = _ml_arg._nodecount;
for (int _iml = 0; _iml < _cnt; ++_iml) {
  Datum* _ppvar = _ml_arg._pdata[_iml];
 _ion_cai =  ca [ 0 ] ;
   }
}
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _cvode_sparse(&_cvsparseobj1, 32, _dlist1, neuron::scopmath::row_view{_ml, _iml}, _ode_matsol1, &_coef1);
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 static void* _difspace1;
extern double nrn_nernst_coef(int);
static double _difcoef1(int _i, Memb_list* _ml_arg, size_t _iml, Datum* _ppvar, double* _pdvol, double* _pdfcdc, Datum* _thread, NrnThread* _nt, _nrn_model_sorted_token const& _sorted_token) {
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _ml_arg->_type()};
  auto* const _ml = &_lmr;
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  *_pdvol =  diam * diam * vrat [ ((int) _i ) ] ;
 if (_i ==  0) {
  *_pdfcdc = nrn_nernst_coef(_type_ica)*( ( - ( _ion_dicadv  - ica_pmp_last ) * PI * diam / ( 2.0 * FARADAY ) ));
 }else{ *_pdfcdc=0.;}
   return DCa * diam * diam * vrat [ ((int) _i ) ] ;
return 0;
}
 static void* _difspace2;
extern double nrn_nernst_coef(int);
static double _difcoef2(int _i, Memb_list* _ml_arg, size_t _iml, Datum* _ppvar, double* _pdvol, double* _pdfcdc, Datum* _thread, NrnThread* _nt, _nrn_model_sorted_token const& _sorted_token) {
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _ml_arg->_type()};
  auto* const _ml = &_lmr;
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  *_pdvol =  diam * diam * vrat [ ((int) _i ) ] ; *_pdfcdc=0.;
   return DBufm * diam * diam * vrat [ ((int) _i ) ] ;
return 0;
}
 static void* _difspace3;
extern double nrn_nernst_coef(int);
static double _difcoef3(int _i, Memb_list* _ml_arg, size_t _iml, Datum* _ppvar, double* _pdvol, double* _pdfcdc, Datum* _thread, NrnThread* _nt, _nrn_model_sorted_token const& _sorted_token) {
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _ml_arg->_type()};
  auto* const _ml = &_lmr;
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  *_pdvol =  diam * diam * vrat [ ((int) _i ) ] ; *_pdfcdc=0.;
   return DBufm * diam * diam * vrat [ ((int) _i ) ] ;
return 0;
}
 static void _difusfunc(ldifusfunc2_t _f, _nrn_model_sorted_token const& sorted_token, NrnThread& _nt) {int _i;
  for (_i=0; _i < 4; ++_i) (*_f)(_mechtype, _difcoef1, &_difspace1, _i,  12, 36 , sorted_token, _nt);
  for (_i=0; _i < 4; ++_i) (*_f)(_mechtype, _difcoef2, &_difspace2, _i,  17, 41 , sorted_token, _nt);
  for (_i=0; _i < 4; ++_i) (*_f)(_mechtype, _difcoef3, &_difspace3, _i,  18, 42 , sorted_token, _nt);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
 for (_i=0; _i<4; _i++) bapta[_i] = bapta0;
 for (_i=0; _i<4; _i++) bufm[_i] = bufm0;
 for (_i=0; _i<4; _i++) bufs[_i] = bufs0;
 for (_i=0; _i<4; _i++) cabapta[_i] = cabapta0;
 for (_i=0; _i<4; _i++) cabufm[_i] = cabufm0;
 for (_i=0; _i<4; _i++) cabufs[_i] = cabufs0;
 for (_i=0; _i<4; _i++) ca[_i] = ca0;
 for (_i=0; _i<4; _i++) ho[_i] = ho0;
 for (_i=0; _i<4; _i++) hc[_i] = hc0;
 for (_i=0; _i<4; _i++) ip3cas[_i] = ip3cas0;
 {
   if ( _zfactors_done  == 0.0 ) {
     _zfactors_done = 1.0 ;
     factors ( _threadargs_ ) ;
     }
   cai = cai0 ;
   jip3 = 0.0 ;
   bufs_0 = KDs * TBufs / ( KDs + ( 1000.0 ) * cai0 ) ;
   bufm_0 = KDm * TBufm / ( KDm + ( 1000.0 ) * cai0 ) ;
   bapta_0 = KDBAPTA * TBufBAPTA / ( KDBAPTA + ( 1000.0 ) * cai0 ) ;
   {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
     ca [ _li ] = cai ;
     bufs [ _li ] = bufs_0 ;
     cabufs [ _li ] = TBufs - bufs_0 ;
     bapta [ _li ] = bapta_0 ;
     cabapta [ _li ] = TBufBAPTA - bapta_0 ;
     bufm [ _li ] = bufm_0 ;
     cabufm [ _li ] = TBufm - bufm_0 ;
     } }
   ica = 0.0 ;
   ica_pmp = 0.0 ;
   ica_pmp_last = 0.0 ;
   {int  _li ;for ( _li = 0 ; _li <= 4 - 1 ; _li ++ ) {
     ho [ _li ] = Kinh / ( ca [ _li ] + Kinh ) ;
     hc [ _li ] = 1.0 - ho [ _li ] ;
     _zjx = ( - vmax * pow( ca [ _li ] , 2.0 ) / ( pow( ca [ _li ] , 2.0 ) + pow( Kp , 2.0 ) ) ) ;
     _zjx = _zjx + jmax * ( 1.0 - ( ca [ _li ] / caer ) ) * pow( ( ( ip3i / ( ip3i + Kip3 ) ) * ( ca [ _li ] / ( ca [ _li ] + Kact ) ) * ho [ _li ] ) , sites ) ;
     L [ _li ] = - _zjx / ( 1.0 - ( ca [ _li ] / caer ) ) ;
     } }
   sump = cath ;
   parea = PI * diam ;
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 initmodel();
  _ion_cai = cai;
   nrn_wrote_conc(_ca_sym, _ion_ca_erev, _ion_cai, _ion_cao, _style_ca);
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ica_pmp_last = ica_pmp ;
   ica = ica_pmp ;
   }
 _current += ica;

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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
if (_nt->_vcv) { _ode_spec1(); }
 auto const _g_local = _nrn_current(_v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g_local - _rhs)/.001;
  _ion_cai = cai;
  _ion_ica += ica ;
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 { error = sparse(&_sparseobj1, 32, _slist1, _dlist1, neuron::scopmath::row_view{_ml, _iml}, &t, dt, state, &_coef1, _linmat1);
 if(error){
  std_cerr_stream << "at line 103 in file cal4.mod:\nBREAKPOINT {\n";
  std_cerr_stream << _ml << ' ' << _iml << '\n';
  abort_run(error);
}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 32; ++_i) {
      _ml->data(_iml, _slist1[_i]) += dt*_ml->data(_iml, _dlist1[_i]);
    }}
 }  _ion_cai = cai;
 }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 for(_i=0;_i<4;_i++){_slist1[0+_i] = {bapta_columnindex, _i};  _dlist1[0+_i] = {Dbapta_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[4+_i] = {bufs_columnindex, _i};  _dlist1[4+_i] = {Dbufs_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[8+_i] = {cabapta_columnindex, _i};  _dlist1[8+_i] = {Dcabapta_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[12+_i] = {cabufs_columnindex, _i};  _dlist1[12+_i] = {Dcabufs_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[16+_i] = {ca_columnindex, _i};  _dlist1[16+_i] = {Dca_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[20+_i] = {ho_columnindex, _i};  _dlist1[20+_i] = {Dho_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[24+_i] = {hc_columnindex, _i};  _dlist1[24+_i] = {Dhc_columnindex, _i};}
 for(_i=0;_i<4;_i++){_slist1[28+_i] = {ip3cas_columnindex, _i};  _dlist1[28+_i] = {Dip3cas_columnindex, _i};}
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/Users/romanbaravalle/Documents/Consultancy/JaxleyPYRModel/.cache/neuron_optim_mod/cal4.mod";
    const char* nmodl_file_text = 
  ":Modified from NEURON implementation of Fink et al., 2000\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX cal4\n"
  "	USEION ca READ cao,  ica WRITE cai, ica\n"
  "	:USEION ip3 READ ip3i VALENCE 1\n"
  "	RANGE ica_pmp,ca1, ca2,alpha,beta,ca3,gamma,ip3ca, DCa,jip3, cip3, sites\n"
  " 	GLOBAL vrat, TBufs, KDs, TBufs, TBufm, KDm, KDBAPTA, TBufBAPTA\n"
  "}\n"
  "\n"
  "DEFINE Nannuli 4\n"
  "\n"
  "UNITS {\n"
  "	(mol)	= (1)\n"
  " 	(molar) = (1/liter)\n"
  "  	(uM)    = (micromolar)\n"
  "  	(mM)    = (millimolar)\n"
  "  	(um)    = (micron)\n"
  "  	(mA)    = (milliamp)\n"
  "  	FARADAY = (faraday)  (10000 coulomb)\n"
  "  	PI      = (pi)       (1)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	ip3i = 10e-3 (mM):10e-3 for cal wave 0.16 baseline 16 for noND\n"
  "	cai0 = 50e-6(mM)\n"
  "	:caoc = 2 (mM)\n"
  "	cath = 0.2e-3 (mM) : threshold for ca pump activity\n"
  "	gamma = 8 (um/s) : ca pump flux density\n"
  "	jmax = 3.5e-3 (mM/ms) :3.5e-3\n"
  "	caer = 0.400 (mM)\n"
  "	Kip3 = 0.8e-3 (mM)\n"
  "	Kact = 0.7e-3 (mM)\n"
  "	kon = 2.7 (/mM-ms) :2.7\n"
  "	Kinh = 0.6e-3 (mM)\n"
  "	sites=3\n"
  "	alpha = 1 (1) : relative abundance of ER mechanisms : alpha only specific for ip3 receptor,\n"
  "	beta  = 1(1)           :introducing beta to take care of other ER mechanisms(SERCA and leak channel density)\n"
  "\n"
  "	vmax =1e-4   (mM/ms) :1e-4 revised\n"
  "	Kp = 0.27e-3 (mM)	:0.27e-3\n"
  "	DCa = 0.22 (um2/ms) :Fink et al 2000\n"
  "	TBufs = 0.45 (mM)\n"
  "        kfs = 1000 (/mM-ms) : try these for now\n"
  "        KDs = 10 (uM)\n"
  "    TBufBAPTA = 0 (mM) :10 uM for concentration of BAPTA\n"
  "        kfBAPTA = 500 (/mM-ms) : try these for now\n"
  "        KDBAPTA = 0.2 (uM)	:KD is 0.2uM\n"
  "	TBufm = 0.075 (mM)\n"
  "	kfm = 1000 (/mM-ms) : try these for now\n"
  "        KDm = 0.24 (uM)\n"
  "        DBufm = 0.050 (um2/ms)\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	diam      (um)\n"
  "	cip3		(mA/cm2)\n"
  "  	ica       (mA/cm2)\n"
  "        cai       (mM)\n"
  "	jip3	  (mM/ms)\n"
  "        ca1	  (mM)\n"
  "        ca2       (mM)\n"
  "        ca3       (mM)\n"
  "        ica_pmp   (mA/cm2)\n"
  "	ica_pmp_last   (mA/cm2)\n"
  "        parea     (um)    :pump area peer unit length\n"
  "        sump      (mM)\n"
  "        cao       (mM)\n"
  "        :ip3i      (mM)\n"
  "        jchnl    (mM/ms)\n"
  "        vrat[Nannuli]  (1)\n"
  "	L[Nannuli] (mM/ms)  adjusted so that\n"
  "                         : jchnl + jpump + jleak = 0  when  ca = 0.05 uM and h = Kinh/(ca + Kinh)\n"
  "        bufs_0 (mM)\n"
  "	bufm_0 (mM)\n"
  "	bapta_0 (mM)\n"
  "	\n"
  "	ip3ca	(mM)\n"
  "} \n"
  "\n"
  "\n"
  "CONSTANT { volo = 1e10 (um2) }\n"
  "\n"
  "STATE {\n"
  "     	ca[Nannuli]     (mM) <1e-7>\n"
  "     	hc[Nannuli]    \n"
  "     	ho[Nannuli]\n"
  "     	bufs[Nannuli]    (mM) <1e-3>\n"
  "     	cabufs[Nannuli]  (mM) <1e-7>\n"
  "	bufm[Nannuli]    (mM) <1e-4>\n"
  "        cabufm[Nannuli]  (mM) <1e-8>\n"
  "        bapta[Nannuli]    (mM) <1e-3>\n"
  "     	cabapta[Nannuli]  (mM) <1e-7>\n"
  "	ip3cas [Nannuli] (mM)\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "     	SOLVE state METHOD sparse\n"
  "     	ica_pmp_last = ica_pmp\n"
  "     	ica = ica_pmp\n"
  "\n"
  "}\n"
  "LOCAL factors_done, jx\n"
  "INITIAL {\n"
  "	\n"
  "    	if (factors_done==0) {\n"
  "		factors_done= 1\n"
  "		factors()\n"
  "    	}\n"
  " \n"
  "        cai = cai0\n"
  "	jip3=0\n"
  "	bufs_0 = KDs*TBufs/(KDs + (1000)*cai0)\n"
  "	bufm_0 = KDm*TBufm/(KDm + (1000)*cai0)\n"
  "	bapta_0 = KDBAPTA*TBufBAPTA/(KDBAPTA + (1000)*cai0)\n"
  "\n"
  "	FROM i=0 TO Nannuli-1 {    \n"
  "     		ca[i] = cai\n"
  "		bufs[i] = bufs_0\n"
  "       		cabufs[i] = TBufs - bufs_0\n"
  "       	bapta[i] = bapta_0\n"
  "       		cabapta[i] = TBufBAPTA - bapta_0\n"
  "		bufm[i] = bufm_0\n"
  "    		cabufm[i] = TBufm - bufm_0\n"
  "\n"
  "   	}\n"
  "	\n"
  "   	ica=0\n"
  "   	ica_pmp = 0 \n"
  "   	ica_pmp_last = 0\n"
  "\n"
  "\n"
  "	FROM i=0 TO Nannuli-1 {\n"
  "    		ho[i] = Kinh/(ca[i]+Kinh)\n"
  "    		hc[i] = 1-ho[i]\n"
  "    		jx = (-vmax*ca[i]^2 / (ca[i]^2 + Kp^2))\n"
  "    		jx = jx + jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^sites\n"
  "     	   	L[i] = -jx/(1 - (ca[i]/caer))\n"
  "    	}\n"
  "\n"
  "    	sump = cath\n"
  "    	parea = PI*diam   \n"
  "}\n"
  "\n"
  "LOCAL frat[Nannuli]\n"
  "\n"
  "PROCEDURE factors() {\n"
  "	LOCAL r, dr2\n"
  "  	r = 1/2                : starts at edge (half diam)\n"
  "  	dr2 = r/(Nannuli-1)/2  : full thickness of outermost annulus,\n"
  "                               : half thickness of all other annuli\n"
  "  	vrat[0] = 0\n"
  "  	frat[0] = 2*r\n"
  "\n"
  "  	FROM i=0 TO Nannuli-2 {\n"
  "    		vrat[i] = vrat[i] + PI*(r-dr2/2)*2*dr2  : interior half\n"
  "   		 r = r - dr2\n"
  "   		 frat[i+1] = 2*PI*r/(2*dr2)  : outer radius of annulus\n"
  "                                             : div by distance between centers\n"
  "   		 r = r - dr2\n"
  "    		vrat[i+1] = PI*(r+dr2/2)*2*dr2  : outer half of annulus\n"
  "  	}\n"
  "}\n"
  "\n"
  "\n"
  "LOCAL dsq, dsqvol\n"
  "\n"
  "KINETIC state {\n"
  "  	COMPARTMENT i, diam*diam*vrat[i] {ca  bufs cabufs bufm cabufm sump}\n"
  "  	COMPARTMENT volo {cao}\n"
  "  	LONGITUDINAL_DIFFUSION i, DCa*diam*diam*vrat[i] {ca}\n"
  "  	LONGITUDINAL_DIFFUSION i, DBufm*diam*diam*vrat[i] {bufm cabufm}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "        :cell membrane ca pump\n"
  "  	~ ca[0] <-> sump  ((0.001)*parea*gamma*u(ca[0]/(1 (mM)), cath/(1 (mM))), (0.001)*parea*gamma*u(ca[0]/(1 (mM)), cath/(1 (mM))))\n"
  "  	ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea\n"
  "\n"
  "  	: all currents except cell membrane ca pump\n"
  "  	~ ca[0] << (-(ica - ica_pmp_last)*PI*diam/(2*FARADAY))  : ica is Ca efflux\n"
  "\n"
  " 	 : radial diffusion\n"
  "   	FROM i=0 TO Nannuli-2 {\n"
  "   		~ ca[i] <-> ca[i+1] (DCa*frat[i+1], DCa*frat[i+1])\n"
  " 	}\n"
  "\n"
  "	: buffering\n"
  "   	dsq = diam*diam\n"
  "   \n"
  "   	FROM i=0 TO Nannuli-1 {\n"
  "	 	dsqvol = dsq*vrat[i]\n"
  "     	 	~ ca[i] + bufs[i] <-> cabufs[i]  (kfs*dsqvol, (0.001)*KDs*kfs*dsqvol)\n"
  "     	 	~ ca[i] + bapta[i] <-> cabapta[i]  (kfBAPTA*dsqvol, (0.001)*KDBAPTA*kfBAPTA*dsqvol)\n"
  "		:~ ca[i] + bufm[i] <-> cabufm[i]  (kfm*dsqvol, (0.001)*KDm*kfm*dsqvol) :to simulate high affinity dyes, used only for the simplified 3 cylinder model in the paper\n"
  "\n"
  "    	}\n"
  "\n"
  "\n"
  "       	:SERCA pump, channel\n"
  "  	FROM i=0 TO Nannuli-1 {\n"
  "    		dsqvol = dsq*vrat[i]\n"
  "\n"
  "   	 	: pump\n"
  "   	 	~ ca[i] << (-dsqvol*beta*vmax*ca[i]^2 / (ca[i]^2 + Kp^2))\n"
  "\n"
  "    		: channel\n"
  "   	 	~ hc[i] <-> ho[i]  (kon*Kinh, kon*ca[i])\n"
  "   	 	~ ca[i] << ( dsqvol*alpha*jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^sites )\n"
  " 	 	: leak\n"
  "   	 	~ ca[i] << (dsqvol*beta*L[i]*(1 - (ca[i]/caer)))\n"
  "		~ ip3cas[i] << (dsqvol*alpha*jmax*(1-(ca[i]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[i]/(ca[i]+Kact)) * ho[i] )^sites )\n"
  "  	}\n"
  "	\n"
  "	jip3 = (jmax*(1-(ca[0]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[0]/(ca[0]+Kact)) * ho[0] )^sites ) \n"
  "	: turn this into a current ican can read\n"
  "	:make the diam small assuming its only being released into certain parts\n"
  "	cip3 = ( dsqvol*alpha*jmax*(1-(ca[0]/caer)) * ( (ip3i/(ip3i+Kip3)) * (ca[0]/(ca[0]+Kact)) * ho[0] )^sites )*(2*FARADAY)/(PI*(diam))\n"
  "\n"
  ":	ip3ca=0\n"
  ":	FROM i=0 TO Nannuli-1 {\n"
  ":		ip3ca=ip3ca+ip3cas[i]\n"
  ":	}\n"
  "\n"
  "	ip3ca=ip3cas[0]\n"
  "\n"
  "  	cai = ca[0]\n"
  "  	ca1 = ca[1]\n"
  "  	ca2 = ca[2]\n"
  "  	ca3 = ca[3]\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION u (x, th) {\n"
  "  	if (x>th) {\n"
  "    		u = 1\n"
  "  	} else {\n"
  "    		u = 0\n"
  "  	}\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
