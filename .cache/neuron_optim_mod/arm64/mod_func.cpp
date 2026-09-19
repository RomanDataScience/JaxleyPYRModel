#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern "C" void _Kv2like_reg(void);
extern "C" void _Nav16_a_reg(void);
extern "C" void _PotassiumInwardRectifier_reg(void);
extern "C" void _cagk_reg(void);
extern "C" void _cal_reg(void);
extern "C" void _cal4_reg(void);
extern "C" void _calH_reg(void);
extern "C" void _car_reg(void);
extern "C" void _cat_reg(void);
extern "C" void _d3_reg(void);
extern "C" void _h_reg(void);
extern "C" void _icand_reg(void);
extern "C" void _kadist_reg(void);
extern "C" void _kaprox_reg(void);
extern "C" void _kca_reg(void);
extern "C" void _kd_reg(void);
extern "C" void _km_reg(void);
extern "C" void _na3dend_reg(void);
extern "C" void _nap_reg(void);
extern "C" void _nax_reg(void);

extern "C" void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"Kv2like.mod\"");
    fprintf(stderr, " \"Nav16_a.mod\"");
    fprintf(stderr, " \"PotassiumInwardRectifier.mod\"");
    fprintf(stderr, " \"cagk.mod\"");
    fprintf(stderr, " \"cal.mod\"");
    fprintf(stderr, " \"cal4.mod\"");
    fprintf(stderr, " \"calH.mod\"");
    fprintf(stderr, " \"car.mod\"");
    fprintf(stderr, " \"cat.mod\"");
    fprintf(stderr, " \"d3.mod\"");
    fprintf(stderr, " \"h.mod\"");
    fprintf(stderr, " \"icand.mod\"");
    fprintf(stderr, " \"kadist.mod\"");
    fprintf(stderr, " \"kaprox.mod\"");
    fprintf(stderr, " \"kca.mod\"");
    fprintf(stderr, " \"kd.mod\"");
    fprintf(stderr, " \"km.mod\"");
    fprintf(stderr, " \"na3dend.mod\"");
    fprintf(stderr, " \"nap.mod\"");
    fprintf(stderr, " \"nax.mod\"");
    fprintf(stderr, "\n");
  }
  _Kv2like_reg();
  _Nav16_a_reg();
  _PotassiumInwardRectifier_reg();
  _cagk_reg();
  _cal_reg();
  _cal4_reg();
  _calH_reg();
  _car_reg();
  _cat_reg();
  _d3_reg();
  _h_reg();
  _icand_reg();
  _kadist_reg();
  _kaprox_reg();
  _kca_reg();
  _kd_reg();
  _km_reg();
  _na3dend_reg();
  _nap_reg();
  _nax_reg();
}
