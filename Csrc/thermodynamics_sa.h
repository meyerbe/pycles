#pragma once
#include "parameters.h"
#include "grid.h"
#include "thermodynamic_functions.h"
#include "advection_interpolation.h"
#include "lookup.h"
#include "entropies.h"
#include <stdio.h>

inline double temperature_no_ql(double pd, double pv, double s, double qt){
    return T_tilde * exp((s -
                            (1.0-qt)*(sd_tilde - Rd * log(pd/p_tilde))
                            - qt * (sv_tilde - Rv * log(pv/p_tilde)))
                            /((1.0-qt)*cpd + qt * cpv));
}





void eos_c(struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
                    const double p0, const double s, const double qt, double* T, double* qv, double* qc){
    *qc = 0.0;
    *qv = qt;
    double pd_1 = pd_c(p0,qt,qt);
    double pv_1 = pv_c(p0,qt,qt );
    double T_1 = temperature_no_ql(pd_1,pv_1,s,qt);
    double pv_star_1 = lookup(LT, T_1);
    double qv_star_1 = qv_star_c(p0,qt,pv_star_1);

    ///printf("%f\t%f\t%f\t%f\t%f\n",p0,s,qv_star_1,qt-qv_star_1,T_1);
    /// If not saturated
    if(qt <= qv_star_1){
        *T = T_1;
        return;
    }
    else{
        double sigma_1 = qt - qv_star_1;
        double lam_1 = lam_fp(T_1);
        double L_1 = L_fp(lam_1,T_1);
        double s_1 = sd_c(pd_1,T_1) * (1.0 - qt) + sv_c(pv_1,T_1) * qt + sc_c(L_1,T_1)*sigma_1;
        double f_1 = s - s_1;
        double T_2 = T_1 + sigma_1 * L_1 /((1.0 - qt)*cpd + qv_star_1 * cpv);
        double delta_T  = fabs(T_2 - T_1);
        double qv_star_2;
        double lam_2;
        do{
            double pv_star_2 = lookup(LT, T_2);
            qv_star_2 = qv_star_c(p0,qt,pv_star_2);
            double pd_2 = pd_c(p0,qt,qv_star_2);
            double pv_2 = pv_c(p0,qt,qv_star_2);
            double sigma_2 = qt - qv_star_2;
            lam_2 = lam_fp(T_2);
            double L_2 = L_fp(lam_2,T_2);
            double s_2 = sd_c(pd_2,T_2) * (1.0 - qt) + sv_c(pv_2,T_2) * qt + sc_c(L_2,T_2)*sigma_2;
            double f_2 = s - s_2;
            double T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1);
            T_1 = T_2;
            T_2 = T_n;
            f_1 = f_2;
            delta_T  = fabs(T_2 - T_1);
        } while(delta_T >= 1.0e-3);
        *T  = T_2;
        *qv = qv_star_2;
        *qc = (qt - qv_star_2);
        return;
    }
}


void eos_update(struct DimStruct *dims, struct LookupStruct *LT, double (*lam_fp)(double), double (*L_fp)(double, double),
    double* restrict p0, double* restrict s, double* restrict qt, double* restrict T,
    double* restrict qv, double* restrict ql, double* restrict qi, double* restrict alpha ){

    long i,j,k;
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];
    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;
    const long imax = dims->nlg[0];
    const long jmax = dims->nlg[1];
    const long kmax = dims->nlg[2];


    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    double qc;
                    ///printf("%f\n",s[ijk]);
                    eos_c(LT, lam_fp, L_fp, p0[k], s[ijk],qt[ijk],&T[ijk],&qv[ijk],&qc);
                    const double lam = lam_fp(T[ijk]);
                    ql[ijk] = lam * qc;
                    qi[ijk] = (1.0 - lam) * qc;
                    alpha[ijk] = alpha_c(p0[k], T[ijk],qt[ijk],qv[ijk]);
                }
            }
        }


    }


void buoyancy_update_sa(struct DimStruct *dims, double* restrict alpha0, double* restrict alpha, double* restrict buoyancy, double* restrict wt){

    long i,j,k;
    const long istride = dims->nlg[1] * dims->nlg[2];
    const long jstride = dims->nlg[2];
    const long imin = 0;
    const long jmin = 0;
    const long kmin = 0;
    const long imax = dims->nlg[0];
    const long jmax = dims->nlg[1];
    const long kmax = dims->nlg[2];

    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    buoyancy[ijk] = buoyancy_c(alpha0[k],alpha[ijk]);

                };
        };
    };


    for (i=imin; i<imax; i++){
       const long ishift = i * istride;
        for (j=jmin;j<jmax;j++){
            const long jshift = j * jstride;
                for (k=kmin;k<kmax;k++){
                    const long ijk = ishift + jshift + k;
                    wt[ijk] = wt[ijk] + interp_4(buoyancy[ijk-1],buoyancy[ijk],buoyancy[ijk+1],buoyancy[ijk+2]);
                };
        };
    };

    return;
}
