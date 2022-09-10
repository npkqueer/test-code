#q,m,k,i,l,j,x,y,
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from astropy.stats import sigma_clip
import glob
import math
import datetime as dt
import wget
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.coordinates import Angle
from astropy.table import Table
from tqdm import tqdm
from scipy.stats import f

def mjd_to_date(mjd):
    mjd = mjd + 2400000.5
    jd = mjd + 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = math.trunc((I - 1867216.25)/36524.25)
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I  
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    day = C - E + F - math.trunc(30.6001 * G)
    day = int(day)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    date = f"{day}-{month}-{year}"
    return date
#This is to convert session duration to hours
def dur_to_time(val):
    mi,hr = math.modf(val)
    hr = int(hr)
    mi = mi*60
    mi = round(mi,2)
    time = f"{hr} Hrs, {mi} Mins"
    return time
#list of quasars imported 
qsr_RA, qsr_DEC = np.loadtxt("/home/ubuntu/INOV/mywork14/qsrdata.dat",unpack=True,usecols=[1,2])
#Downloading ZTF data of all quasars
#radius limit is 1.5 arcsec
col1,col2,col3,col4,col5,col6,col7,col8 = [],[],[],[],[],[],[],[]
variability = []
for q in tqdm(range(len(qsr_RA))):
    qlink = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{qsr_RA[q]}+{qsr_DEC[q]}+0.000416667&BANDNAME=r&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table"
    qfilename = f"{round(qsr_RA[q],5)}{round(qsr_DEC[q],5)}_qsr.dat"
    wget.download(qlink, qfilename)
    qsr_files = glob.glob('/home/ubuntu/INOV/mywork15/*qsr.dat', recursive = True) #loading ZTF data of all quasars
    qsrdata = []
    for doc1 in qsr_files:
        file1 = open(doc1, 'r')
        lines1 = file1.readlines()
        #print(len(lines1))
        if(len(lines1)<=54):
            continue
        else:
            qsrdata.append(doc1) #Quasars having data is saved in this array
    for m in range(len(qsrdata)):
        qsr_obsid,qsr_mjd,qsr_mag,qsr_magerr,qsr_catflags,qsr_ra,qsr_dec = np.loadtxt(qsrdata[m],unpack=True,usecols=[0,3,4,5,6,8,9],skiprows=54)
        qsr_cat_ind = np.where(qsr_catflags == 0)[0] #catflag criteria
        qsr_obsid,qsr_mjd,qsr_mag,qsr_magerr,qsr_catflags,qsr_ra,qsr_dec = qsr_obsid[qsr_cat_ind],qsr_mjd[qsr_cat_ind],qsr_mag[qsr_cat_ind],qsr_magerr[qsr_cat_ind],qsr_catflags[qsr_cat_ind],qsr_ra[qsr_cat_ind],qsr_dec[qsr_cat_ind]
        qsr_mjd_ind = np.argsort(qsr_mjd, axis = 0) #mjd sorting
        qsr_obsid,qsr_mjd,qsr_mag,qsr_magerr,qsr_catflags,qsr_ra,qsr_dec = qsr_obsid[qsr_mjd_ind],qsr_mjd[qsr_mjd_ind],qsr_mag[qsr_mjd_ind],qsr_magerr[qsr_mjd_ind],qsr_catflags[qsr_mjd_ind],qsr_ra[qsr_mjd_ind],qsr_dec[qsr_mjd_ind]
        diff_arr = []
        for k in range(len(qsr_mjd)):
            if(k==0):
                diff_arr.append(float(0))
            else:
                cdiff = abs(qsr_mjd[k]-qsr_mjd[k-1])
                diff_arr.append(cdiff) #difference between consecutive mjds stored
        gap_ind = np.where(np.array(diff_arr)>= (15/(60*24)))[0] #indices with gap greater than 15 minutes
        all_session_mjds, all_session_mjds_UT, all_session_mags,all_session_errs = [],[],[],[]
        all_session_durs,all_session_counts = [],[]
        for i in range(len(gap_ind)):
            if(i == len(gap_ind)-1):
                continue    
            session_mjd = qsr_mjd[gap_ind[i]:gap_ind[i+1]]
            session_mjd_int = np.array(session_mjd,dtype=int)
            session_mjd_UT = session_mjd  - np.min(session_mjd_int)
            session_mag = qsr_mag[gap_ind[i]:gap_ind[i+1]]
            session_err = qsr_magerr[gap_ind[i]:gap_ind[i+1]]
            session_dur = (np.max(session_mjd)-np.min(session_mjd))*24
            session_counts = gap_ind[i+1]-gap_ind[i]
            if(session_counts>=10 and session_dur>=0):
                date = mjd_to_date(np.min(session_mjd_int))
                time = dur_to_time(session_dur)
                #print(session_counts, 'datapoints in current session', time)
                #print('session max mjd=',np.max(session_mjd),'and session min mjd=',np.min(session_mjd),"; Date =",date)
                all_session_mjds.append(session_mjd)
                all_session_mjds_UT.append(session_mjd_UT)
                all_session_durs.append(session_dur)
                all_session_counts.append(session_counts)
                all_session_mags.append(session_mag)
                all_session_errs.append(session_err)
        #Downloading GAIA data if quasar
        qsr_radius = Angle(2/60, u.arcminute)
        qsr_rlimit = qsr_radius.degree 
        qsr_gaia_ra = qsr_RA[q]
        qsr_gaia_dec = qsr_DEC[q]
        qsr_job = Gaia.launch_job("SELECT TOP 2000 "
                          "gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,gaia_source.pm,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.phot_g_mean_mag "
                          "FROM gaiaedr3.gaia_source "
                          f"WHERE CONTAINS(POINT('ICRS',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),CIRCLE('ICRS',{qsr_gaia_ra},{qsr_gaia_dec},{qsr_rlimit}))=1 ",
                          dump_to_file=True, output_format='votable')
        print(qsr_job.outputFile)
        gaia_qsr = qsr_job.get_results()
        qsr_mag_g = gaia_qsr["phot_g_mean_mag"]
        #Downloading GAIA data of stars
        str_radius = Angle(20, u.arcminute)
        str_rlimit = str_radius.degree 
        str_ra = qsr_RA[q]
        str_dec = qsr_DEC[q]
        pmlimit = 20
        magupperlimit = qsr_mag_g+0.5
        maglowerlimit = qsr_mag_g-1.5
        str_job = Gaia.launch_job("SELECT TOP 2000 "
                          "gaia_source.source_id,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.parallax_error,gaia_source.pm,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.phot_g_mean_mag,gaia_source.dr2_radial_velocity,gaia_source.dr2_radial_velocity_error "
                          "FROM gaiaedr3.gaia_source "
                          f"WHERE CONTAINS(POINT('ICRS',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),CIRCLE('ICRS',{str_ra},{str_dec},{str_rlimit}))=1 "
                          f"AND (gaiaedr3.gaia_source.pm>={pmlimit} "
                          f"AND gaiaedr3.gaia_source.phot_g_mean_mag>={maglowerlimit} "
                          f"AND gaiaedr3.gaia_source.phot_g_mean_mag<={magupperlimit}) ", 
                          dump_to_file=True, output_format='votable')
        #print(str_job.outputFile)
        gaia_str = str_job.get_results()
        #print(gaia_str)
        for l in range(len(all_session_mjds)):
            #quasar data
            ses_mjd = np.array(all_session_mjds[l])
            ses_mag = np.array(all_session_mags[l])
            ses_magerr = np.array(all_session_errs[l])
            ses_clipped_mag = sigma_clip(ses_mag,sigma=5,maxiters=3,masked=True)
            ses_unmasked_id = ma.nonzero(ses_clipped_mag)
            ses_mjd,ses_mag,ses_magerr = np.array(ses_mjd)[ses_unmasked_id],np.array(ses_mag)[ses_unmasked_id],np.array(ses_magerr)[ses_unmasked_id]
            ses_mjd_int = np.array(ses_mjd,dtype=int)
            ses_mjd_UT = ses_mjd  - np.min(ses_mjd_int)
            gRA = gaia_str['ra']
            gDEC = gaia_str['dec']
            MJDMIN = np.min(ses_mjd)
            MJDMAX = np.max(ses_mjd)
            for n in tqdm(range(len(gRA))):
                slink = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{gRA[n]}+{gDEC[n]}+0.000416667&TIME={MJDMIN}+{MJDMAX}&BANDNAME=r&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table"
                sfilename = f"{round(gRA[n],5)}{round(gDEC[n],5)}_str.dat"
                wget.download(slink, sfilename)
            str_files = glob.glob('/home/ubuntu/INOV/mywork10/*str.dat', recursive = True)
            str_data = []
            for doc2 in str_files:
                file2 = open(doc2, 'r')
                lines2 = file2.readlines()
                #print(len(lines2))
                if(len(lines2)<=54):
                    continue
                else:
                    str_data.append(doc2)
            chi_val = []
            test_mjd, test_mag, test_err = [], [], []
            for j in range(len(str_data)):
                str_obsid,str_mjd,str_mag,str_magerr,str_catflags,str_ra,str_dec = np.loadtxt(str_data[j],unpack=True,usecols=[0,3,4,5,6,8,9],skiprows=54)
                str_cat_ind = np.where(str_catflags == 0)[0]
                str_obsid,str_mjd,str_mag,str_magerr,str_catflags,str_ra,str_dec = str_obsid[str_cat_ind],str_mjd[str_cat_ind],str_mag[str_cat_ind],str_magerr[str_cat_ind],str_catflags[str_cat_ind],str_ra[str_cat_ind],str_dec[str_cat_ind]
                str_mjd_ind = np.argsort(str_mjd, axis = 0)
                str_obsid,str_mjd,str_mag,str_magerr,str_catflags,str_ra,str_dec = str_obsid[str_mjd_ind],str_mjd[str_mjd_ind],str_mag[str_mjd_ind],str_magerr[str_mjd_ind],str_catflags[str_mjd_ind],str_ra[str_mjd_ind],str_dec[str_mjd_ind]
                str_clipped_mag = sigma_clip(str_mag,sigma=5,maxiters=3,masked=True)
                str_unmasked_id = ma.nonzero(str_clipped_mag)
                str_obsid,str_mjd,str_mag,str_magerr,str_catflags,str_ra,str_dec = np.array(str_obsid)[str_unmasked_id],np.array(str_mjd)[str_unmasked_id],np.array(str_mag)[str_unmasked_id],np.array(str_magerr)[str_unmasked_id],np.array(str_catflags)[str_unmasked_id],np.array(str_ra)[str_unmasked_id],np.array(str_dec)[str_unmasked_id] 
                str_mjd_int = np.array(str_mjd,dtype=int)
                str_mjd_UT = str_mjd  - np.min(str_mjd_int)
                nodpts = int(len(ses_mjd) * 0.9) #90%
                str_mean_mag = np.mean(str_mag)
                chi_squared = (sum((str_mag-np.mean(str_mag))**2/str_magerr**2))/len(str_mag-1)
                #print(f'chi_square of star {j} = {chi_squared}')
                chi_minlimit = 0.8
                chi_maxlimit = 1.5
                if(len(str_mjd)<=nodpts):
                    continue
                if(str_mean_mag>qsr_mag_g):
                     continue
                if(chi_squared < chi_maxlimit and chi_squared > chi_minlimit):
                    chi_val.append(chi_squared)
                    test_mjd.append(str_mjd)
                    test_mag.append(str_mag)
                    test_err.append(str_magerr)
            chi_sort = np.argsort(chi_val)
            x = chi_sort[0]
            y = chi_sort[1]
            C1 = np.where(test_mjd[x][np.in1d(test_mjd[x], test_mjd[y])])
            C2 = np.where(test_mjd[y][np.in1d(test_mjd[y], test_mjd[x])])
            D1 = np.where(ses_mjd[np.in1d(ses_mjd,test_mjd[x])])
            D2 = np.where(test_mjd[x][np.in1d(test_mjd[x],ses_mjd)])
            E1 = np.where(ses_mjd[np.in1d(ses_mjd,test_mjd[y])])
            E2 = np.where(test_mjd[y][np.in1d(test_mjd[y],ses_mjd)])
            QS1 = ses_mag[D1] - test_mag[x][D2]
            QS2 = ses_mag[E1] - test_mag[y][E2]
            S1S2 = test_mag[x][C1] - test_mag[y][C2]
            QS1_e = np.sqrt(ses_magerr[D1]**2 + test_err[x][D2]**2)
            QS2_e = np.sqrt(ses_magerr[E1]**2 + test_err[y][E2]**2)
            S1S2_e = np.sqrt(test_err[x][C1]**2 + test_err[y][C2]**2)
            # critical values
            F_c_95 = (f.ppf(0.95, len(QS1)-1, len(QS1)-1))
            F_c_99 = (f.ppf(0.999, len(QS1)-1, len(QS1)-1))
            # variance of LCs
            var_QS1 = ( (QS1 - np.mean(QS1))**2 ).sum()/(len(QS1)-1) 
            var_QS2 = ( (QS2 - np.mean(QS2))**2 ).sum()/(len(QS2)-1) 
            var_S1S2 =( (S1S2 - np.mean(S1S2))**2 ).sum()/(len(S1S2)-1)
            # standard deviation of LCs 
            sig_QS1 = np.mean(QS1_e**2)
            sig_QS2 = np.mean(QS2_e**2)
            sig_S1S2 = np.mean(S1S2_e**2)
            eta = 1.5
            eta_s = 1.5*1.5
            si1, si2 = (max(QS1) - min(QS1) )**2 - 2*( eta**2 * np.mean(QS1_e**2) ), (max(QS2) - min(QS2) )**2 - 2*( eta**2 * np.mean(QS2_e**2) )
            var_am = (si1 + si2)/2.0 
            precision = 0.5*( np.sqrt( eta**2 * np.mean(QS1_e**2) ) + np.sqrt( eta**2 * np.mean( QS2_e**2) ) ) 
            # F-eta test
            f1_eta = var_QS1 / (eta_s * sig_QS1)
            f2_eta = var_QS2 / (eta_s * sig_QS2)  
						##test
            if f1_eta>=F_c_99:
                V = "Variable"
                variability.append(V)
            elif f1_eta > F_c_95 and f1_eta <=F_c_99: 
                V = "Probable Variable"
                variability.append(V)
            else:
                V = "Non Variable"
                variability.append(V)
            if f2_eta>=F_c_99: 
                V = "Variable"
                variability.append(V)
            elif f2_eta > F_c_95 and f2_eta <=F_c_99:
                V = "Probable Variable"
                variability.append(V)
            else:
                V = "Non Variable"
                variability.append(V)
            col1.append(f"QSR_{q}")
            col2.append(date)
            col3.append(time)
            col4.append(len(ses_mag))
            col5.append(round(f1_eta,6))
            col6.append(round(f2_eta,6))
            col7.append(round(si1,6))
            col8.append(round(si2,6))
            
result = Table()
result['Quasar Name'] = col1
result['Session Date'] = col2
result['Session Duration'] = col3
result['Session Datapoints'] = col4
result['f1_eta'] = col5
result['f2_eta'] = col6
result['Psi1'] = col7
result['Psi2'] = col8
result['Variability Status'] = variability
#print(result)
ascii.write(result, 'result.dat', overwrite=True)
