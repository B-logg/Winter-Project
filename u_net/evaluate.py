from cmath import nan
import numpy as np

def corr(A,B):
  ################ with zero
    # a = A-A.mean()
    # b = B-B.mean()
    # return (a*b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

  ################ without zero
  A = np.where(B!=0,A,np.nan)
  B = np.where(B!=0,B,np.nan)
  a = A-np.nanmean(A)
  b = B-np.nanmean(B)
  if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
    return nan
  else: 
    return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))

def corr_wZero(A,B):
  ################ with zero
    a = A-A.mean()
    b = B-B.mean()
    if (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())) ==0:
        return nan
    else:
      return (a*b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))
 
def corr_wCla(A,B,C):  
  A = np.where(C!=255,A,np.nan)
  B = np.where(C!=255,B,np.nan)  
  # A = np.where(B!=0,A,np.nan)
  # B = np.where(B!=0,B,np.nan)
  # B_nonnan_cnt = np.count_nonzero(~np.isnan(B))
  # B_nz_cnt = np.count_nonzero(B)
  a = A-np.nanmean(A)
  b = B-np.nanmean(B)
  if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
    return nan#, B_nonnan_cnt, B_nz_cnt
  else: 
    return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))#, B_nonnan_cnt, B_nz_cnt
 
########################################################################################################################
  
def r_square(A,B):  
  A = np.where(B!=0,A,np.nan)
  B = np.where(B!=0,B,np.nan)
  if np.nansum((B-np.nanmean(B))**2) == 0:
    return nan
  else:
    return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )

def r_square_wZero(A,B):  
  if np.nansum((B-np.nanmean(B))**2) ==0:
    return nan
  else:
    return 1.0 - (    ((B-A)**2).sum() / ((B-B.mean())**2).sum() )

def r_square_wCla(A,B,C):  
  A = np.where(C!=255,A,np.nan)
  B = np.where(C!=255,B,np.nan)
  # A = np.where(B!=0,A,np.nan)
  # B = np.where(B!=0,B,np.nan)
  if np.nansum((B-np.nanmean(B))**2) == 0:
    return nan
  else:
    return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )  


