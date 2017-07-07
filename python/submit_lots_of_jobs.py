import os

# ----------- This block to make stencils ---------------
# # inlats = [15, 25, 35, 45, 55]
# # inlats = [20, 30, 40, 50]
# # inlats = [15, 20, 25, 30, 35, 40, 45, 50, 55]
# inlats = [ 40, 45, 50, 55]
# kps = [6]
# side = 'nightside'
# for kp in kps:
#     for inlat in inlats:
#         cmd = 'qsub -N %s_%d_kp%d'%(side, inlat, kp)  + ' -v inlat=%d'%inlat +\
#             ',niteday=%s,kp=%d jobs/stencil_input2.pbs'%(side,kp)

#         os.system(cmd)


# ----------- This block to run seasonal stats ---------

suffixes = ['AE8MAX_flux_0','AE8MIN_flux_0','AE8MAX_flux_2']
# suffixes = ['AE8MAX_flux_0']
modes = ['energy']

for suffix in suffixes:
    for mode in modes:
        cmd = 'qsub -N %s_%s'%(mode, suffix)  +\
        ' -o logs/postrun_log_seasonal_%s_%s.txt'%(mode,suffix) +\
        ' -v suffix=%s,precip_mode=%s'%(suffix, mode) +\
            ' jobs/seasonal_job.pbs'
        print cmd
        os.system(cmd)