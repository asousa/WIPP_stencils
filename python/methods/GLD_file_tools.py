#import threading
import time
#from Queue import Queue
import logging
import numpy as np
import os
import datetime
#import matplotlib
#matplotlib.use('GTKAgg')
#from matplotlib import pyplot as plt
#from mpl_toolkits.basemap import Basemap
import datetime
#import ephem
import math
#from StreamReader import StreamReader
#import json
import fnmatch




# ----------------------------------------------------------
# A set of modules to quickly search and parse GLD entries
# ----------------------------------------------------------
#
# Version 1.1:  Added logic to deal with overlapping files
#               (asking for flashes spanning two files)
#               6.8.2016 APS
#
# Version 1.2:  Added try/catch blocks around datetime_from_row
#               added line[0] == 0 condition for happy rows
#               (Attempting to bulletproof reads on junky files)
#               6.14.2016 APS                
# ----------------------------------------------------------
class GLD_file_tools(object):
  def __init__(self,filepath, prefix='GLD'):
    self.GLD_root = filepath
    self.file_list = []
    self.suffix = '.dat'
    self.prefix = prefix
    #self.refresh_directory()

  

  def refresh_directory(self):
    ''' Get file list within directory '''
    logging.info('Refreshing file list')
    self.file_list = []
    # Get datetime objects for each file in directory:
    for root, dirs, files in os.walk(self.GLD_root):
        if file.endswith(self.suffix):
          print file
          self.file_list.append([(datetime.datetime.strptime(file,self.prefix + '-%Y%m%d%H%M%S.dat')),
                            (os.path.join(root,file))])
          #filepaths.append(os.path.join(root,file))
          #startdates.append(datetime.datetime.strptime(file,'GLD-%Y%m%d%H%M%S.dat'))
    
    self.file_list.sort(key=lambda tup: tup[0])

    #logging.info('Refreshed file list')
    #print file_list
  # def get_filename(self, t):
  #   '''returns a subdirectory string -- just here to avoid
  #     a ton of redundant string parsing'''
  #   folder = datetime.datetime.strftime(t,'%Y-%m-%d')
  #   files = os.listdir(os.path.join(self.GLD_root,folder))
  #   file_list = []
  #   for file in files:
  #    if file.endswith(self.suffix):
  #     print file
  #     file_list.append([(datetime.datetime.strptime(file,self.prefix + '-%Y%m%d%H%M%S.dat')),
  #                         (os.path.join(self.GLD_root,file))])

  #   file_list.sort(key=lambda tup: tup[0])
  #   return file_list[-1] # newest file



  def get_file_at(self,t):
    ''' t: datetime object
           Finds the last file in self.file_list with time less than t
    '''    
    #startfile = filter(lambda row: row[0] <= t, self.file_list)[-1]
    #startfiles = filter(lambda row: row[0] <= t, self.file_list)
    
    # if len(startfiles) > 0:
    #   return startfiles[-1]
    # else:
    #   logging.info("No files found!")
    #   return 
    folder = datetime.datetime.strftime(t,'%Y-%m-%d')
    
    if not os.path.exists(os.path.join(self.GLD_root,folder)):
      return None, None

    files = os.listdir(os.path.join(self.GLD_root,folder))
    file_list = []
    for file in files:
     if file.endswith(self.suffix) and file.startswith(self.prefix):
      #print file
      file_list.append([(datetime.datetime.strptime(file,self.prefix + '-%Y%m%d%H%M%S.dat')),
                          (os.path.join(self.GLD_root,folder,file))])

    file_list.sort(key=lambda tup: tup[0])
    #logging.info(file_list)

    # Select only files which start earlier than our target time:
    file_list = [f for f in file_list if t >= f[0]]

    return file_list[-1]  # Last one in the list
    # return file_list[-1] # newest file



  def load_flashes(self, t, dt = datetime.timedelta(0,0,0,0,1,0)):
    '''filepath: GLD file to sift thru
       t: datetime object to search around
       dt: datetime.timedelta 

       returns: A list of tuples: <time ob
    '''
    # print t
    # print type(t)
    
    filetime,filepath = self.get_file_at(t)
    
    tprev = t - dt

    rows =  []
    times = []
    if filetime is not None:
      # print "section 1"
      # return None, None
    # else:

      # print "t:",t
      # print "tprev:",tprev
      # print "filetime:",filetime
      #buff_size = 100000 # bytes
      # Binary search thru entries:
      imax = np.floor(os.path.getsize(filepath)).astype('int')
      imin = 0

      thefile = open(filepath,'r')

      
      # Find closest index to target time:
      # print "recursing t_ind"
      t_ind = self.recursive_search_kernel(thefile, t, imin, imax)
      # print t_ind
      # print self.datetime_from_row(self.parse_line(thefile,t_ind))
      
      # Find closest index to window time:
      # print "recursing t_prev"
      tprev_ind = self.recursive_search_kernel(thefile,tprev,imin,imax)
      # print tprev_ind
      # print self.datetime_from_row(self.parse_line(thefile,tprev_ind))


      # print "t_ind, tprev_ind (section 1)",t_ind, tprev_ind
      if (t_ind is not None) and (tprev_ind is not None):
        # return None, None
        # Load rows between tprev_ind and t_ind:
        while (thefile.tell() < t_ind):
          curr_line = self.parse_line(thefile,thefile.tell())

          newtime = self.datetime_from_row(curr_line)
          # datetime_from_row will return None if line is unhappy
          if newtime is not None:
            rows.append(curr_line)
            times.append(newtime)
      
      
      # In the case that tprev runs over the start of the file (asking for flashes at 12:01...)
      # This could be more-elegant but I'm just repeating the previous code
    print tprev, filetime
    if (filetime is None) or (tprev < filetime):
      # print "section 2"
      filetime,filepath = self.get_file_at(tprev)


      if filetime is not None:
        # print "doing overlap"
        imax = np.floor(os.path.getsize(filepath)).astype('int')
        imin = 0

        thefile = open(filepath,'r')
        t_ind = imax
        # Find closest index to window time:
        tprev_ind = self.recursive_search_kernel(thefile,tprev,imin,imax)
        #print self.datetime_from_row(self.parse_line(thefile,tprev_ind))
        
        # print "t_ind, t_prev:",t_ind, tprev_ind

        rows_prev = []
        times_prev= []
        # if (t_ind is None) or (tprev_ind is None):
        #   return None, None
        # Load rows between tprev_ind and t_ind:
        if (t_ind is not None) and (tprev_ind is not None):
          while (thefile.tell() < t_ind):
            curr_line = self.parse_line(thefile,thefile.tell())
            newtime = self.datetime_from_row(curr_line)

            # datetime_from_row will return None if line is unhappy

            if newtime is not None:
              rows_prev.append(curr_line)
              times_prev.append(newtime)

            # rows_prev.append(curr_line)
            # times_prev.append(self.datetime_from_row(curr_line))
        
        
        rows[0:0] = rows_prev
        times[0:0] = times_prev

    logging.info(" Found " + str(len(rows)) + " entries between " + str(tprev) + " and " + str(t))
    if len(rows) > 0:
      # The searching stuff mostly works, but has issues when the
      # files are out of order or contain duplicate entries. So here we'll mask off 
      # everything outside of our interval, just to be sure.
     
      # return np.asarray(rows), np.asarray(times)
      times = np.array(times)
      rows = np.array(rows)
      mask = ((times >=tprev) & (times <= t))
      return rows[mask], times[mask]
    else:
      return None, None
       
  def recursive_search_kernel(self, thefile, target_time, imin, imax, n= 0 ):
    ''' Recursively searches thefile (previously open) for the closest entry
        to target_time (datetime object)
    '''
    imid = imin + ((imax - imin)/2)
    #imid = ((imax-imin)/2)
    l = self.parse_line(thefile,imid)
    if l is None:
      return None
    # print l
    y,m,d,H,M,S = l[0:6].astype('int')
    curr_time = datetime.datetime(y,m,d,H,M,S)
    
    if n > 50: 
      print 'max recursions!'
      return None
    if abs(imin - imax) <= 200:
      # print n, imin, imax, imid, imax-imin, curr_time
      return imin
    else:
      if curr_time >= target_time:
        # print 'too high: ',imin, imax, imid, imax-imin, curr_time
        imax = imid
        #imax -=1
      else:
        # print 'too low: ',imin, imax, imid, imax-imin, curr_time
        imin = imid
        #imin += 1
      # Uncomment this to show recursion (hella sweet)  
      # print imin, imax, imax-imin, curr_time
      return self.recursive_search_kernel(thefile,target_time,imin,imax,n+1)
      
  def parse_line(self, thefile, theindex, n=0):
    '''
    Returns a parsed line; recursively skips forward if line isn't full-length
    '''
    thefile.seek(theindex,0)
    line = thefile.readline()
    vec = line.split('\t')

    # print "index:",theindex
    if n > 50:
      # print "failed to find an entry"
      logging.info("Failed to find an entry")
      return None

    if (len(vec)==25) and (vec[0] == '0'):
      # print vec[0]
      try:
        return np.array(vec[1:11],'float')
      except:
        return self.parse_line(thefile=thefile,theindex=thefile.tell(), n=(n+1))

    else:
      # if (thefile.tell() == theindex):
        # At end of file -- jump back a few lines, use that value
      # print vec  
      #   logging.info("Hit end of file hit -- jumping backwards " + str(n) + ' ' + str(theindex))
      # thefile.seek(thefile.tell() - 200,0) # Healthy line is ~96 long  

      # Something we would like to do: If we're less than a full line away
      # from the end of the file, back up and try for the whole line.
      # This would help on some of the broken days (multiple weird files).
      # For example: datetime(2015,2,8,6,0,0)  -- aps 3.2017

      # if thefile.tell() > thefile.totalsize - 96:
        # Scroll backward
      # thefile.seek(thefile.tell() - 200,0) # Healthy line is ~96 long    
      return self.parse_line(thefile=thefile,theindex=thefile.tell(), n=(n+1))

  def datetime_from_row(self, row):
    # print row
    y,m,d,H,M,S,n = row[0:7].astype('int')
    micros = n/1000
    try:
      return datetime.datetime(y,m,d,H,M,S,micros)
    except:
      return None



# ---------------------------
# Main block
# ---------------------------

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG,
                      format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                      )  

  GLD_root = 'alex/array/home/Vaisala/feed_data/GLD'
  t = datetime.datetime(2015,04,02,16,20,00)
  #startfile, startfile_time = get_file_at(t)
  #startfile ='alex/array/home/Vaisala/feed_data/GLD/2015-03-26/GLD-201503260223.dat'

  print 'initializing'

  G = GLD_file_tools(GLD_root)

  print 'doing'
  #G.load_flashes(t)

  for hours in xrange(0,24):
    for mins in xrange(0,60):
      f = G.load_flashes(datetime.datetime(2015,04,02, hours, mins,0))
    