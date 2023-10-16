#Copyright 2023 University Of Houston
"""THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

#import statements
import numpy as np
import csv
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt


#Function defininition###################################################################################################################################
def extract_Spikes(continuous_arr, time_arr, threshold, pre_window_mus=400, post_window_mus=800, fs=40000, deadtime_mus=800):
    #this function takes a vector and thresholding parameters and returns the spike timestamps and voltages for that data
    #continuous_arr: numpy array of continuous voltage data from which spikes are extracted
    #time_arr: numpy array of timestamps for the continuous data, must be the same size
    #threshold: threshold value to use for spike detection (spikes are points above this threshold)
    #pre_window_mus: time extracted before threshold crossing in microseconds, default 400 mus
    #post_window_mus: time extracted after threshold crossing in microseconds , default 800 mus
    #fs: sampling frequency, default 40kHz
    #deadtime: time in microseconds after which a new spike cannot be detected, default 800 mus

    #convert from microseconds to samples
    s2mus = 1000000
    mus2s = 1/ s2mus
    deadtime = int(deadtime_mus * mus2s*fs) #dead time in indices 
    pre_window = int(pre_window_mus * mus2s *fs)
    post_window = int(post_window_mus * mus2s *fs)


    #find the spike indices
    spike_idxs, spike_vals = scipy.signal.find_peaks(continuous_arr, height=threshold, distance=deadtime)

    #initialize some dicts to hold our spikes
    spike_vals_dict = dict()
    spike_timestamps_dict = dict()
    for cidx, crossing in enumerate(spike_idxs):
        if crossing-pre_window < 0 or crossing + pre_window > continuous_arr.size:
            #if our spike is too close to the beginning or end of the recording we'll ignore it
            continue
        else:
            #deadtime was enforced by find_peaks, so we can simply slice here
            spbegin = crossing-pre_window
            spend = crossing+post_window
            spike_vals_dict["Spike_" + str(cidx) + "_values"] = continuous_arr[spbegin:spend]
            spike_timestamps_dict["Spike_" + str(cidx) + "_times"] = time_arr[spbegin:spend]
    
    #RETURNS 2 dictionaries, each entry is one spike. The values dict contains the voltages of each spike while the timestamps dict contains the raw times
    return spike_vals_dict, spike_timestamps_dict
#########################################################################################################################################################



#Main Code###############################################################################################################################################

#filename variables
directory = "~/py/pygammasignal/"
nexfile = directory + "exp2_phe_.1ml_800_.9%_continous.csv" #main input csv
cardiofile = directory + "exp2_cardiovascular_variables_simplified.csv"

#importing the data
#which channel we use is determined by the list of column names from the file. Multiple columns could be processed by iteration
data_column_list = list(["Phe2_02_values"]) #for channel 3
time_column_list = list(["Phe2_02_timestamps"])
continuous_neural_df = pd.read_csv(nexfile, usecols=data_column_list)
timeseries = pd.read_csv(nexfile, usecols=time_column_list)

#This line will also read in all of the cardiovascular data
cardiovacular_df = pd.read_csv(cardiofile)

#for the numpy approach we need the timeseries and continuous data as arrays
#we'll iterate for each column in the column list
for datacol, timecol in zip(data_column_list, time_column_list):
    #next we'll convert that columns from pandas df into np array
    col_voltage = np.asarray(continuous_neural_df[datacol])
    col_times = np.asarray(timeseries[timecol]) 

    #we'll set the threshold at 3 standard deviations above the mean voltage
    col_thresh = 3 * np.std(col_voltage) #the threshold

    #now we'll call our function, using the default values for everything except the arrays and the threshold
    col_spikes, col_times = extract_Spikes(col_voltage, col_times, col_thresh)

    #finally we'll save to file
    spikes_df = pd.DataFrame.from_dict(col_spikes, orient='index')
    spikes_df.to_csv(directory + datacol +'_spikes.csv',index = False) #main file

    time_df = pd.DataFrame.from_dict(col_times, orient='index')
    time_df.to_csv(directory + datacol +'_spiketimes.csv', index = False) #timestamps, not using
    
print("Done")
