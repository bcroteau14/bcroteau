
import numpy as np
import pandas as pd
# from api_sensordata import filter_emi


class PVT:
    '''
    Takes an NP2 NRR dataframe and extracts PVT information from it. Saves it to this class.
    This class assumes 3 minute PVT with 355ms lapse threshold.
    '''
    false_start_threshold = 0.1
    lapse_threshold = 0.355
    
    def __init__(self, nrr_df):

        nrr_df['reaction'] = pd.to_numeric(nrr_df['reaction'], errors = 'coerce')

        lapses = 0
        total_lapses = 0
        false_starts = 0
        reaction_times = []
        self.trials = nrr_df['trail'].nunique() - 1

        for trial, trial_df in nrr_df.groupby('trail', sort = False):
            # total lapses are lapses
            if trial_df.shape[0] == 1:
                total_lapses += 1
                rt = 5.0
            else:
                rt = trial_df['reaction'].iloc[1] / 1000.0
                if np.isnan(rt):
                    # subject left their figer on the the screen from the previous trial, this trial is fault and should be excluded from the experiment
                    self.trials -= 1
                    continue

            if rt < self.false_start_threshold:
                false_starts += 1
            else:
                # false starts don't contribute to reaction times
                reaction_times.append(rt)

            if rt > self.lapse_threshold:
                lapses += 1

        self.false_starts = false_starts
        self.lapses = lapses
        self.reaction_times = np.array(reaction_times)
        self.mean_rt = self.reaction_times.mean()
        self.coeff_of_var_rt = self.coeffcient_of_var(self.reaction_times)
        
        self.performance = self.get_performance()
       

    def get_performance(self):
        valid_stimuli = self.trials - self.false_starts
        performance = 1.0 - ((self.lapses + self.false_starts) / (valid_stimuli + self.false_starts))
        return performance

    def coeffcient_of_var(self, array):
        return np.std(array, ddof=0) / np.mean(array)
    
    def __str__(self):
        return f"FS: {self.false_starts}, Lapses: {self.lapses}, Mean RT: {self.mean_rt}, Performance: {self.performance}, Coeff of Var (RT): {self.coeff_of_var_rt}"
    
    def as_dict(self):
        return {
            'performance': self.performance,
            'mean_rt': self.mean_rt,
            'lapses': self.lapses,
            'false_starts': self.false_starts,
            'coeff_of_var_rt': self.coeff_of_var_rt,
            'trials': self.trials
        }





required_columns = ['ch0_raw', 'ch1_raw', 'ch2_raw', 'ch3_raw', 'ch4_raw', 'timestamp', 'label', 'rep', 'session_id', 'user_id', 'protocol']
channel_names = ['ch0_hp', 'ch1_hp', 'ch2_hp', 'ch3_hp', 'ch4_hp']

def preprocess_emg_df(emg_df):
    # keep subset of columns, rename label and filter
    emg_df = emg_df[required_columns]
    emg_df = emg_df.rename(columns={'label': 'gesture_posture'})
    emg_df = filter_emi(emg_df, channel_names, sampling_frequency=1000, use_lfilter_zi=True)
    return emg_df


def generate_np2_session_idf(np2_idf, pvt, kss):
    # Generate np2 idf for one seession of data
    
    missing_columns = [col for col in required_columns if col not in np2_idf.columns]
    if missing_columns:
        print(f"Not including: Following columns were not found in the session dataframe: {', '.join(missing_columns)}")
        return None

    # preprocess np2_idf
    np2_idf = preprocess_emg_df(np2_idf)
    
    # drop @pison.com
    np2_idf['subject_id'] = np2_idf['user_id'].iloc[0].split('@')[0]
    np2_idf.drop('user_id', axis=1, inplace=True)
    
    # fill in remaining columns with data from this session
    np2_idf['pvt_performance'] = pvt.performance
    np2_idf['pvt_lapses'] = pvt.lapses
    np2_idf['pvt_RT'] = pvt.mean_rt
    
    np2_idf['kss'] = kss
    
    return np2_idf


def get_np_blocking(idf, max_ble_gap):
    # standard blocking (increments when the below columns change or when time delta > max_ble_gap)
    idf_block = idf.timestamp.diff().transform(lambda x: np.where((np.abs(x) > max_ble_gap) | (x.isna()), 1, 0) | 
                                     (idf['subject_id'] != idf['subject_id'].shift(1)) |
                                     (idf['session_id'] != idf['session_id'].shift(1)) |
                                   (idf['gesture_posture'] != idf['gesture_posture'].shift(1)) |
                                   (idf['protocol'].fillna(0) != idf['protocol'].fillna(0).shift(1)) |
                                   (idf['rep'].fillna(0) != idf['rep'].fillna(0).shift(1)))
    idf_block = idf_block.cumsum()
    return idf_block


def update_blocking(idf, max_ble_gap):
    idf = idf.sort_values(['subject_id', 'session_id', 'timestamp'])
    idf['block'] = get_np_blocking(idf, max_ble_gap)
    idf = idf.reset_index(drop=True)
    return idf
