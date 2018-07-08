import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter


LOG_DIR = "logs/"
PLOT_DIR = "plots/"


class QRSDetectorOffline(object):
   

    def __init__(self, ecg_data_path, verbose=True, log_data=False, plot_data=False, show_plot=False):
        """
        Metode inisialisasi kelas QRSDetectorOffline.
        : param string ecg_data_path: path ke dataset ECG
        : param bool verbose: flag untuk mencetak hasilnya
        : param bool log_data: flag untuk mencatat hasil
        : param bool plot_data: flag untuk plot hasil ke file
        : param bool show_plot: flag untuk menunjukkan plot hasil yang dihasilkan - tidak akan menampilkan apa pun jika plot tidak dihasilkan
        """
        # parameter yang digunakan.
        self.ecg_data_path = ecg_data_path

        self.signal_frequency = 250  # Set panjang frekuensi QRS disini
        self.filter_lowcut = 0.0
        self.filter_highcut = 30.0
        self.filter_order = 1

        self.integration_window = 15  

        self.findpeaks_limit = 0.35
        self.findpeaks_spacing = 50 

        self.refractory_period = 120  
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # load ecg data
        self.ecg_data_raw = None

        # nilai ukuran dan yang dihitung
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None
 	self.detected_peaks_values1 = None
	self.max = None
        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # hasil deteksi
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)

        # Final data ECG dan QRS detection array dan yang terdeteksi di beri tanda nilai 1
        self.ecg_data_detected = None

        # fungsi untuk menjalankan program.
        self.load_ecg_data()
        self.detect_peaks()
        self.detect_qrs()

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = "{:s}QRS_offline_detector_log_{:s}.csv".format(LOG_DIR,
                                                                           strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.log_detection_data()

        if plot_data:
            self.plot_path = "{:s}QRS_offline_detector_plot_{:s}.png".format(PLOT_DIR,
                                                                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
            self.plot_detection_data(show_plot=show_plot)

    """memproses metode ECG measurements data"""

    def load_ecg_data(self):
        """
        Buka file ECG.
        """
        self.ecg_data_raw = np.loadtxt(self.ecg_data_path, skiprows=1, delimiter=',')

    """ECG measurements preprocessing"""

    def detect_peaks(self):
        """
        Metode yang bertanggung jawab untuk mengekstraksi puncak dari data pengukuran ECG yang dimuat melalui pemrosesan pengukuran.
        """
        # Extract measurements dari data.
        ecg_measurements = self.ecg_data_raw[:, 1]

        # Measurements filtering - 0-15 Hz band pass filter.
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut, signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # Derivative - mencari kemiringan QRS
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)

        # Squaring - mengintensifkan nilai yang diterima dari derivative.
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # Moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, np.ones(self.integration_window))

        # Fiducial mark - deteksi puncak pada integrated measurements.
        self.detected_peaks_indices = self.findpeaks(data=self.integrated_ecg_measurements,
                                                     limit=self.findpeaks_limit,
                                                     spacing=self.findpeaks_spacing)
	print("%s/n",self.detected_peaks_indices)
	self.max = 1
		

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_indices]
	
    """metode untuk deteksi QRS."""

    def detect_qrs(self):
        """
       Metode yang bertanggung jawab untuk mengklasifikasikan pengukuran EKG yang terdeteksi sebagai puncak sebagai kompleks QRS (detak jantung).
        """
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):

            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # Setelah QRS kompleks di deteksi, dijeda 200ms sebelum yang berikutnya dapat dideteksi.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                # Puncak harus diklasifikasikan sebagai puncak noise atau puncak QRS.
                # Untuk diklasifikasikan sebagai puncak QRS, ia harus melebihi nilai ambang yang ditetapkan secara dinamis.
                if detected_peaks_value > self.threshold_value:
		    print(self.threshold_value)
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Sesuaikan nilai puncak QRS yang digunakan nanti untuk menetapkan QRS-noise threshold.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Sesuaikan nilai puncak noise yang digunakan nanti untuk pengaturan QRS-noise threshold.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Sesuaikan nilai ambang batas QRS-noise berdasarkan nilai QRS atau noise yang dideteksi sebelumnya
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Buat array yang berisi data pengukuran ECG input dan kolom indikasi deteksi QRS.
        # tandai deteksi QRS dengan kolom '1' di kolom log 'qrs_detected' ('0' sebaliknya).
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw[:, 1]), 1])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, 1)
	
    """Results reporting methods."""

    def print_detection_data(self):
        """
        fungsi untuk mencetak data
        """
        print("qrs peaks indices")
        print(self.qrs_peaks_indices)
        print("noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """
        fungsi untuk memasukan hasil data ke file
        """
        with open(self.log_path, "wb") as fin:
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """
      Metode untuk plot data
        """
        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices):
            axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw[:, 1], title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements, title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.qrs_peaks_indices)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:, 1], title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:, 1], indices=self.qrs_peaks_indices)

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    """Tools methods."""

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
        """
        
	Metode yang bertanggung jawab untuk membuat dan menerapkan filter Butterworth.
        : data param deque: data raw
        : param float lowcut: filter nilai frekuensi lowcut
        : param float highcut: filter nilai frekuensi highcut
        : param int signal_freq: frekuensi sinyal dalam sampel per detik (Hz)
        : param int filter_order: urutan filter
        : return array: data yang difilter
	sumber : https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        """
	

        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm dan implementasi.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
      Menemukan puncak dalam `data` yang memiliki lebar` spacing` dan> = `limit`.
        : param ndarray data: data
        : param float spacing: jarak minimum ke puncak berikutnya (harus 1 atau lebih)
        : param float limit: puncak harus memiliki nilai lebih besar atau sama
        : return array: mendeteksi index array
        """
        len1 = data.size
	panjang = None
	maks = 0
        x = np.zeros(len1 + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len1] = data
        peak_candidate = np.zeros(len1)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len1]  # before
            start = spacing
            h_c = x[start: start + len1]  # central
            start = spacing + s + 1
            h_a = x[start: start + len1]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
	print(ind)
        if limit is not None:
            ind = ind[data[ind] > limit]
	
	return ind
	
	

	


if __name__ == "__main__":
    qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/samples(1).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector1 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(2).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector2 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(3).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector4 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(5).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector5 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(6).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector6 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(7).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=True)
    qrs_detector7 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(8).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector8 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(9).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector9 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(10).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector10 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(11).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector11 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(12).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector12 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(13).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector13 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(14).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector14 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(15).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector15 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(16).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector16 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(17).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector17 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(18).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector18 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(19).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    qrs_detector19 = QRSDetectorOffline(ecg_data_path="ecg_data/samples(20).csv", verbose=True,
                                      log_data=True, plot_data=True, show_plot=False)
    
    
